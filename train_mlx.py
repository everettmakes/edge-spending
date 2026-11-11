"""Fine-tune (or zero-shot evaluate) a small LLM on Apple Silicon with MLX, logging to MLflow.

Every run is scored on two test sets: "seen" (training merchants) and "unseen" (new merchants and
line formats). Unseen metrics are logged with an "unseen_" prefix.

Examples:
  python train_mlx.py --run-name baseline --zero-shot
  python train_mlx.py --run-name legacy-format --template legacy
  python train_mlx.py --run-name r8-attn
  python train_mlx.py --run-name r16-attn-mlp --rank 16 --targets attn_mlp

  # recipe ablation: old recipe, same test sets
  python train_mlx.py --run-name r8-v2recipe --mlx-data mlx_v2recipe

  # size ladder: same recipe, bigger and smaller base models (see MODELS in common.py)
  python train_mlx.py --run-name qwen-3b-r8 --base-model qwen-3b-4bit
"""
#----------IMPORTS----------#
import argparse
import json
import math
import shutil
import time
import types
from pathlib import Path

import mlflow
import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import generate, load
from mlx_lm.lora import CONFIG_DEFAULTS, train_model
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.datasets import load_dataset

from common import EXPERIMENT, MODELS, SPLITS, build_messages, load_jsonl, metric_prefix, score


#----------CONFIG----------#

#-----lora_target_layers-----#
# attn = attention projections only; attn_mlp also adapts the feed-forward (MLP) layers.
TARGETS = {
    "attn": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
    "attn_mlp": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                 "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
}



#----------TRAINING----------#

#-----log_training_to_mlflow-----#
class MLflowCallback(TrainingCallback):
    """Forward MLX's training/validation reports to MLflow."""

    def __init__(self):
        self.peak_memory_gb = 0.0
        self.val_losses = {}  # iteration -> val loss

    def on_train_loss_report(self, info: dict):
        step = info["iteration"]
        # stop straight away if training blows up, instead of wasting the rest of the run.
        if math.isnan(info["train_loss"]):
            raise RuntimeError(f"Training loss is NaN at step {step}: try --dtype bfloat16 (float16 overflow).")
        mlflow.log_metrics({"train_loss": info["train_loss"],
                            "tokens_per_second_train": info["tokens_per_second"]}, step=step)
        self.peak_memory_gb = max(self.peak_memory_gb, info["peak_memory"])

    def on_val_loss_report(self, info: dict):
        mlflow.log_metric("val_loss", info["val_loss"], step=info["iteration"])
        self.val_losses[info["iteration"]] = info["val_loss"]



#----------EVALUATION----------#

#-----generate_and_score-----#
# greedy decoding with the chat template, exactly like deployment.
def evaluate(model, tokenizer, rows: list[dict], max_tokens: int = 64) -> tuple[dict, list[str]]:
    """Greedy-generate a reply per test line with the chat template, then score."""
    replies = []
    for row in rows:
        prompt = tokenizer.apply_chat_template(build_messages(row["line"]), tokenize=False,
                                               add_generation_prompt=True)
        replies.append(generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False))
    return score(replies, [r["label"] for r in rows]), replies



#----------CHECKPOINTS----------#

#-----early_stopping_after_the_fact-----#
def keep_best_checkpoint(run_dir: Path, val_losses: dict, every: int) -> int:
    """Early stopping after the fact: copy the checkpoint with the lowest val loss over adapters.safetensors.
    Validation uses held-out merchants, so this picks the point before the model starts memorising."""
    # MLX reports validation at iteration N-1 and saves the checkpoint at N (e.g. val 49 -> 0000050_*)
    saved = {}
    for it, loss in val_losses.items():
        for ckpt in (it, it + 1):
            if ckpt > 0 and ckpt % every == 0 and (run_dir / f"{ckpt:07d}_adapters.safetensors").exists():
                saved[ckpt] = loss
    if not saved:
        print("No checkpoint matched a validation step: keeping the final adapters.")
        return -1
    best = min(saved, key=saved.get)
    shutil.copy(run_dir / f"{best:07d}_adapters.safetensors", run_dir / "adapters.safetensors")
    mlflow.log_metrics({"best_iter": best, "best_val_loss": saved[best]})
    print(f"Keeping checkpoint from iteration {best} (val loss {saved[best]:.4f})")
    return best



#----------NUMERICS----------#

#-----set_dtype-----#
# MLX's 4-bit models compute in float16 (max ~65,504). Qwen2.5-1.5B's activations overflow that and the loss
# goes NaN. bfloat16 has the same 16 bits but float32's exponent range, so it doesn't overflow.
def set_dtype(model, dtype: str) -> None:
    if dtype != "native":
        model.set_dtype(getattr(mx, dtype))  # casts float params (incl. quantisation scales); packed 4-bit ints untouched


#----------FINE_TUNING----------#

#-----one_lora_run-----#
def train(args, base_model: str, run_dir: Path) -> float:
    """Train LoRA adapters into run_dir, keep the best checkpoint, and return MLX's peak memory in GB."""
    #-----mlx_lm_config-----#
    config = dict(CONFIG_DEFAULTS)
    config.update({
        "model": base_model, "train": True, "fine_tune_type": "lora",
        "data": f"{args.mlx_data}/{args.template}", "adapter_path": str(run_dir),
        "num_layers": -1,                      # add adapters to every transformer layer
        "iters": args.iters, "batch_size": args.batch_size, "learning_rate": args.lr,
        "lora_parameters": {"rank": args.rank, "scale": args.scale, "dropout": 0.0,
                            "keys": TARGETS[args.targets]},
        "mask_prompt": args.template == "chat",  # loss on answer tokens only (not possible for plain text)
        "steps_per_report": 10, "steps_per_eval": args.eval_every, "val_batches": -1,
        "save_every": args.eval_every,         # a checkpoint at every validation, so the best can be kept
        "max_seq_length": 256, "seed": args.seed,
    })
    config = types.SimpleNamespace(**config)

    # train_model doesn't seed anything itself: LoRA init uses mx.random, batch shuffling uses numpy
    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    #-----load_model_and_data-----#
    model, tokenizer = load(base_model)
    set_dtype(model, args.dtype)
    train_set, valid_set, _ = load_dataset(config, tokenizer)
    #-----train-----#
    callback = MLflowCallback()
    train_model(config, model, train_set, valid_set, training_callback=callback)

    #-----report_and_keep_best-----#
    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    mlflow.log_metric("trainable_params", trainable)
    if not args.keep_last:
        keep_best_checkpoint(run_dir, callback.val_losses, args.eval_every)
    return callback.peak_memory_gb



#----------MAIN----------#
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--zero-shot", action="store_true", help="evaluate the base model without training")
    parser.add_argument("--base-model", default="qwen-0.5b-4bit",
                        help=f"one of {list(MODELS)} or any MLX / Hugging Face repo id")
    parser.add_argument("--template", choices=["chat", "legacy"], default="chat")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--scale", type=float, default=2.0, help="LoRA multiplier (= alpha / rank in PEFT terms)")
    parser.add_argument("--targets", choices=list(TARGETS), default="attn")
    parser.add_argument("--iters", type=int, default=300, help="2 epochs of 1200 rows at batch 8")
    parser.add_argument("--eval-every", type=int, default=50, help="validate + checkpoint every N iterations")
    parser.add_argument("--keep-last", action="store_true", help="use the final checkpoint, not the best val loss")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=0, help="rerun a config with seeds 0, 1, 2 to measure training noise")
    parser.add_argument("--adapter-dir", default="adapters")
    parser.add_argument("--mlx-data", default="mlx_data", help="training data folder from prepare_mlx_data.py")
    parser.add_argument("--dtype", choices=["native", "bfloat16", "float16", "float32"], default="native",
                        help="compute dtype; use bfloat16 if the loss goes NaN (float16 overflow)")
    args = parser.parse_args()

    #-----resolve_model_and_test_sets-----#
    base_model = MODELS.get(args.base_model, args.base_model)
    splits = {name: load_jsonl(path) for name, path in SPLITS.items() if Path(path).exists()}
    if "unseen" not in splits:
        print("No data/test_unseen.jsonl yet: run `python make_data.py` to create it. Scoring seen split only.")

    #-----start_mlflow_run-----#
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=args.run_name):
        mlflow.set_tags({"stage": "baseline" if args.zero_shot else "train",
                         "base_model": base_model, "device": "apple-silicon-mlx"})
        mlflow.log_params({"zero_shot": args.zero_shot, "base_model_name": args.base_model,
                           **{f"n_test_{name}": len(rows) for name, rows in splits.items()}})

        #-----train_unless_zero_shot-----#
        adapter_path = None
        if not args.zero_shot:
            mlflow.log_params({"template": args.template, "rank": args.rank, "scale": args.scale,
                               "targets": args.targets, "iters": args.iters, "seed": args.seed,
                               "checkpoint": "last" if args.keep_last else "best_val",
                               "mlx_data": args.mlx_data, "dtype": args.dtype,
                               "batch_size": args.batch_size, "lr": args.lr})
            run_dir = Path(args.adapter_dir) / args.run_name
            start = time.perf_counter()
            peak_gb = train(args, base_model, run_dir)
            mlflow.log_metrics({"train_time_s": time.perf_counter() - start, "peak_mem_mb": peak_gb * 1000})
            mlflow.log_param("adapter_path", str(run_dir))
            adapter_path = str(run_dir)

        #-----evaluate_every_split-----#
        # reload with the kept adapter, then score seen, unseen (and real if present).
        model, tokenizer = load(base_model, adapter_path=adapter_path)
        set_dtype(model, args.dtype)
        out_dir = Path("predictions") / args.run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        all_metrics = {}
        for name, rows in splits.items():
            start = time.perf_counter()
            metrics, replies = evaluate(model, tokenizer, rows)
            metrics["eval_time_s"] = time.perf_counter() - start
            all_metrics.update({metric_prefix(name) + k: v for k, v in metrics.items()})

            path = out_dir / f"{name}.jsonl"
            with open(path, "w") as f:
                for row, reply in zip(rows, replies):
                    f.write(json.dumps({"line": row["line"], "label": row["label"], "reply": reply}) + "\n")
            mlflow.log_artifact(str(path), artifact_path="predictions")

        mlflow.log_metrics(all_metrics)
        print(json.dumps({k: round(v, 4) for k, v in all_metrics.items()}, indent=2))


if __name__ == "__main__":
    main()
