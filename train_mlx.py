"""Fine-tune (or zero-shot evaluate) Qwen2.5-0.5B on Apple Silicon with MLX, logging to MLflow.

Examples:
  python train_mlx.py --run-name baseline --zero-shot
  python train_mlx.py --run-name legacy-format --template legacy
  python train_mlx.py --run-name r8-attn
  python train_mlx.py --run-name r16-attn-mlp --rank 16 --targets attn_mlp
"""
import argparse
import json
import time
import types
from pathlib import Path

import mlflow
from mlx.utils import tree_flatten
from mlx_lm import generate, load
from mlx_lm.lora import CONFIG_DEFAULTS, train_model
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.datasets import load_dataset

from common import EXPERIMENT, build_messages, load_jsonl, score

BASE_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"  # 4-bit base -> QLoRA
TARGETS = {
    "attn": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
    "attn_mlp": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                 "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
}


class MLflowCallback(TrainingCallback):
    """Forward MLX's training/validation reports to MLflow."""

    def __init__(self):
        self.peak_memory_gb = 0.0

    def on_train_loss_report(self, info: dict):
        step = info["iteration"]
        mlflow.log_metrics({"train_loss": info["train_loss"],
                            "tokens_per_second_train": info["tokens_per_second"]}, step=step)
        self.peak_memory_gb = max(self.peak_memory_gb, info["peak_memory"])

    def on_val_loss_report(self, info: dict):
        mlflow.log_metric("val_loss", info["val_loss"], step=info["iteration"])


def evaluate(model, tokenizer, rows: list[dict], max_tokens: int = 64) -> tuple[dict, list[str]]:
    """Greedy-generate a reply per test line with the chat template, then score."""
    replies = []
    for row in rows:
        prompt = tokenizer.apply_chat_template(build_messages(row["line"]), tokenize=False,
                                               add_generation_prompt=True)
        replies.append(generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False))
    return score(replies, [r["label"] for r in rows]), replies


def train(args, run_dir: Path) -> float:
    """Train LoRA adapters into run_dir and return MLX's peak memory in GB."""
    config = dict(CONFIG_DEFAULTS)
    config.update({
        "model": BASE_MODEL, "train": True, "fine_tune_type": "lora",
        "data": f"mlx_data/{args.template}", "adapter_path": str(run_dir),
        "num_layers": -1,                      # add adapters to every transformer layer
        "iters": args.iters, "batch_size": args.batch_size, "learning_rate": args.lr,
        "lora_parameters": {"rank": args.rank, "scale": args.scale, "dropout": 0.0,
                            "keys": TARGETS[args.targets]},
        "mask_prompt": args.template == "chat",  # loss on answer tokens only (not possible for plain text)
        "steps_per_report": 10, "steps_per_eval": 50, "val_batches": -1,
        "save_every": args.iters, "max_seq_length": 256, "seed": 0,
    })
    config = types.SimpleNamespace(**config)

    model, tokenizer = load(BASE_MODEL)
    train_set, valid_set, _ = load_dataset(config, tokenizer)
    callback = MLflowCallback()
    train_model(config, model, train_set, valid_set, training_callback=callback)

    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    mlflow.log_metric("trainable_params", trainable)
    return callback.peak_memory_gb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--zero-shot", action="store_true", help="evaluate the base model without training")
    parser.add_argument("--template", choices=["chat", "legacy"], default="chat")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--scale", type=float, default=2.0, help="LoRA multiplier (= alpha / rank in PEFT terms)")
    parser.add_argument("--targets", choices=list(TARGETS), default="attn")
    parser.add_argument("--iters", type=int, default=150, help="~2 epochs of 540 rows at batch 8")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--adapter-dir", default="adapters")
    args = parser.parse_args()

    test_rows = load_jsonl("data/test.jsonl")
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=args.run_name):
        mlflow.set_tags({"stage": "baseline" if args.zero_shot else "train",
                         "base_model": BASE_MODEL, "device": "apple-silicon-mlx"})
        mlflow.log_params({"zero_shot": args.zero_shot, "n_test": len(test_rows)})

        adapter_path = None
        if not args.zero_shot:
            mlflow.log_params({"template": args.template, "rank": args.rank, "scale": args.scale,
                               "targets": args.targets, "iters": args.iters,
                               "batch_size": args.batch_size, "lr": args.lr})
            run_dir = Path(args.adapter_dir) / args.run_name
            start = time.perf_counter()
            peak_gb = train(args, run_dir)
            mlflow.log_metrics({"train_time_s": time.perf_counter() - start, "peak_mem_mb": peak_gb * 1000})
            mlflow.log_param("adapter_path", str(run_dir))
            adapter_path = str(run_dir)

        model, tokenizer = load(BASE_MODEL, adapter_path=adapter_path)
        start = time.perf_counter()
        metrics, replies = evaluate(model, tokenizer, test_rows)
        metrics["eval_time_s"] = time.perf_counter() - start
        mlflow.log_metrics(metrics)

        with open("predictions.jsonl", "w") as f:
            for row, reply in zip(test_rows, replies):
                f.write(json.dumps({"line": row["line"], "label": row["label"], "reply": reply}) + "\n")
        mlflow.log_artifact("predictions.jsonl")
        print(json.dumps({k: round(v, 4) for k, v in metrics.items()}, indent=2))


if __name__ == "__main__":
    main()
