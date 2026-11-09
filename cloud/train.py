"""Fine-tune (or zero-shot evaluate) Qwen2.5-0.5B on the transaction task, logging to MLflow.

Run in Colab (GPU). Examples:
  python train.py --run-name baseline --zero-shot
  python train.py --run-name legacy-format --template legacy
  python train.py --run-name r8-attn
  python train.py --run-name r16-attn-mlp --rank 16 --alpha 32 --targets attn_mlp
"""
import unsloth  # noqa: F401  (must be imported before transformers / trl)

import argparse
import json
import os
import time
from pathlib import Path

import mlflow
import torch
from datasets import Dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

from common import EXPERIMENT, build_messages, legacy_text, load_jsonl, score

BASE_MODEL = "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit"
TARGETS = {
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "attn_mlp": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


class MLflowLossCallback(TrainerCallback):
    """Send the trainer's loss to MLflow at every logging step."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            mlflow.log_metric("train_loss", logs["loss"], step=state.global_step)


def training_texts(rows: list[dict], tokenizer, template: str) -> list[str]:
    """Training strings: Qwen chat template (fixed) or the tutorial's ### format (legacy)."""
    if template == "legacy":
        return [legacy_text(r["line"], r["label"]) for r in rows]
    return [tokenizer.apply_chat_template(build_messages(r["line"], r["label"]), tokenize=False) for r in rows]


@torch.no_grad()
def generate_replies(model, tokenizer, lines: list[str], batch_size: int = 16, max_new_tokens: int = 64) -> list[str]:
    """Greedy replies for each line, always prompted with the chat template (as at inference)."""
    tokenizer.padding_side = "left"  # left-pad so every prompt ends right where generation starts
    replies = []
    for i in range(0, len(lines), batch_size):
        prompts = [
            tokenizer.apply_chat_template(build_messages(line), tokenize=False, add_generation_prompt=True)
            for line in lines[i:i + batch_size]
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id)
        new_tokens = output[:, inputs["input_ids"].shape[1]:]
        replies += tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return replies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--zero-shot", action="store_true", help="evaluate the base model without training")
    parser.add_argument("--template", choices=["chat", "legacy"], default="chat")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--targets", choices=list(TARGETS), default="attn")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--adapter-dir", default=os.environ.get("ADAPTER_DIR", "adapters"))
    args = parser.parse_args()

    torch.manual_seed(0)
    train_rows = load_jsonl(f"{args.data_dir}/train.jsonl")
    test_rows = load_jsonl(f"{args.data_dir}/test.jsonl")

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=args.run_name):
        mlflow.set_tags({"stage": "baseline" if args.zero_shot else "train", "base_model": BASE_MODEL})
        mlflow.log_params({"n_train": len(train_rows), "n_test": len(test_rows), "zero_shot": args.zero_shot})

        model, tokenizer = FastLanguageModel.from_pretrained(
            BASE_MODEL, max_seq_length=args.max_seq_length, load_in_4bit=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if not args.zero_shot:
            model = FastLanguageModel.get_peft_model(
                model, r=args.rank, lora_alpha=args.alpha, target_modules=TARGETS[args.targets],
                lora_dropout=0, bias="none", random_state=0,
            )
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            mlflow.log_params({
                "template": args.template, "rank": args.rank, "lora_alpha": args.alpha,
                "targets": args.targets, "epochs": args.epochs, "batch_size": args.batch_size,
                "lr": args.lr, "max_seq_length": args.max_seq_length,
            })
            mlflow.log_metric("trainable_params", trainable)

            texts = training_texts(train_rows, tokenizer, args.template)
            mlflow.log_text(texts[0], "example_training_text.txt")
            config = SFTConfig(
                output_dir="sft_out", per_device_train_batch_size=args.batch_size,
                num_train_epochs=args.epochs, learning_rate=args.lr, logging_steps=5,
                optim="adamw_8bit", save_strategy="no", report_to="none", seed=0,
                bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
                dataset_text_field="text", max_seq_length=args.max_seq_length, packing=False,
            )
            trainer = SFTTrainer(model=model, processing_class=tokenizer, args=config,
                                 train_dataset=Dataset.from_dict({"text": texts}),
                                 callbacks=[MLflowLossCallback()])

            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            result = trainer.train()
            mlflow.log_metrics({
                "train_time_s": time.perf_counter() - start,
                "final_train_loss": result.training_loss,
                "peak_gpu_mem_mb": torch.cuda.max_memory_allocated() / 1e6,
            })

        FastLanguageModel.for_inference(model)
        start = time.perf_counter()
        replies = generate_replies(model, tokenizer, [r["line"] for r in test_rows])
        metrics = score(replies, [r["label"] for r in test_rows])
        metrics["eval_time_s"] = time.perf_counter() - start
        mlflow.log_metrics(metrics)

        preds_path = Path("predictions.jsonl")
        with open(preds_path, "w") as f:
            for row, reply in zip(test_rows, replies):
                f.write(json.dumps({"line": row["line"], "label": row["label"], "reply": reply}) + "\n")
        mlflow.log_artifact(str(preds_path))

        if not args.zero_shot:
            out = Path(args.adapter_dir) / args.run_name
            model.save_pretrained(out)
            tokenizer.save_pretrained(out)
            mlflow.log_param("adapter_path", str(out))

        print(json.dumps({k: round(v, 4) for k, v in metrics.items()}, indent=2))


if __name__ == "__main__":
    main()
