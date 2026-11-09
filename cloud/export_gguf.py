"""Merge a trained LoRA adapter into the base model and export GGUF files at several quantisation levels.

Run in Colab (GPU):  python export_gguf.py --adapter adapters/r8-attn --out gguf
"""
import unsloth  # noqa: F401

import argparse
from pathlib import Path

from unsloth import FastLanguageModel

QUANTS = ["q8_0", "q4_k_m", "q4_0"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="folder saved by train.py")
    parser.add_argument("--out", default="gguf")
    parser.add_argument("--quants", nargs="+", default=QUANTS)
    args = parser.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(args.adapter, max_seq_length=256, load_in_4bit=True)
    for quant in args.quants:
        out_dir = Path(args.out) / quant
        print(f"Exporting {quant} -> {out_dir}")
        model.save_pretrained_gguf(str(out_dir), tokenizer, quantization_method=quant)

    for path in sorted(Path(args.out).rglob("*.gguf")):
        print(f"{path}  {path.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
