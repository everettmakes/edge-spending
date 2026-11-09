"""Convert data/*.jsonl into the layouts mlx_lm expects: train.jsonl, valid.jsonl, test.jsonl.

  mlx_data/chat/    {"messages": [...]}   -> MLX applies Qwen's chat template (the fix)
  mlx_data/legacy/  {"text": "### Instruction ..."}  -> the tutorial's mismatched format

Usage: python prepare_mlx_data.py --valid 60
"""
import argparse
import json
from pathlib import Path

from common import build_messages, legacy_text, load_jsonl


def write(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="mlx_data")
    parser.add_argument("--valid", type=int, default=60, help="rows held out of train for validation loss")
    args = parser.parse_args()

    train = load_jsonl(f"{args.data_dir}/train.jsonl")
    test = load_jsonl(f"{args.data_dir}/test.jsonl")
    train, valid = train[: -args.valid], train[-args.valid:]

    chat = lambda rows: [{"messages": build_messages(r["line"], r["label"])} for r in rows]  # noqa: E731
    legacy = lambda rows: [{"text": legacy_text(r["line"], r["label"])} for r in rows]  # noqa: E731

    out = Path(args.out)
    for name, rows in [("train", train), ("valid", valid), ("test", test)]:
        write(chat(rows), out / "chat" / f"{name}.jsonl")
        write(legacy(rows), out / "legacy" / f"{name}.jsonl")

    print(f"train={len(train)} valid={len(valid)} test={len(test)} -> {out}/chat and {out}/legacy")


if __name__ == "__main__":
    main()
