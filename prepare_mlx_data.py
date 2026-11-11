"""Convert data/*.jsonl into the layouts mlx_lm expects: train.jsonl, valid.jsonl, test.jsonl.

  mlx_data/chat/    {"messages": [...]}   -> MLX applies the model's own chat template (the fix)
  mlx_data/legacy/  {"text": "### Instruction ..."}  -> the tutorial's mismatched format

Validation uses data/valid.jsonl: merchants that are in neither training nor the unseen test, so val_loss
tracks generalisation and train_mlx.py can keep the best checkpoint.

Training prompts list the categories in a random order, so the model has to read the list rather than
memorise it (this is what lets a user add their own category later). Validation and test keep the fixed order.

Usage: python prepare_mlx_data.py
"""
#----------IMPORTS----------#
import argparse
import json
import random
from pathlib import Path

from common import CATEGORIES, build_messages, legacy_text, load_jsonl


#----------HELPERS----------#

#-----write-----#
def write(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


#----------MAIN----------#
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="mlx_data")
    parser.add_argument("--no-shuffle-categories", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data = {name: load_jsonl(f"{args.data_dir}/{name}.jsonl") for name in ["train", "valid", "test"]}

    def categories(split: str) -> list[str]:
        if split != "train" or args.no_shuffle_categories:
            return CATEGORIES
        return rng.sample(CATEGORIES, len(CATEGORIES))

    out = Path(args.out)
    for split, rows in data.items():
        cats = [categories(split) for _ in rows]
        write([{"messages": build_messages(r["line"], r["label"], c)} for r, c in zip(rows, cats)],
              out / "chat" / f"{split}.jsonl")
        write([{"text": legacy_text(r["line"], r["label"], c)} for r, c in zip(rows, cats)],
              out / "legacy" / f"{split}.jsonl")

    print(", ".join(f"{k}={len(v)}" for k, v in data.items()), f"-> {out}/chat and {out}/legacy")


if __name__ == "__main__":
    main()
