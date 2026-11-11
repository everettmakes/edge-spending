"""Build docs/index.html: an interactive results site for GitHub Pages, from your real runs.

Reads predictions/<run>/<split>.jsonl (written by train_mlx.py) and mlflow.db, and writes one
self-contained HTML page: headline numbers, a results chart, a browser for every prediction,
a category confusion heatmap and the story of what didn't work.

Usage: python make_site.py        # then open docs/index.html, or push and enable GitHub Pages on /docs
"""

#----------IMPORTS----------#
import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

from common import CATEGORIES, parse_json, score_one

#----------CONFIG----------#
PRED_DIR = Path("predictions")
DB_PATH = Path("mlflow.db")
OUT_PATH = Path("docs/index.html")

# friendly names for runs; anything not listed keeps its run name.
LABELS = {
    "qwen-0.5b-zeroshot": "0.5B zero-shot",
    "qwen-0.5b-v2recipe": "0.5B, old recipe",
    "qwen-0.5b-r8": "0.5B, new recipe",
    "qwen-1.5b-zeroshot": "1.5B zero-shot",
    "qwen-1.5b-r8": "1.5B, new recipe",
    "qwen-3b-zeroshot": "3B zero-shot",
    "qwen-3b-r8": "3B, new recipe",
    "baseline": "Zero-shot",
    "legacy-format": "Legacy prompt format",
    "r8-attn": "Attention LoRA r8",
    "r16-attn-mlp": "Attention+MLP LoRA r16",
}

# display order inside each group: zero-shot first, then by model size.
ORDER = ["baseline", "legacy-format", "r8-attn", "r16-attn-mlp",
         "qwen-0.5b-zeroshot", "qwen-0.5b-v2recipe", "qwen-0.5b-r8",
         "qwen-1.5b-zeroshot", "qwen-1.5b-r8", "qwen-3b-zeroshot", "qwen-3b-r8"]

GROUP_NAMES = {
    "v3": "Frozen test set: recipe + model size",
    "v2": "Earlier experiments: prompt format + adapter size",
}


#----------STATISTICS----------#

#-----wilson_interval-----#
def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    # 95% interval for a proportion; stays sensible near 0% and 100%.
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


#-----mcnemar_exact-----#
def mcnemar(a: list[bool], b: list[bool]) -> float:
    # paired test: only lines where the two runs disagree count.
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


#----------LOAD_RUNS----------#

#-----read_mlflow-----#
def mlflow_info() -> dict:
    # newest run of each name: params + final metrics, read straight from SQLite (no MLflow import needed).
    if not DB_PATH.exists():
        return {}
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("SELECT name, run_uuid FROM runs ORDER BY start_time").fetchall()
    info = {}
    for name, uid in rows:  # later runs overwrite earlier ones with the same name
        params = dict(con.execute("SELECT key, value FROM params WHERE run_uuid=?", (uid,)).fetchall())
        metrics = dict(con.execute("SELECT key, value FROM latest_metrics WHERE run_uuid=?", (uid,)).fetchall())
        info[name] = {"params": params, "metrics": metrics}
    return info


#-----read_predictions-----#
def load_split(run: str, split: str) -> list[dict] | None:
    path = PRED_DIR / run / f"{split}.jsonl"
    if not path.exists():
        return None
    return [json.loads(line) for line in open(path) if line.strip()]


#-----group_by_test_set-----#
def test_set_key(rows: list[dict]) -> str:
    # runs are only comparable if they were scored on exactly the same lines.
    return hashlib.md5("\n".join(r["line"] for r in rows).encode()).hexdigest()[:8]


#----------BUILD_DATA----------#

def build() -> dict:
    meta = mlflow_info()
    runs = sorted((p.name for p in PRED_DIR.iterdir() if p.is_dir()),
                  key=lambda r: ORDER.index(r) if r in ORDER else len(ORDER))

    #-----collect_runs_with_both_splits-----#
    groups = defaultdict(lambda: {"runs": [], "lines": {}})
    run_data = {}
    for run in runs:
        seen, unseen = load_split(run, "seen"), load_split(run, "unseen")
        if seen is None or unseen is None:
            continue  # still training, or an old run without the unseen split.
        key = test_set_key(unseen)
        group = groups[key]
        group["runs"].append(run)
        group["lines"] = {"seen": [[r["line"], r["label"]] for r in seen],
                          "unseen": [[r["line"], r["label"]] for r in unseen]}

        #-----score_every_line-----#
        splits = {}
        for split, rows in [("seen", seen), ("unseen", unseen)]:
            flags = [score_one(r["reply"], r["label"]) for r in rows]
            preds = []
            for r in rows:
                p = parse_json(r["reply"])
                preds.append(p if p is not None else {"_raw": r["reply"][:120]})  # keep invalid replies visible
            n = len(rows)
            summary = {}
            for field in ["valid", "category", "merchant", "amount", "exact"]:
                k = sum(f[field] for f in flags)
                summary[field] = {"acc": k / n, "ci": wilson(k, n)}
            splits[split] = {
                "preds": preds,
                "flags": [[int(f["valid"]), int(f["category"]), int(f["merchant"]), int(f["amount"]), int(f["exact"])]
                          for f in flags],
                "summary": summary,
            }

        m = meta.get(run, {"params": {}, "metrics": {}})
        run_data[run] = {
            "label": LABELS.get(run, run),
            "group": key,
            "base_model": m["params"].get("base_model_name", "qwen-0.5b-4bit"),
            "zero_shot": m["params"].get("zero_shot") == "True",
            "trainable_params": m["metrics"].get("trainable_params"),
            "train_time_s": m["metrics"].get("train_time_s"),
            "best_iter": m["metrics"].get("best_iter"),
            "splits": splits,
        }

    #-----name_groups-----#
    named = {}
    for key, group in groups.items():
        name = "v3" if any(r.startswith("qwen-") for r in group["runs"]) else "v2"
        named[name] = {"title": GROUP_NAMES.get(name, name), **group}

    #-----significance_vs_best_run-----#
    for group in named.values():
        exact = {r: [f[4] == 1 for f in run_data[r]["splits"]["unseen"]["flags"]] for r in group["runs"]}
        best = max(group["runs"], key=lambda r: sum(exact[r]))
        group["best"] = best
        for r in group["runs"]:
            run_data[r]["p_vs_best"] = mcnemar(exact[r], exact[best])

    #-----headline_numbers-----#
    headline = {}
    if "v3" in named:
        v3 = named["v3"]
        headline["best_run"] = run_data[v3["best"]]["label"]
        headline["best_unseen"] = run_data[v3["best"]]["splits"]["unseen"]["summary"]["exact"]["acc"]
        if "qwen-0.5b-r8" in run_data and "qwen-0.5b-v2recipe" in run_data:
            new = run_data["qwen-0.5b-r8"]["splits"]["unseen"]["summary"]["exact"]["acc"]
            old = run_data["qwen-0.5b-v2recipe"]["splits"]["unseen"]["summary"]["exact"]["acc"]
            headline["recipe_gain"] = new - old
            a = [f[4] == 1 for f in run_data["qwen-0.5b-r8"]["splits"]["unseen"]["flags"]]
            b = [f[4] == 1 for f in run_data["qwen-0.5b-v2recipe"]["splits"]["unseen"]["flags"]]
            headline["recipe_p"] = mcnemar(a, b)
        r8 = run_data.get("qwen-0.5b-r8")
        if r8 and r8["trainable_params"]:
            headline["trainable_pct"] = r8["trainable_params"] / 494_033_000
            headline["train_min"] = (r8["train_time_s"] or 0) / 60

    return {"categories": CATEGORIES, "groups": named, "runs": run_data, "headline": headline}


#----------WRITE_HTML----------#

def main() -> None:
    data = build()
    if not data["runs"]:
        raise SystemExit("No predictions found: run train_mlx.py first.")
    template = Path("docs/site_template.html").read_text()
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")  # safe inside <script>
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(template.replace("/*__DATA__*/null", payload))
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1e3:.0f} kB) with {len(data['runs'])} runs "
          f"in {len(data['groups'])} groups")


if __name__ == "__main__":
    main()
