"""Pull every MLflow run into results tables, significance tests and an accuracy-vs-size chart.

Run on your laptop after training and benchmarking:  python results.py

Statistics (all on exact-match accuracy, per test line):
  * 95% Wilson confidence interval for each accuracy: how much it could move on a different test set
  * exact McNemar test between two models on the same lines: is the difference more than noise?
      p_vs_best  each run against the best run in its table
      p_vs_f16   each quantised file against the 16-bit file from the same training run
  * mean and spread across seeds, for configs trained more than once
"""
#----------IMPORTS----------#
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

from common import EXPERIMENT, SPLITS, score_one  # noqa: E402

#----------CONFIG----------#

ALPHA = 0.05


#----------STATISTICS----------#

#-----wilson-----#
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for k successes out of n (behaves well near 0% and 100%)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


#-----mcnemar-----#
def mcnemar(a: list[bool], b: list[bool]) -> float:
    """Exact two-sided McNemar p-value for paired correctness lists (same lines, same order).

    Only the lines where the models disagree count: if A alone is right on `only_a` lines and B alone on
    `only_b`, then under 'no difference' each disagreement is a coin flip, so only_a ~ Binomial(n, 0.5).
    """
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


#----------LOADING----------#

#-----load_correct-----#
def load_correct(run_name: str, split: str) -> list[bool] | None:
    """Per-line exact-match correctness from predictions/<run>/<split>.jsonl, if it exists."""
    path = Path("predictions") / run_name / f"{split}.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return [score_one(r["reply"], r["label"])["exact"] for r in rows]


#-----col-----#
def col(runs: pd.DataFrame, name: str) -> pd.Series:
    return runs[name] if name in runs.columns else pd.Series([None] * len(runs), index=runs.index)


#-----latest_runs-----#
def latest_runs() -> pd.DataFrame:
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT])
    if runs.empty:
        raise SystemExit("No runs found: check MLFLOW_TRACKING_URI=sqlite:///mlflow.db is set.")
    runs = runs.sort_values("start_time", ascending=False)
    return runs.drop_duplicates("tags.mlflow.runName").reset_index(drop=True)  # keep the newest of each name


#-----add_stats-----#
def add_stats(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Add CI and p_vs_best columns for one split, using per-line predictions on disk."""
    correct = {run: load_correct(run, split) for run in df["run"]}
    acc, lo, hi = [], [], []
    for run in df["run"]:
        c = correct[run]
        if c is None:
            acc.append(float("nan")); lo.append(float("nan")); hi.append(float("nan"))  # noqa: E702
            continue
        a, b = wilson(sum(c), len(c))
        acc.append(sum(c) / len(c)); lo.append(a); hi.append(b)  # noqa: E702
    df = df.assign(**{f"{split}_exact": acc, f"{split}_lo": lo, f"{split}_hi": hi})

    scored = df.dropna(subset=[f"{split}_exact"])
    if not scored.empty:
        best = scored.sort_values(f"{split}_exact", ascending=False)["run"].iloc[0]
        df[f"{split}_p_vs_best"] = [
            mcnemar(correct[run], correct[best]) if correct[run] is not None else float("nan")
            for run in df["run"]
        ]
    return df


#-----fmt_ci-----#
def fmt_ci(df: pd.DataFrame, split: str) -> pd.Series:
    return df.apply(lambda r: "" if pd.isna(r[f"{split}_exact"]) else
                    f"{r[f'{split}_exact']:.3f} [{r[f'{split}_lo']:.2f}, {r[f'{split}_hi']:.2f}]", axis=1)


#----------TABLES----------#

#-----training_table-----#
def training_table(runs: pd.DataFrame) -> pd.DataFrame:
    sub = runs[runs["tags.stage"].isin(["baseline", "train"])]
    df = pd.DataFrame({
        "run": sub["tags.mlflow.runName"],
        "base_model": col(sub, "params.base_model_name").fillna("qwen-0.5b-4bit"),
        "template": col(sub, "params.template").fillna("-"),
        "rank": col(sub, "params.rank").fillna("-"),
        "targets": col(sub, "params.targets").fillna("-"),
        "seed": col(sub, "params.seed").fillna("0"),
        "trainable_params": col(sub, "metrics.trainable_params"),
        "train_time_s": col(sub, "metrics.train_time_s"),
        "best_iter": col(sub, "metrics.best_iter"),
    }).reset_index(drop=True)
    for split in SPLITS:
        df = add_stats(df, split)
    return df


#-----benchmark_table-----#
def benchmark_table(runs: pd.DataFrame, training: pd.DataFrame) -> pd.DataFrame:
    sub = runs[runs["tags.stage"] == "benchmark"]
    if sub.empty:
        return pd.DataFrame()
    df = pd.DataFrame({
        "run": sub["tags.mlflow.runName"],
        "model_run": col(sub, "tags.model_run"),
        "quant": col(sub, "tags.quant"),
        "device": col(sub, "tags.device"),
        "kleidiai": col(sub, "tags.kleidiai"),
        "size_mb": col(sub, "metrics.file_size_mb"),
        "model_ram_mb": col(sub, "metrics.model_ram_mb"),
        "ms_per_txn": col(sub, "metrics.ms_per_txn_mean"),
        "tokens_per_s": col(sub, "metrics.tokens_per_s"),
    }).reset_index(drop=True)
    base = dict(zip(training["run"], training["base_model"]))
    df.insert(1, "base_model", df["model_run"].map(base))
    for split in SPLITS:
        df = add_stats(df, split)

    # does quantisation hurt? each file vs the F16 file from the same training run, on the unseen split
    split = "unseen" if df["unseen_exact"].notna().any() else "seen"
    p_f16 = []
    for _, row in df.iterrows():
        ref = df[(df["model_run"] == row["model_run"]) & (df["quant"] == "F16") & (df["device"] == row["device"])]
        mine, theirs = load_correct(row["run"], split), None
        if not ref.empty:
            theirs = load_correct(ref["run"].iloc[0], split)
        p_f16.append(mcnemar(mine, theirs) if mine is not None and theirs is not None and row["quant"] != "F16"
                     else float("nan"))
    df[f"{split}_p_vs_f16"] = p_f16
    return df.sort_values("size_mb").reset_index(drop=True)


#-----seed_table-----#
def seed_table(training: pd.DataFrame) -> pd.DataFrame:
    trained = training[training["template"] != "-"]
    keys = ["base_model", "template", "rank", "targets"]
    groups = trained.groupby(keys)
    rows = []
    for key, g in groups:
        if len(g) < 2:
            continue
        for split in SPLITS:
            vals = g[f"{split}_exact"].dropna()
            if len(vals) >= 2:
                rows.append({**dict(zip(keys, key)), "split": split, "n_seeds": len(vals),
                             "mean_exact": vals.mean(), "std_exact": vals.std(ddof=1),
                             "min": vals.min(), "max": vals.max()})
    return pd.DataFrame(rows)


#-----to_md-----#
def to_md(df: pd.DataFrame, columns: list[str]) -> str:
    present = [c for c in columns if c in df.columns]
    return df[present].round(3).to_markdown(index=False)


#----------CHART----------#

#-----chart-----#
def chart(bench: pd.DataFrame, split: str) -> None:
    data = bench.dropna(subset=[f"{split}_exact"])
    if data.empty:
        return
    markers = {"F16": "s", "Q8_0": "D", "Q4_K_M": "o", "Q4_0": "^"}
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for model, g in data.groupby("model_run"):
        colour = None
        for _, r in g.iterrows():
            err = [[r[f"{split}_exact"] - r[f"{split}_lo"]], [r[f"{split}_hi"] - r[f"{split}_exact"]]]
            line = ax.errorbar(r["size_mb"], r[f"{split}_exact"], yerr=err, fmt=markers.get(r["quant"], "o"),
                               color=colour, capsize=3, label=model if colour is None else None)
            colour = line[0].get_color()
            ax.annotate(r["quant"], (r["size_mb"], r[f"{split}_exact"]), textcoords="offset points",
                        xytext=(6, -10), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Model file size (MB, log scale)")
    ax.set_ylabel(f"Exact match, {split} test (95% CI)")
    ax.set_title("How small can it get? Accuracy vs size on the laptop CPU")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("accuracy_vs_size.png", dpi=150)
    print("Saved accuracy_vs_size.png")


#----------MAIN----------#

def main() -> None:
    runs = latest_runs()
    training = training_table(runs).sort_values(["unseen_exact", "seen_exact"], ascending=False, na_position="last")
    bench = benchmark_table(runs, training)
    seeds = seed_table(training)

    training["unseen (95% CI)"] = fmt_ci(training, "unseen")
    has_real = training["real_exact"].notna().any()
    if has_real:
        training["real (95% CI)"] = fmt_ci(training, "real")
    lines = ["## Training runs", "",
             "Exact match per test split. `unseen_p_vs_best`: McNemar p-value against the best run "
             f"(p < {ALPHA} means it is genuinely worse, not noise).", "",
             to_md(training, ["run", "base_model", "template", "rank", "targets", "seed", "trainable_params",
                              "best_iter", "seen_exact", "unseen (95% CI)", "unseen_p_vs_best"]
                             + (["real (95% CI)", "real_p_vs_best"] if has_real else [])), ""]

    if not seeds.empty:
        lines += ["## Seed variation", "", to_md(seeds, list(seeds.columns)), ""]

    if not bench.empty:
        split = "unseen" if bench["unseen_exact"].notna().any() else "seen"
        bench[f"{split} (95% CI)"] = fmt_ci(bench, split)
        lines += ["## Edge benchmark (laptop CPU)", "",
                  f"`{split}_p_vs_f16`: does this quantisation lose accuracy against the 16-bit file of the same model?",
                  "", to_md(bench, ["model_run", "base_model", "quant", "kleidiai", "size_mb", "model_ram_mb",
                                    "ms_per_txn", "tokens_per_s", "seen_exact", f"{split} (95% CI)",
                                    f"{split}_p_vs_best", f"{split}_p_vs_f16"]), ""]
        keep = bench[bench[f"{split}_p_vs_best"] >= ALPHA].sort_values("size_mb")
        if not keep.empty:
            r = keep.iloc[0]
            lines += [f"**Smallest file not significantly worse than the best:** `{r['model_run']}` {r['quant']} "
                      f"({r['size_mb']:.0f} MB, {split} exact match {r[f'{split}_exact']:.3f}, "
                      f"p = {r[f'{split}_p_vs_best']:.2f} vs best, {r['ms_per_txn']:.0f} ms per transaction)", ""]
        chart(bench, split)

    Path("results.md").write_text("\n".join(lines))
    print("\n".join(lines))

    with mlflow.start_run(run_name="summary"):
        mlflow.set_tag("stage", "summary")
        mlflow.log_artifact("results.md")
        if Path("accuracy_vs_size.png").exists():
            mlflow.log_artifact("accuracy_vs_size.png")


if __name__ == "__main__":
    main()
