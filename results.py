"""Pull every MLflow run into results tables and an accuracy-vs-size chart.

Run on your laptop after benchmarking:  python results.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402

from common import EXPERIMENT  # noqa: E402

TRAIN_COLS = {
    "tags.mlflow.runName": "run", "params.template": "template", "params.rank": "rank",
    "params.targets": "targets", "metrics.trainable_params": "trainable_params",
    "metrics.peak_mem_mb": "peak_mem_mb", "metrics.train_time_s": "train_time_s",
    "metrics.valid_json_rate": "valid_json", "metrics.category_accuracy": "category_acc",
    "metrics.exact_match": "exact_match",
}
BENCH_COLS = {
    "tags.quant": "quant", "tags.device": "device", "tags.kleidiai": "kleidiai", "metrics.file_size_mb": "size_mb",
    "metrics.peak_ram_mb": "peak_ram_mb", "metrics.ms_per_txn_mean": "ms_per_txn",
    "metrics.tokens_per_s": "tokens_per_s", "metrics.valid_json_rate": "valid_json",
    "metrics.category_accuracy": "category_acc", "metrics.exact_match": "exact_match",
}


def table(runs, stages: list[str], columns: dict):
    subset = runs[runs["tags.stage"].isin(stages)]
    present = [c for c in columns if c in subset.columns]
    return subset[present].rename(columns=columns).round(3).reset_index(drop=True)


def main() -> None:
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT])
    if runs.empty:
        raise SystemExit("No runs found: check MLFLOW_TRACKING_URI points at your mlruns folder.")

    training = table(runs, ["baseline", "train"], TRAIN_COLS).sort_values("exact_match", ascending=False)
    bench = table(runs, ["benchmark"], BENCH_COLS).sort_values("size_mb")

    with open("results.md", "w") as f:
        f.write("## Training runs\n\n" + training.to_markdown(index=False) + "\n\n")
        f.write("## Edge benchmark (laptop CPU)\n\n" + bench.to_markdown(index=False) + "\n")
    print(open("results.md").read())

    if not bench.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(bench["size_mb"], bench["exact_match"])
        for _, row in bench.iterrows():
            ax.annotate(f"{row['quant']}\n{row['ms_per_txn']:.0f} ms", (row["size_mb"], row["exact_match"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=9)
        ax.set_xlabel("Model file size (MB)")
        ax.set_ylabel("Exact-match accuracy")
        ax.set_title("On-device trade-off: accuracy vs size")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig("accuracy_vs_size.png", dpi=150)
        print("Saved accuracy_vs_size.png")

    with mlflow.start_run(run_name="summary"):
        mlflow.set_tag("stage", "summary")
        mlflow.log_artifact("results.md")
        if not bench.empty:
            mlflow.log_artifact("accuracy_vs_size.png")


if __name__ == "__main__":
    main()
