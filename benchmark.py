"""Benchmark GGUF models on this machine's CPU with llama-cpp-python, logging one MLflow run per file.

Each file is scored on both test splits (seen + unseen); latency and RAM are measured across all lines.
Per-line predictions go to predictions/bench-<file>/ so results.py can run significance tests.

Run on your laptop:  python benchmark.py --gguf-dir gguf
Only some files:     python benchmark.py --gguf-dir gguf --match smollm2
Optional tag for the Arm KleidiAI comparison:  --kleidiai on|off
"""
#----------IMPORTS----------#
import argparse
import json
import platform
import re
import time
from pathlib import Path

import mlflow
import psutil
from llama_cpp import Llama

from common import EXPERIMENT, SPLITS, build_messages, load_jsonl, metric_prefix, score

#----------CONFIG----------#

QUANT_RE = re.compile(r"(Q\d_K_[SML]|Q\d_\d|Q\d_K|BF16|F16)", re.IGNORECASE)


#----------HELPERS----------#

#-----quant_label-----#
def quant_label(path: Path) -> str:
    """Pull 'Q4_K_M' etc. out of a filename; fall back to the parent folder name."""
    match = QUANT_RE.search(path.name)
    return match.group(1).upper() if match else path.parent.name.upper()


#-----model_run-----#
def model_run(path: Path) -> str:
    """'smollm2-135m-r8-Q4_0.gguf' -> 'smollm2-135m-r8' (the training run the file came from)."""
    return QUANT_RE.split(path.stem)[0].rstrip("-_")


#----------BENCHMARK----------#

#-----benchmark-----#
def benchmark(path: Path, splits: dict, threads: int, kleidiai: str, n_gpu_layers: int) -> None:
    process = psutil.Process()
    device = "cpu" if n_gpu_layers == 0 else "gpu"
    run_name = f"bench-{path.stem}-{device}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({
            "stage": "benchmark", "quant": quant_label(path), "model_run": model_run(path),
            "kleidiai": kleidiai, "device": device,
            "cpu": platform.processor() or platform.machine(), "machine": platform.machine(),
        })
        mlflow.log_params({"model_file": path.name, "threads": threads,
                           **{f"n_test_{name}": len(rows) for name, rows in splits.items()}})

        rss_before = process.memory_info().rss
        start = time.perf_counter()
        llm = Llama(model_path=str(path), n_ctx=512, n_threads=threads, n_gpu_layers=n_gpu_layers,
                    use_mmap=False, verbose=False)
        load_s = time.perf_counter() - start
        peak_rss = process.memory_info().rss

        metrics, latencies, completion_tokens = {}, [], 0
        out_dir = Path("predictions") / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, rows in splits.items():
            replies = []
            for row in rows:
                start = time.perf_counter()
                out = llm.create_chat_completion(messages=build_messages(row["line"]), temperature=0, max_tokens=64)
                latencies.append(time.perf_counter() - start)
                replies.append(out["choices"][0]["message"]["content"])
                completion_tokens += out["usage"]["completion_tokens"]
                peak_rss = max(peak_rss, process.memory_info().rss)
            metrics.update({metric_prefix(name) + k: v
                            for k, v in score(replies, [r["label"] for r in rows]).items()})

            pred_path = out_dir / f"{name}.jsonl"
            with open(pred_path, "w") as f:
                for row, reply in zip(rows, replies):
                    f.write(json.dumps({"line": row["line"], "label": row["label"], "reply": reply}) + "\n")
            mlflow.log_artifact(str(pred_path), artifact_path="predictions")

        latencies.sort()
        metrics.update({
            "file_size_mb": path.stat().st_size / 1e6,
            "load_time_s": load_s,
            "peak_ram_mb": peak_rss / 1e6,
            "model_ram_mb": (peak_rss - rss_before) / 1e6,  # RAM this model added; comparable across files
            "ms_per_txn_mean": 1000 * sum(latencies) / len(latencies),
            "ms_per_txn_p95": 1000 * latencies[int(0.95 * (len(latencies) - 1))],
            "tokens_per_s": completion_tokens / sum(latencies),
        })
        mlflow.log_metrics(metrics)
        print(f"{path.name}: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
        del llm  # free the model before loading the next file, so peak RAM isn't cumulative


#----------MAIN----------#
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf-dir", default="gguf")
    parser.add_argument("--match", default="", help="only benchmark files whose name contains this")
    parser.add_argument("--limit", type=int, default=None, help="use fewer test rows per split for a quick check")
    parser.add_argument("--threads", type=int, default=psutil.cpu_count(logical=False) or 4)
    parser.add_argument("--kleidiai", choices=["on", "off", "unknown"], default="unknown")
    parser.add_argument("--n-gpu-layers", type=int, default=0, help="0 = pure CPU (the edge case); -1 = all on GPU/Metal")
    args = parser.parse_args()

    splits = {name: load_jsonl(path)[: args.limit] for name, path in SPLITS.items() if Path(path).exists()}
    files = sorted(p for p in Path(args.gguf_dir).rglob("*.gguf") if args.match in p.name)
    if not files:
        raise SystemExit(f"No matching .gguf files found under {args.gguf_dir}/")

    mlflow.set_experiment(EXPERIMENT)
    for path in files:
        benchmark(path, splits, args.threads, args.kleidiai, args.n_gpu_layers)


if __name__ == "__main__":
    main()
