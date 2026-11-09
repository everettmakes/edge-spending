"""Benchmark GGUF models on this machine's CPU with llama-cpp-python, logging one MLflow run per file.

Run on your laptop:  python benchmark.py --gguf-dir gguf
Optional tag for the Arm KleidiAI comparison:  --kleidiai on|off
"""
import argparse
import platform
import re
import time
from pathlib import Path

import mlflow
import psutil
from llama_cpp import Llama

from common import EXPERIMENT, build_messages, load_jsonl, score


def quant_label(path: Path) -> str:
    """Pull 'Q4_K_M' etc. out of a filename; fall back to the parent folder name."""
    match = re.search(r"(Q\d_K_[SML]|Q\d_\d|Q\d_K|F16|BF16)", path.name, re.IGNORECASE)
    return match.group(1).upper() if match else path.parent.name.upper()


def benchmark(path: Path, rows: list[dict], threads: int, kleidiai: str, n_gpu_layers: int) -> None:
    process = psutil.Process()
    with mlflow.start_run(run_name=f"bench-{quant_label(path)}-{'cpu' if n_gpu_layers == 0 else 'gpu'}"):
        mlflow.set_tags({
            "stage": "benchmark", "quant": quant_label(path), "kleidiai": kleidiai,
            "device": "cpu" if n_gpu_layers == 0 else "gpu",
            "cpu": platform.processor() or platform.machine(), "machine": platform.machine(),
        })
        mlflow.log_params({"model_file": path.name, "threads": threads, "n_test": len(rows)})

        start = time.perf_counter()
        llm = Llama(model_path=str(path), n_ctx=512, n_threads=threads, n_gpu_layers=n_gpu_layers,
                    use_mmap=False, verbose=False)
        load_s = time.perf_counter() - start
        peak_rss = process.memory_info().rss

        replies, latencies, completion_tokens = [], [], 0
        for row in rows:
            start = time.perf_counter()
            out = llm.create_chat_completion(messages=build_messages(row["line"]), temperature=0, max_tokens=64)
            latencies.append(time.perf_counter() - start)
            replies.append(out["choices"][0]["message"]["content"])
            completion_tokens += out["usage"]["completion_tokens"]
            peak_rss = max(peak_rss, process.memory_info().rss)

        metrics = score(replies, [r["label"] for r in rows])
        latencies.sort()
        metrics.update({
            "file_size_mb": path.stat().st_size / 1e6,
            "load_time_s": load_s,
            "peak_ram_mb": peak_rss / 1e6,
            "ms_per_txn_mean": 1000 * sum(latencies) / len(latencies),
            "ms_per_txn_p95": 1000 * latencies[int(0.95 * (len(latencies) - 1))],
            "tokens_per_s": completion_tokens / sum(latencies),
        })
        mlflow.log_metrics(metrics)
        print(f"{path.name}: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf-dir", default="gguf")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--limit", type=int, default=None, help="use fewer test rows for a quick check")
    parser.add_argument("--threads", type=int, default=psutil.cpu_count(logical=False) or 4)
    parser.add_argument("--kleidiai", choices=["on", "off", "unknown"], default="unknown")
    parser.add_argument("--n-gpu-layers", type=int, default=0, help="0 = pure CPU (the edge case); -1 = all on GPU/Metal")
    args = parser.parse_args()

    rows = load_jsonl(f"{args.data_dir}/test.jsonl")[: args.limit]
    files = sorted(Path(args.gguf_dir).rglob("*.gguf"))
    if not files:
        raise SystemExit(f"No .gguf files found under {args.gguf_dir}/")

    mlflow.set_experiment(EXPERIMENT)
    for path in files:
        benchmark(path, rows, args.threads, args.kleidiai, args.n_gpu_layers)


if __name__ == "__main__":
    main()
