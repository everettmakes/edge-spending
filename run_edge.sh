#!/usr/bin/env bash
# Export every ladder model to GGUF (F16, Q8_0, Q4_K_M, Q4_0) and benchmark on the CPU.
# Needs ../llama.cpp built (see README). Usage: bash run_edge.sh 2>&1 | tee run_edge.log
set -euo pipefail
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db

for run in qwen-0.5b-r8 qwen-1.5b-r8 qwen-3b-r8; do
  [ -d adapters/$run ] || { echo "skipping $run (not trained)"; continue; }
  python export_gguf_mac.py --adapter adapters/$run --llama-cpp ../llama.cpp
done

# On an 8 GB machine, skip the 3B F16 file (about 6 GB): add  --match "Q"  to benchmark quantised files only
python benchmark.py --gguf-dir gguf --kleidiai off
python results.py
