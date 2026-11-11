#!/usr/bin/env bash
# Optional 3B reference point (~50+ min). Run after run_1_5b.sh, e.g. overnight.
set -euo pipefail
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
python train_mlx.py --run-name qwen-3b-zeroshot --base-model qwen-3b-4bit --zero-shot --dtype bfloat16
python train_mlx.py --run-name qwen-3b-r8 --base-model qwen-3b-4bit --dtype bfloat16
python results.py
python make_site.py
