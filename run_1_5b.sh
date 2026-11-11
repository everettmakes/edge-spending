#!/usr/bin/env bash
# 1.5B fine-tune (~30 min). bfloat16: in float16 the 1.5B model's loss goes NaN (overflow).
set -euo pipefail
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
python train_mlx.py --run-name qwen-1.5b-r8 --base-model qwen-1.5b-4bit --dtype bfloat16
python results.py
python make_site.py
