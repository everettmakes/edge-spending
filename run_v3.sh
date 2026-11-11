#!/usr/bin/env bash
# v3: fixed test sets, recipe ablation, 0.5B + 1.5B ladder. Runs unattended; ~45 min on a MacBook Air.
# The 3B runs are in run_3b.sh (~50 min): run them overnight if you want the bigger reference point.
# Usage: source .venv/bin/activate && bash run_v3.sh 2>&1 | tee run_v3.log
set -euo pipefail
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# data: v3 recipe + the old recipe on identical test sets
python make_data.py
python make_data.py --out data_v2recipe --no-augment --no-extra
python prepare_mlx_data.py
python prepare_mlx_data.py --data-dir data_v2recipe --out mlx_v2recipe

# 1) does the new recipe help? same model, same tests
python train_mlx.py --run-name qwen-0.5b-v2recipe --base-model qwen-0.5b-4bit --mlx-data mlx_v2recipe
python train_mlx.py --run-name qwen-0.5b-r8 --base-model qwen-0.5b-4bit

# 2) size ladder: zero-shot and fine-tuned
python train_mlx.py --run-name qwen-0.5b-zeroshot --base-model qwen-0.5b-4bit --zero-shot
python train_mlx.py --run-name qwen-1.5b-zeroshot --base-model qwen-1.5b-4bit --zero-shot
python train_mlx.py --run-name qwen-1.5b-r8 --base-model qwen-1.5b-4bit

python results.py
