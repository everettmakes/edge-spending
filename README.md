# Edge spending categoriser

Fine-tune a 0.5B language model to turn messy bank-statement lines into clean JSON, then prove it is
small and fast enough to run **on-device**, so financial data never leaves the phone or laptop.

```
"TESCO STORES 3297 BRISTOL £23.40"  →  {"merchant": "Tesco", "amount": 23.40, "category": "groceries"}
```

The whole pipeline runs on one Apple Silicon (Arm) laptop: QLoRA fine-tuning with MLX, GGUF export with
llama.cpp, and CPU inference benchmarks with llama-cpp-python. Every run is tracked in MLflow.

## Pipeline

| stage | script | runs on |
|---|---|---|
| generate synthetic transactions | `make_data.py` | CPU |
| convert to MLX chat / legacy formats | `prepare_mlx_data.py` | CPU |
| zero-shot baseline + QLoRA fine-tuning + evaluation | `train_mlx.py` | Apple GPU (MLX) |
| fuse adapters → GGUF → Q8_0 / Q4_K_M / Q4_0 | `export_gguf_mac.py` | CPU |
| on-device benchmark (size, RAM, latency, accuracy) | `benchmark.py` | Arm CPU |
| results tables + accuracy-vs-size chart | `results.py` | CPU |

`cloud/` holds the equivalent Unsloth scripts for an NVIDIA GPU (Colab/Kaggle).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# llama.cpp, for GGUF conversion + quantisation (KleidiAI on for the Arm bonus)
git clone https://github.com/ggml-org/llama.cpp ../llama.cpp
cmake -S ../llama.cpp -B ../llama.cpp/build -DGGML_CPU_KLEIDIAI=ON
cmake --build ../llama.cpp/build --config Release -j
pip install -r ../llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
```

## Run order

```bash
python make_data.py                                   # 600 train / 150 test
python prepare_mlx_data.py                            # 540 train / 60 valid / 150 test

python train_mlx.py --run-name baseline --zero-shot   # base model, no training
python train_mlx.py --run-name legacy-format --template legacy   # tutorial's mismatched format
python train_mlx.py --run-name r8-attn                # chat template, rank 8, attention only
python train_mlx.py --run-name r16-attn-mlp --rank 16 --targets attn_mlp

python export_gguf_mac.py --adapter adapters/r8-attn --llama-cpp ../llama.cpp   # use your best run
python benchmark.py --gguf-dir gguf                   # CPU only (n_gpu_layers=0)
python results.py                                     # results.md + accuracy_vs_size.png

mlflow ui --backend-store-uri sqlite:///mlflow.db     # browse every run
```

## Results

_Fill in from `results.md` once the runs finish._

## What didn't work

_The tutorial's training format (`### Instruction / ### Response`) did not match the chat template used at
inference. Compare the `legacy-format` and `r8-attn` runs._
