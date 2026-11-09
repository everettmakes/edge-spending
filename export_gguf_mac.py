"""Fuse MLX LoRA adapters into the model, convert to GGUF and quantise, using a local llama.cpp build.

Usage: python export_gguf_mac.py --adapter adapters/r8-attn --llama-cpp ../llama.cpp
"""
import argparse
import subprocess
import sys
from pathlib import Path

BASE_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
QUANTS = ["Q8_0", "Q4_K_M", "Q4_0"]


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--llama-cpp", required=True, help="path to your cloned + built llama.cpp")
    parser.add_argument("--out", default="gguf")
    parser.add_argument("--quants", nargs="+", default=QUANTS)
    args = parser.parse_args()

    name = Path(args.adapter).name
    fused = Path("fused") / name
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    f16 = out / f"{name}-F16.gguf"
    llama = Path(args.llama_cpp)

    # 1) merge LoRA into the weights and dequantise to 16-bit Hugging Face format
    run([sys.executable, "-m", "mlx_lm", "fuse", "--model", BASE_MODEL,
         "--adapter-path", args.adapter, "--save-path", str(fused), "--dequantize"])
    # 2) Hugging Face safetensors -> GGUF (16-bit)
    run([sys.executable, str(llama / "convert_hf_to_gguf.py"), str(fused),
         "--outfile", str(f16), "--outtype", "f16"])
    # 3) quantise for the edge
    for quant in args.quants:
        run([str(llama / "build" / "bin" / "llama-quantize"), str(f16), str(out / f"{name}-{quant}.gguf"), quant])

    for path in sorted(out.glob(f"{name}-*.gguf")):
        print(f"{path.name}: {path.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
