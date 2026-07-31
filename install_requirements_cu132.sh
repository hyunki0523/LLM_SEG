#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk}"
PYTHON_EXE="${PYTHON_EXE:-python}"
INSTALL_LORA="${INSTALL_LORA:-0}"
EXPECTED_GPUS="${EXPECTED_GPUS:-4}"
FORCE_TORCH_REINSTALL="${FORCE_TORCH_REINSTALL:-0}"
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-cu132}"

cd "$PROJECT_DIR"

"$PYTHON_EXE" - <<'PY'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit(
        f"Python 3.11 is required for the locked environment; got {sys.version}"
    )
print(f"[CHECK] Python={sys.version.split()[0]}")
PY

"$PYTHON_EXE" -m pip install --upgrade pip setuptools wheel

# These packages belong to the base pathology/image environment, are not
# imported by LLM_SEG_hk, and impose mutually incompatible torchvision,
# Transformers, PyYAML and scikit-image constraints.
"$PYTHON_EXE" -m pip uninstall -y \
    torchaudio \
    torchvision \
    tiatoolbox \
    timm \
    ninja \
    || true

# Keep an existing torch 2.12.1 CUDA 13.x build. cu130 and cu132 are both
# supported targets for this project. Set FORCE_TORCH_REINSTALL=1 to replace
# it explicitly with the wheel selected by TORCH_CUDA_INDEX.
torch_is_compatible="$(
    "$PYTHON_EXE" - <<'PY'
try:
    import torch
    version_ok = torch.__version__.split("+", 1)[0] == "2.12.1"
    cuda = tuple(int(part) for part in (torch.version.cuda or "0.0").split(".")[:2])
    print("1" if version_ok and cuda >= (13, 0) else "0")
except Exception:
    print("0")
PY
)"

if [ "$FORCE_TORCH_REINSTALL" = "1" ] || [ "$torch_is_compatible" != "1" ]; then
    "$PYTHON_EXE" -m pip install \
        --no-cache-dir \
        --force-reinstall \
        torch==2.12.1 \
        --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_INDEX}"
else
    "$PYTHON_EXE" - <<'PY'
import torch
print(
    f"[KEEP] torch={torch.__version__}, CUDA runtime={torch.version.cuda}; "
    "already compatible with CUDA 13.x."
)
PY
fi

"$PYTHON_EXE" -m pip install -r requirements_current.txt

if [ "$INSTALL_LORA" = "1" ]; then
    "$PYTHON_EXE" -m pip install -r requirements_lora.txt
fi

"$PYTHON_EXE" -m pip check

EXPECTED_GPUS="$EXPECTED_GPUS" \
CHECK_PEFT="$INSTALL_LORA" \
"$PYTHON_EXE" verify_runtime_environment.py

echo "[DONE] CUDA 13.x LLM_SEG_hk environment is ready."
