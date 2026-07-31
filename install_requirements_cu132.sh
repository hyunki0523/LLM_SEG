#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk}"
PYTHON_EXE="${PYTHON_EXE:-python}"
INSTALL_LORA="${INSTALL_LORA:-0}"
EXPECTED_GPUS="${EXPECTED_GPUS:-4}"

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

# Audio and torchvision are unused here; stale builds pinning an older torch
# would make the CUDA 13.2 environment inconsistent.
"$PYTHON_EXE" -m pip uninstall -y torchaudio torchvision || true

# Install the official CUDA 13.2 wheel separately so the generic PyPI resolver
# cannot silently select a CPU or another CUDA build. torchvision is omitted
# because this repository does not import it.
"$PYTHON_EXE" -m pip install \
    torch==2.12.1 \
    --index-url https://download.pytorch.org/whl/cu132

"$PYTHON_EXE" -m pip install -r requirements_current.txt

if [ "$INSTALL_LORA" = "1" ]; then
    "$PYTHON_EXE" -m pip install -r requirements_lora.txt
fi

"$PYTHON_EXE" -m pip check

EXPECTED_GPUS="$EXPECTED_GPUS" \
CHECK_PEFT="$INSTALL_LORA" \
"$PYTHON_EXE" verify_runtime_environment.py

echo "[DONE] CUDA 13.2 LLM_SEG_hk environment is ready."
