#!/usr/bin/env bash

# Job
#SBATCH -J NEUROCAD_MULTIMODAL
#SBATCH -t 7-00:00:00
#SBATCH -o train_logs/%x_%A_%N.out
#SBATCH --mail-type END,TIME_LIMIT_90,REQUEUE,INVALID_DEPEND # BEGIN,
#SBATCH --mail-user qshedll@gmail.com
#SBATCH -p RTX6000Ada
#SBATCH --gres=gpu:2

set -euo pipefail

IMAGE_TAG="${IMAGE_TAG:-localhost/lhyunki/llmseg_cfg:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-llmseg_dcm_a100_train_${SLURM_JOB_ID:-$$}}"
PROJECT_DIR="${PROJECT_DIR:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_seg_jhk}"
GPU_DEVICES="${GPU_DEVICES:-all}"
RUNTIME_BIN="${RUNTIME_BIN:-podman}"

mkdir -p train_logs

if ! command -v "$RUNTIME_BIN" >/dev/null 2>&1; then
    if command -v docker >/dev/null 2>&1; then
        RUNTIME_BIN="docker"
    else
        echo "[ERROR] podman/docker not found." >&2
        exit 1
    fi
fi

echo "[INFO] train_dcm_A100.sh safe heredoc version: 2026-07-05"
echo "[INFO] submit cwd=$(pwd)"
echo "[INFO] runtime=$RUNTIME_BIN image=$IMAGE_TAG container=$CONTAINER_NAME"
echo "[INFO] SLURM_JOB_ID=${SLURM_JOB_ID:-N/A} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-N/A}"

if [ "${REBUILD_IMAGE:-0}" = "1" ]; then
    "$RUNTIME_BIN" build -t "$IMAGE_TAG" .
fi

GPU_ARGS=()
if [ "$RUNTIME_BIN" = "docker" ]; then
    GPU_ARGS=(--gpus "$GPU_DEVICES")
else
    GPU_ARGS=(--device "nvidia.com/gpu=$GPU_DEVICES")
fi

"$RUNTIME_BIN" run --rm \
    --name "$CONTAINER_NAME" \
    --ipc=host \
    --shm-size=64g \
    "${GPU_ARGS[@]}" \
    -v /mnt/nas206:/mnt/nas206 \
    -v /mnt/nas125:/mnt/nas125 \
    -w "$PROJECT_DIR" \
    "$IMAGE_TAG" \
    bash -s <<'CONTAINER_SCRIPT'
set -euo pipefail

echo "[INFO] inside container: $(hostname)"
python - <<'PY'
import torch
print(f"[CUDA CHECK] torch={torch.__version__}, torch_cuda={torch.version.cuda}, available={torch.cuda.is_available()}, count={torch.cuda.device_count()}")
PY

PY_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$PY_SITE/nvidia/nvjitlink/lib:$PY_SITE/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

accelerate launch --num_processes 2  --mixed_precision bf16 --dynamo_backend no train.py --mixed_precision bf16  --patch_size 32 224 224  --batch_size 2 --llm_repo "/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_seg_jhk/model_custom/qwen3/Qwen3-8B-Base/"  --checkpoint_dir "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/qwen/bs2_cfg/" --experiment_name "context_qwen" --train_csv "/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_train_set.xlsx" --valid_csv "/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_valid_set.xlsx"  --epochs 300 --positive_prob 0.80   --loss_fct tversky --context

CONTAINER_SCRIPT
