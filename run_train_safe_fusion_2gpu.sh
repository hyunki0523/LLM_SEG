#!/bin/bash
set -euo pipefail

# Sequential 2-GPU launcher for the safe Llama + DICOM FiLM experiments.
#
# Examples:
#   bash run_train_safe_fusion_2gpu.sh
#   GPU_PAIR=2,3 EXPERIMENTS_2GPU=dicom_film bash run_train_safe_fusion_2gpu.sh
#   GPU_PAIR=6,7 SMOKE_TEST=1 bash run_train_safe_fusion_2gpu.sh

PROJECT_DIR="${PROJECT_DIR:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk}"
BASE_LAUNCHER="${BASE_LAUNCHER:-${PROJECT_DIR}/run_train_dicom_ablation_8gpu.sh}"
GPU_PAIR="${GPU_PAIR:-0,1}"
SMOKE_TEST="${SMOKE_TEST:-0}"
if [ -n "${EXPERIMENTS_2GPU:-}" ]; then
    EXPERIMENTS_2GPU="$EXPERIMENTS_2GPU"
elif [ "$SMOKE_TEST" = "1" ]; then
    EXPERIMENTS_2GPU="dicom_text_safe"
else
    EXPERIMENTS_2GPU="vision_only,dicom_film,text_safe,dicom_text_safe"
fi

IFS=',' read -r gpu0 gpu1 extra_gpu <<< "$GPU_PAIR"
if [ -z "${gpu0:-}" ] || [ -z "${gpu1:-}" ] || [ -n "${extra_gpu:-}" ]; then
    echo "[ERROR] GPU_PAIR must contain exactly two GPU IDs, for example GPU_PAIR=2,3"
    exit 2
fi
if ! [[ "$gpu0" =~ ^[0-9]+$ && "$gpu1" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] GPU_PAIR values must be non-negative integers: $GPU_PAIR"
    exit 2
fi
if [ "$gpu0" = "$gpu1" ]; then
    echo "[ERROR] GPU_PAIR must contain two different GPU IDs: $GPU_PAIR"
    exit 2
fi
if [ ! -f "$BASE_LAUNCHER" ]; then
    echo "[ERROR] Base launcher not found: $BASE_LAUNCHER"
    exit 2
fi

if [[ ",$EXPERIMENTS_2GPU," == *",dicom_film_frozen,"* ]] \
   || [[ ",$EXPERIMENTS_2GPU," == *",dicom_text_safe_frozen,"* ]]; then
    export RUN_EXTRA_MODES=1
fi

export PROJECT_DIR
export GPU_PAIRS="$GPU_PAIR"
export MAX_PARALLEL=1
export ONLY_EXPERIMENTS="$EXPERIMENTS_2GPU"

if [ "$SMOKE_TEST" = "1" ]; then
    export EPOCHS="${EPOCHS:-1}"
    export N_ITER_PER_EPOCH="${N_ITER_PER_EPOCH:-2}"
    export N_ITER_VALID="${N_ITER_VALID:-1}"
    export BATCH_SIZE="${BATCH_SIZE:-1}"
    export GRAD_ACCUM="${GRAD_ACCUM:-1}"
    export NUM_WORKERS="${NUM_WORKERS:-2}"
    export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1}"
    export OVERWRITE_TRAIN="${OVERWRITE_TRAIN:-1}"
    export CHECKPOINT_BASE="${CHECKPOINT_BASE:-${PROJECT_DIR}/_debug_ckpt/llama_safe_fusion_2gpu}"
    export LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/train_logs/llama_safe_fusion_2gpu_smoke_$(date +%Y%m%d_%H%M%S)}"
    export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
    export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
    export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
    export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
    # Smoke runs prioritize a trustworthy traceback over throughput. Without
    # this, an asynchronous kernel fault is commonly reported later at an
    # unrelated tensor .to() call.
    export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
fi

echo "=========================================================="
echo "[2-GPU SEQUENTIAL LAUNCHER]"
echo "GPU pair          : $GPU_PAIR"
echo "Experiments       : $EXPERIMENTS_2GPU"
echo "Smoke test        : $SMOKE_TEST"
echo "Base launcher     : $BASE_LAUNCHER"
echo "PRETRAINED        : ${PRETRAINED:-<none>}"
echo "CUDA blocking     : ${CUDA_LAUNCH_BLOCKING:-0}"
echo "=========================================================="

exec bash "$BASE_LAUNCHER"
