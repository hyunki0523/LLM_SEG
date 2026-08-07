#!/bin/bash
set -euo pipefail

# True DICOM-FiLM-only counterpart to run_train_wave0_vision_ema_2gpu.sh.
# Keep every optimization/data setting identical to the Vision baseline; only
# use_dicom changes from false to true. No LLM, CC, or text cache is loaded.

PROJECT_DIR="${PROJECT_DIR:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk}"
BASE_LAUNCHER="${BASE_LAUNCHER:-${PROJECT_DIR}/run_train_dicom_ablation_8gpu.sh}"
PYTHON_EXE="${PYTHON_EXE:-python}"
GPU_PAIR="${GPU_PAIR:-0,1}"
TRAIN_CSV="${TRAIN_CSV:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_train_set.xlsx}"
VALID_CSV="${VALID_CSV:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_valid_set.xlsx}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/llama/hybrid_wave0_dicom_film_ema_ds2}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/train_logs/hybrid_wave0_dicom_film_ema_ds2_2gpu_$(date +%Y%m%d_%H%M%S)}"
SMOKE_TEST="${SMOKE_TEST:-0}"

IFS=',' read -r -a gpu_list <<< "$GPU_PAIR"
if [ "${#gpu_list[@]}" -ne 2 ] \
   || [ "${gpu_list[0]}" = "${gpu_list[1]}" ] \
   || ! [[ "${gpu_list[0]}" =~ ^[0-9]+$ ]] \
   || ! [[ "${gpu_list[1]}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] GPU_PAIR must contain two distinct IDs, e.g. GPU_PAIR=0,1"
    exit 2
fi
if [ ! -f "$BASE_LAUNCHER" ]; then
    echo "[ERROR] Base launcher not found: $BASE_LAUNCHER"
    exit 2
fi

CUDA_VISIBLE_DEVICES="$GPU_PAIR" EXPECTED_GPUS=2 \
    "$PYTHON_EXE" "${PROJECT_DIR}/verify_runtime_environment.py"

# These defaults are deliberately identical to the Vision Wave0 launcher.
EPOCHS="${EPOCHS:-120}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
N_ITER_PER_EPOCH="${N_ITER_PER_EPOCH:-128}"
N_ITER_VALID="${N_ITER_VALID:-25}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10}"
OVERWRITE_TRAIN="${OVERWRITE_TRAIN:-0}"

if [ "$SMOKE_TEST" = "1" ]; then
    EPOCHS="${SMOKE_EPOCHS:-1}"
    BATCH_SIZE="${SMOKE_BATCH_SIZE:-1}"
    GRAD_ACCUM="${SMOKE_GRAD_ACCUM:-1}"
    N_ITER_PER_EPOCH="${SMOKE_N_ITER_PER_EPOCH:-2}"
    N_ITER_VALID="${SMOKE_N_ITER_VALID:-1}"
    NUM_WORKERS="${SMOKE_NUM_WORKERS:-2}"
    CHECKPOINT_INTERVAL=1
    OVERWRITE_TRAIN=1
    CHECKPOINT_BASE="${CHECKPOINT_BASE_SMOKE:-${PROJECT_DIR}/_debug_ckpt/hybrid_wave0_dicom_film_ema_ds2}"
    LOG_ROOT="${LOG_ROOT_SMOKE:-${PROJECT_DIR}/train_logs/hybrid_wave0_dicom_film_ema_ds2_2gpu_smoke_$(date +%Y%m%d_%H%M%S)}"
    export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
fi

mkdir -p "$LOG_ROOT" "$CHECKPOINT_BASE"

echo "[PLAN] True DICOM-FiLM-only reference on GPU pair $GPU_PAIR"
echo "[PLAN] context=0 use_dicom=1"
echo "[PLAN] epochs=$EPOCHS batch=$BATCH_SIZE grad_accum=$GRAD_ACCUM iterations=$N_ITER_PER_EPOCH"
echo "[PLAN] EMA=1/0.999 DS2=1.0,0.3"
echo "[PLAN] Checkpoints=$CHECKPOINT_BASE"
echo "[PLAN] Logs=$LOG_ROOT"

PROJECT_DIR="$PROJECT_DIR" \
TRAIN_CSV="$TRAIN_CSV" VALID_CSV="$VALID_CSV" \
GPU_PAIRS="$GPU_PAIR" NUM_PROCESSES_PER_JOB=2 MAX_PARALLEL=1 \
ONLY_EXPERIMENTS=dicom_film \
SOFT_PROMPT_MODE=disabled TEXT_FEATURE_CACHE="" DICOM_PROMPT_MODE=none \
EXPERIMENT_NAME_SUFFIX="${EXPERIMENT_NAME_SUFFIX:-_wave0_dicom_film_ema_ds2}" \
CHECKPOINT_BASE="$CHECKPOINT_BASE" LOG_ROOT="$LOG_ROOT" \
BASE_PORT="${BASE_PORT:-30310}" STREAM_LOGS="${STREAM_LOGS:-1}" \
EPOCHS="$EPOCHS" BATCH_SIZE="$BATCH_SIZE" GRAD_ACCUM="$GRAD_ACCUM" \
N_ITER_PER_EPOCH="$N_ITER_PER_EPOCH" N_ITER_VALID="$N_ITER_VALID" \
NUM_WORKERS="$NUM_WORKERS" CHECKPOINT_INTERVAL="$CHECKPOINT_INTERVAL" \
LR="${LR:-1e-5}" DEEP_SUPERVISION=1 DEEP_SUPERVISION_WEIGHTS="1.0,0.3" \
USE_EMA=1 EMA_DECAY="${EMA_DECAY:-0.999}" \
AUTO_RESUME="${AUTO_RESUME:-1}" OVERWRITE_TRAIN="$OVERWRITE_TRAIN" \
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" \
NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}" \
bash "$BASE_LAUNCHER"
