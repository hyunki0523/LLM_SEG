#!/bin/bash
set -euo pipefail

# Run the v3a no-soft cached-text experiment with one DDP process per GPU.
# The defaults preserve both the effective batch size and optimizer updates per
# epoch used by the single-GPU run:
#   single GPU: 256 iterations / accumulation 16 = 16 updates
#   two GPUs:   128 iterations / accumulation 8  = 16 updates

PROJECT_DIR="${PROJECT_DIR:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk}"
BASE_LAUNCHER="${BASE_LAUNCHER:-${PROJECT_DIR}/run_train_dicom_ablation_8gpu.sh}"
GPU_PAIR="${GPU_PAIR:-0,1}"
PYTHON_EXE="${PYTHON_EXE:-python}"
TRAIN_CSV="${TRAIN_CSV:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_train_set.xlsx}"
VALID_CSV="${VALID_CSV:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_valid_set.xlsx}"
LLM_REPO="${LLM_REPO:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_seg_jhk/model_custom/llama2/Llama-2-7b-chat-hf}"

IFS=',' read -r -a gpu_list <<< "$GPU_PAIR"
if [ "${#gpu_list[@]}" -ne 2 ] \
   || [ "${gpu_list[0]}" = "${gpu_list[1]}" ] \
   || ! [[ "${gpu_list[0]}" =~ ^[0-9]+$ ]] \
   || ! [[ "${gpu_list[1]}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] GPU_PAIR must contain two distinct GPU IDs, e.g. GPU_PAIR=0,1"
    exit 2
fi

SHARED_CACHE="${SHARED_CACHE:-${PROJECT_DIR}/text_feature_cache/llama2_safe_cc_nosoft.sqlite3}"
if [ -z "${TEXT_FEATURE_CACHE:-}" ]; then
    LOCAL_CACHE_DIR="${LOCAL_CACHE_DIR:-/tmp/llmseg_text_cache}"
    TEXT_FEATURE_CACHE="${LOCAL_CACHE_DIR}/$(basename "$SHARED_CACHE")"
    if [ ! -f "$SHARED_CACHE" ]; then
        echo "[ERROR] Shared text feature cache not found: $SHARED_CACHE"
        exit 2
    fi
    mkdir -p "$LOCAL_CACHE_DIR"
    echo "[CACHE] Copying the SQLite cache to local storage: $TEXT_FEATURE_CACHE"
    cp -f "$SHARED_CACHE" "$TEXT_FEATURE_CACHE"
fi
if [ ! -f "$TEXT_FEATURE_CACHE" ]; then
    echo "[ERROR] Text feature cache not found: $TEXT_FEATURE_CACHE"
    exit 2
fi
"$PYTHON_EXE" "${PROJECT_DIR}/verify_text_feature_cache.py" \
    --csv "$TRAIN_CSV" \
    --csv "$VALID_CSV" \
    --llm-repo "$LLM_REPO" \
    --cache "$TEXT_FEATURE_CACHE" \
    --coverage-only

CHECKPOINT_BASE="${CHECKPOINT_BASE:-/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/llama/safe_film_context_v3a_4gpu_parallel}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/train_logs/llama_safe_film_context_v3a_cached_ddp2_$(date +%Y%m%d_%H%M%S)}"

CUDA_VISIBLE_DEVICES="$GPU_PAIR" \
EXPECTED_GPUS=2 \
"$PYTHON_EXE" "${PROJECT_DIR}/verify_runtime_environment.py"

PROJECT_DIR="$PROJECT_DIR" \
TRAIN_CSV="$TRAIN_CSV" \
VALID_CSV="$VALID_CSV" \
LLM_REPO="$LLM_REPO" \
GPU_PAIRS="$GPU_PAIR" \
NUM_PROCESSES_PER_JOB=2 \
MAX_PARALLEL=1 \
ONLY_EXPERIMENTS=text_safe \
SOFT_PROMPT_MODE=disabled \
TEXT_FEATURE_CACHE="$TEXT_FEATURE_CACHE" \
EXPERIMENT_NAME_SUFFIX="${EXPERIMENT_NAME_SUFFIX:-_no_soft_cached_1gpu}" \
CHECKPOINT_BASE="$CHECKPOINT_BASE" \
LOG_ROOT="$LOG_ROOT" \
BASE_PORT="${BASE_PORT:-29803}" \
BATCH_SIZE="${BATCH_SIZE:-2}" \
GRAD_ACCUM="${GRAD_ACCUM:-8}" \
N_ITER_PER_EPOCH="${N_ITER_PER_EPOCH:-128}" \
N_ITER_VALID="${N_ITER_VALID:-25}" \
NUM_WORKERS="${NUM_WORKERS:-6}" \
AUTO_RESUME="${AUTO_RESUME:-1}" \
OVERWRITE_TRAIN="${OVERWRITE_TRAIN:-0}" \
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" \
NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}" \
bash "$BASE_LAUNCHER"
