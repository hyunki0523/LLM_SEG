#!/bin/bash
set -euo pipefail

# Run one v3.4 cached-context fine-tuning branch on one two-GPU DDP pair.
# Select with V34_JOB=cached_control|cached_ds2|cached_ds3.

PROJECT_DIR="${PROJECT_DIR:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk}"
BASE_LAUNCHER="${BASE_LAUNCHER:-${PROJECT_DIR}/run_train_dicom_ablation_8gpu.sh}"
PYTHON_EXE="${PYTHON_EXE:-python}"
GPU_PAIR="${GPU_PAIR:-0,1}"
V34_JOB="${V34_JOB:-cached_control}"
TRAIN_CSV="${TRAIN_CSV:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_train_set.xlsx}"
VALID_CSV="${VALID_CSV:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_valid_set.xlsx}"
LLM_REPO="${LLM_REPO:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_seg_jhk/model_custom/llama2/Llama-2-7b-chat-hf}"
SHARED_CACHE="${SHARED_CACHE:-${PROJECT_DIR}/text_feature_cache/llama2_safe_cc_nosoft.sqlite3}"
PRETRAINED="${PRETRAINED:-/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/llama/safe_film_context_v3a_4gpu_parallel/text_safe_no_soft_cached_1gpu/final_model.pth}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/llama/safe_film_context_v34_partial_ds}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/train_logs/llama_safe_film_context_v34_${V34_JOB}_ddp2_$(date +%Y%m%d_%H%M%S)}"
SMOKE_TEST="${SMOKE_TEST:-0}"

IFS=',' read -r -a gpu_list <<< "$GPU_PAIR"
if [ "${#gpu_list[@]}" -ne 2 ] \
   || [ "${gpu_list[0]}" = "${gpu_list[1]}" ] \
   || ! [[ "${gpu_list[0]}" =~ ^[0-9]+$ ]] \
   || ! [[ "${gpu_list[1]}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] GPU_PAIR must contain two distinct IDs, e.g. GPU_PAIR=0,1"
    exit 2
fi

case "$V34_JOB" in
    cached_control)
        DEEP_SUPERVISION=0
        DEEP_SUPERVISION_WEIGHTS="1.0"
        ;;
    cached_ds2)
        DEEP_SUPERVISION=1
        DEEP_SUPERVISION_WEIGHTS="1.0,0.3"
        ;;
    cached_ds3)
        DEEP_SUPERVISION=1
        DEEP_SUPERVISION_WEIGHTS="1.0,0.3,0.1"
        ;;
    *)
        echo "[ERROR] Unknown V34_JOB: $V34_JOB"
        echo "        Choose cached_control, cached_ds2, or cached_ds3."
        exit 2
        ;;
esac

for required_file in "$BASE_LAUNCHER" "$SHARED_CACHE" "$PRETRAINED"; do
    if [ ! -f "$required_file" ]; then
        echo "[ERROR] Required file not found: $required_file"
        exit 2
    fi
done

CUDA_VISIBLE_DEVICES="$GPU_PAIR" \
EXPECTED_GPUS=2 \
"$PYTHON_EXE" "${PROJECT_DIR}/verify_runtime_environment.py"

"$PYTHON_EXE" "${PROJECT_DIR}/verify_text_feature_cache.py" \
    --csv "$TRAIN_CSV" \
    --csv "$VALID_CSV" \
    --llm-repo "$LLM_REPO" \
    --cache "$SHARED_CACHE" \
    --coverage-only

LOCAL_CACHE_DIR="${LOCAL_CACHE_DIR:-/tmp/llmseg_text_cache}"
TEXT_FEATURE_CACHE="${TEXT_FEATURE_CACHE:-${LOCAL_CACHE_DIR}/$(basename "$SHARED_CACHE")}"
mkdir -p "$(dirname "$TEXT_FEATURE_CACHE")" "$LOG_ROOT" "$CHECKPOINT_BASE"
echo "[CACHE] Copying cache to local storage: $TEXT_FEATURE_CACHE"
cp -f "$SHARED_CACHE" "$TEXT_FEATURE_CACHE"

# Keep the running Bash program stable if repository files are updated.
BASE_LAUNCHER_SNAPSHOT="${LOG_ROOT}/run_train_dicom_ablation_snapshot.sh"
cp "$BASE_LAUNCHER" "$BASE_LAUNCHER_SNAPSHOT"

export BATCH_SIZE="${BATCH_SIZE:-4}"
export GRAD_ACCUM="${GRAD_ACCUM:-4}"
export N_ITER_PER_EPOCH="${N_ITER_PER_EPOCH:-64}"
export N_ITER_VALID="${N_ITER_VALID:-13}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export EPOCHS="${EPOCHS:-60}"
export LR="${LR:-3e-6}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5}"

if [ "$SMOKE_TEST" = "1" ]; then
    export EPOCHS="${SMOKE_EPOCHS:-1}"
    export N_ITER_PER_EPOCH="${SMOKE_N_ITER_PER_EPOCH:-4}"
    export N_ITER_VALID="${SMOKE_N_ITER_VALID:-1}"
    export BATCH_SIZE="${SMOKE_BATCH_SIZE:-2}"
    export GRAD_ACCUM="${SMOKE_GRAD_ACCUM:-2}"
    export NUM_WORKERS="${SMOKE_NUM_WORKERS:-2}"
    export CHECKPOINT_INTERVAL=1
    export OVERWRITE_TRAIN=1
    CHECKPOINT_BASE="${CHECKPOINT_BASE_SMOKE:-${PROJECT_DIR}/_debug_ckpt/llama_safe_film_context_v34_2gpu}"
fi

echo "[V3.4] job=$V34_JOB pair=$GPU_PAIR deep_supervision=$DEEP_SUPERVISION weights=$DEEP_SUPERVISION_WEIGHTS"

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
PRETRAINED="$PRETRAINED" \
DEEP_SUPERVISION="$DEEP_SUPERVISION" \
DEEP_SUPERVISION_WEIGHTS="$DEEP_SUPERVISION_WEIGHTS" \
EXPERIMENT_NAME_SUFFIX="_v34_${V34_JOB}_ddp2" \
CHECKPOINT_BASE="$CHECKPOINT_BASE" \
LOG_ROOT="$LOG_ROOT" \
BASE_PORT="${BASE_PORT:-30100}" \
CHECK_IMAGE_PATHS=0 \
CHECK_DATASET_CONTRACT=0 \
TEXT_FUSION_WARMUP_EPOCHS=0 \
TEXT_FUSION_TRANSITION_EPOCHS=0 \
AUTO_RESUME="${AUTO_RESUME:-1}" \
OVERWRITE_TRAIN="${OVERWRITE_TRAIN:-0}" \
STREAM_LOGS="${STREAM_LOGS:-1}" \
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" \
NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}" \
bash "$BASE_LAUNCHER_SNAPSHOT"
