#!/bin/bash
set -euo pipefail

# Wave 1: isolate the DICOM transport effect with soft prompts disabled.
#   A / GPU 0,1: CC only                 (LLM text; no DICOM)
#   C / GPU 2,3: CC + full DICOM text    (LLM text; no FiLM)
#   E / GPU 4,5: CC + DICOM FiLM         (separate FiLM path)
# Shared recipe: fresh/full tuning, EMA, decoder DS2, two-GPU DDP.

PROJECT_DIR="${PROJECT_DIR:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk}"
BASE_LAUNCHER="${BASE_LAUNCHER:-${PROJECT_DIR}/run_train_dicom_ablation_8gpu.sh}"
PYTHON_EXE="${PYTHON_EXE:-python}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5}"
WAVE1_JOBS="${WAVE1_JOBS:-cc_only,dicom_text_full,dicom_film}"
TRAIN_CSV="${TRAIN_CSV:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_train_set.xlsx}"
VALID_CSV="${VALID_CSV:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_valid_set.xlsx}"
LLM_REPO="${LLM_REPO:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_seg_jhk/model_custom/llama2/Llama-2-7b-chat-hf/}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/llama/hybrid_wave1_no_soft_ema_ds2}"
SMOKE_TEST="${SMOKE_TEST:-0}"
BASE_PORT="${BASE_PORT:-30100}"
BUILD_MISSING_CACHE="${BUILD_MISSING_CACHE:-1}"

SHARED_CACHE_DIR="${SHARED_CACHE_DIR:-${PROJECT_DIR}/text_feature_cache}"
CC_CACHE_SHARED="${CC_CACHE_SHARED:-${SHARED_CACHE_DIR}/llama2_safe_cc_nosoft.sqlite3}"
DICOM_TEXT_CACHE_SHARED="${DICOM_TEXT_CACHE_SHARED:-${SHARED_CACHE_DIR}/llama2_safe_cc_dicom_full_nosoft.sqlite3}"
LOCAL_CACHE_DIR="${LOCAL_CACHE_DIR:-/tmp/llmseg_text_cache}"
COPY_CACHE_TO_LOCAL="${COPY_CACHE_TO_LOCAL:-1}"

DEFAULT_IMAGE_PATH_REWRITE_FROM="/mnt/nas100/Brain_ER/data/BrainCT_NIfTIv2"
DEFAULT_IMAGE_PATH_REWRITE_TO="/mnt/nas100/Brain_ER/IDs/kevin/BrainCT_NIfTIv2"
IMAGE_PATH_REWRITE_FROM="${IMAGE_PATH_REWRITE_FROM:-$DEFAULT_IMAGE_PATH_REWRITE_FROM}"
IMAGE_PATH_REWRITE_TO="${IMAGE_PATH_REWRITE_TO:-$DEFAULT_IMAGE_PATH_REWRITE_TO}"

timestamp="$(date +%Y%m%d_%H%M%S)"
smoke_tag=""
if [ "$SMOKE_TEST" = "1" ]; then
    smoke_tag="_smoke"
fi
LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/train_logs/hybrid_wave1_no_soft_ema_ds2_6gpu${smoke_tag}_${timestamp}}"

read -r -a gpu_list <<< "$GPU_IDS"
if [ "${#gpu_list[@]}" -ne 6 ]; then
    echo "[ERROR] GPU_IDS must contain six IDs, e.g. GPU_IDS='0 1 2 3 4 5'"
    exit 2
fi
for gpu in "${gpu_list[@]}"; do
    if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] Invalid GPU ID: $gpu"
        exit 2
    fi
done
if [ "$(printf '%s\n' "${gpu_list[@]}" | sort -u | wc -l)" -ne 6 ]; then
    echo "[ERROR] GPU_IDS must contain six distinct devices."
    exit 2
fi
if [ ! -f "$BASE_LAUNCHER" ]; then
    echo "[ERROR] Base launcher not found: $BASE_LAUNCHER"
    exit 2
fi

selected() {
    local name="$1"
    [[ ",$WAVE1_JOBS," == *",$name,"* ]]
}

selected_count=0
for known_job in cc_only dicom_text_full dicom_film; do
    if selected "$known_job"; then
        selected_count=$((selected_count + 1))
    fi
done
if [ "$selected_count" -eq 0 ]; then
    echo "[ERROR] WAVE1_JOBS selected no known jobs: $WAVE1_JOBS"
    exit 2
fi

mkdir -p "$LOG_ROOT" "$CHECKPOINT_BASE" "$SHARED_CACHE_DIR" "$LOCAL_CACHE_DIR"
cd "$PROJECT_DIR"

selected_cuda_devices="$(IFS=,; echo "${gpu_list[*]}")"
CUDA_VISIBLE_DEVICES="$selected_cuda_devices" EXPECTED_GPUS=6 \
    "$PYTHON_EXE" "${PROJECT_DIR}/verify_runtime_environment.py"

export LLMSEG_SKIP_MISSING_IMAGE_PATHS="${SKIP_MISSING_IMAGE_PATHS:-1}"
export LLMSEG_IMAGE_PATH_REWRITE_FROM="$IMAGE_PATH_REWRITE_FROM"
export LLMSEG_IMAGE_PATH_REWRITE_TO="$IMAGE_PATH_REWRITE_TO"

echo "[CHECK] Auditing the Wave 1 safe prompt contracts."
"$PYTHON_EXE" audit_dataset_contract.py \
    --train-csv "$TRAIN_CSV" --valid-csv "$VALID_CSV" \
    --dicom-prompt-mode none \
    --output "$LOG_ROOT/dataset_contract_none.json"
"$PYTHON_EXE" audit_dataset_contract.py \
    --train-csv "$TRAIN_CSV" --valid-csv "$VALID_CSV" \
    --dicom-prompt-mode full \
    --output "$LOG_ROOT/dataset_contract_dicom_text_full.json"

echo "[CHECK] Checking train/valid image paths once before launching jobs."
path_check_args=(
    --train-csv "$TRAIN_CSV"
    --valid-csv "$VALID_CSV"
    --rewrite-from "$IMAGE_PATH_REWRITE_FROM"
    --rewrite-to "$IMAGE_PATH_REWRITE_TO"
)
if [ "${SKIP_MISSING_IMAGE_PATHS:-1}" = "1" ]; then
    path_check_args+=(--skip-missing)
fi
"$PYTHON_EXE" check_image_paths.py "${path_check_args[@]}"

ensure_cache() {
    local mode="$1"
    local cache_path="$2"
    local verify_args=(
        --csv "$TRAIN_CSV" --csv "$VALID_CSV"
        --llm-repo "$LLM_REPO"
        --cache "$cache_path"
        --dicom-prompt-mode "$mode"
        --coverage-only
    )
    if [ -f "$cache_path" ] \
       && "$PYTHON_EXE" verify_text_feature_cache.py "${verify_args[@]}"; then
        return
    fi
    if [ "$BUILD_MISSING_CACHE" != "1" ]; then
        echo "[ERROR] Missing/incomplete cache and BUILD_MISSING_CACHE=0: $cache_path"
        exit 1
    fi
    echo "[CACHE] Building/updating mode=$mode cache on physical GPU ${gpu_list[0]}: $cache_path"
    CUDA_VISIBLE_DEVICES="${gpu_list[0]}" "$PYTHON_EXE" precompute_text_features.py \
        --csv "$TRAIN_CSV" --csv "$VALID_CSV" \
        --llm-repo "$LLM_REPO" \
        --output "$cache_path" \
        --dicom-prompt-mode "$mode" \
        --batch-size "${CACHE_BATCH_SIZE:-8}" \
        --device cuda:0
    "$PYTHON_EXE" verify_text_feature_cache.py "${verify_args[@]}"
}

if selected cc_only || selected dicom_film; then
    ensure_cache none "$CC_CACHE_SHARED"
fi
if selected dicom_text_full; then
    ensure_cache full "$DICOM_TEXT_CACHE_SHARED"
fi

copy_local_cache() {
    local source_path="$1"
    if [ "$COPY_CACHE_TO_LOCAL" != "1" ]; then
        echo "[CACHE] Using shared cache in place: $source_path" >&2
        printf '%s' "$source_path"
        return
    fi
    local target_path="${LOCAL_CACHE_DIR}/$(basename "$source_path")"
    if [ ! -f "$target_path" ] \
       || [ "$source_path" -nt "$target_path" ] \
       || [ "$(stat -c %s "$target_path")" != "$(stat -c %s "$source_path")" ]; then
        local source_kb available_kb
        source_kb=$(( ( $(stat -c %s "$source_path") + 1023 ) / 1024 ))
        available_kb="$(df -Pk "$LOCAL_CACHE_DIR" | awk 'NR==2 {print $4}')"
        if [ "$available_kb" -le "$source_kb" ]; then
            echo "[ERROR] Insufficient local cache space: need ${source_kb} KiB, " \
                 "available ${available_kb} KiB in $LOCAL_CACHE_DIR." >&2
            echo "[HINT] Set COPY_CACHE_TO_LOCAL=0 to read the cache from shared storage." >&2
            return 1
        fi
        echo "[CACHE] Copying to local storage: $source_path -> $target_path" >&2
        cp -f "$source_path" "$target_path"
    else
        echo "[CACHE] Reusing local copy: $target_path" >&2
    fi
    printf '%s' "$target_path"
}

CC_CACHE_LOCAL=""
DICOM_TEXT_CACHE_LOCAL=""
if selected cc_only || selected dicom_film; then
    CC_CACHE_LOCAL="$(copy_local_cache "$CC_CACHE_SHARED")"
fi
if selected dicom_text_full; then
    DICOM_TEXT_CACHE_LOCAL="$(copy_local_cache "$DICOM_TEXT_CACHE_SHARED")"
fi

# Preserve 16 optimizer updates and an effective global batch of 32 per epoch:
# 128 micro-steps / grad_accum 8 / two DDP ranks, batch size 2 per rank.
export EPOCHS="${EPOCHS:-120}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export GRAD_ACCUM="${GRAD_ACCUM:-8}"
export N_ITER_PER_EPOCH="${N_ITER_PER_EPOCH:-128}"
export N_ITER_VALID="${N_ITER_VALID:-25}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10}"
export LR="${LR:-1e-5}"
export DEEP_SUPERVISION=1
export DEEP_SUPERVISION_WEIGHTS="${DEEP_SUPERVISION_WEIGHTS:-1.0,0.3}"
export USE_EMA=1
export EMA_DECAY="${EMA_DECAY:-0.999}"
export SOFT_PROMPT_MODE=disabled
export AUTO_RESUME="${AUTO_RESUME:-1}"
export OVERWRITE_TRAIN="${OVERWRITE_TRAIN:-0}"

if [ "$SMOKE_TEST" = "1" ]; then
    export EPOCHS="${SMOKE_EPOCHS:-1}"
    export N_ITER_PER_EPOCH="${SMOKE_N_ITER_PER_EPOCH:-2}"
    export N_ITER_VALID="${SMOKE_N_ITER_VALID:-1}"
    export BATCH_SIZE="${SMOKE_BATCH_SIZE:-1}"
    export GRAD_ACCUM="${SMOKE_GRAD_ACCUM:-1}"
    export NUM_WORKERS="${SMOKE_NUM_WORKERS:-2}"
    export CHECKPOINT_INTERVAL=1
    export OVERWRITE_TRAIN=1
    CHECKPOINT_BASE="${CHECKPOINT_BASE_SMOKE:-${PROJECT_DIR}/_debug_ckpt/hybrid_wave1_no_soft_ema_ds2}"
    mkdir -p "$CHECKPOINT_BASE"
    export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
fi

echo "[PLAN] Wave 1 jobs=$WAVE1_JOBS epochs=$EPOCHS EMA=$USE_EMA DS=$DEEP_SUPERVISION/$DEEP_SUPERVISION_WEIGHTS"
echo "[PLAN] Checkpoints=$CHECKPOINT_BASE"
echo "[PLAN] Logs=$LOG_ROOT"

# Snapshot prevents an active Bash process from observing later file edits.
BASE_LAUNCHER_SNAPSHOT="${LOG_ROOT}/run_train_dicom_ablation_snapshot.sh"
cp "$BASE_LAUNCHER" "$BASE_LAUNCHER_SNAPSHOT"

PIDS=()
NAMES=()

launch_job() {
    local job_name="$1"
    local gpu_pair="$2"
    local base_experiment="$3"
    local dicom_prompt_mode="$4"
    local cache_path="$5"
    local port="$6"
    local suffix="_wave1_${job_name}_nosoft_ema_ds2"
    local job_log_root="${LOG_ROOT}/${job_name}"
    local launcher_log="${LOG_ROOT}/${job_name}_launcher.log"

    if ! selected "$job_name"; then
        return
    fi
    echo "[LAUNCH] $job_name on GPU pair $gpu_pair -> $launcher_log"
    echo "[TRAIN LOG] ${job_log_root}/${base_experiment}${suffix}.log"
    (
        PROJECT_DIR="$PROJECT_DIR" \
        TRAIN_CSV="$TRAIN_CSV" VALID_CSV="$VALID_CSV" \
        LLM_REPO="$LLM_REPO" \
        GPU_PAIRS="$gpu_pair" NUM_PROCESSES_PER_JOB=2 MAX_PARALLEL=1 \
        ONLY_EXPERIMENTS="$base_experiment" \
        DICOM_PROMPT_MODE="$dicom_prompt_mode" \
        TEXT_FEATURE_CACHE="$cache_path" \
        EXPERIMENT_NAME_SUFFIX="$suffix" \
        CHECKPOINT_BASE="$CHECKPOINT_BASE" LOG_ROOT="$job_log_root" \
        BASE_PORT="$port" STREAM_LOGS=0 \
        CHECK_IMAGE_PATHS=0 CHECK_DATASET_CONTRACT=0 \
        NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" \
        NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}" \
        bash "$BASE_LAUNCHER_SNAPSHOT"
    ) >"$launcher_log" 2>&1 &
    PIDS+=("$!")
    NAMES+=("$job_name")
}

launch_job cc_only \
    "${gpu_list[0]},${gpu_list[1]}" text_safe none "$CC_CACHE_LOCAL" "$((BASE_PORT + 0))"
launch_job dicom_text_full \
    "${gpu_list[2]},${gpu_list[3]}" text_safe full "$DICOM_TEXT_CACHE_LOCAL" "$((BASE_PORT + 1))"
launch_job dicom_film \
    "${gpu_list[4]},${gpu_list[5]}" dicom_text_safe none "$CC_CACHE_LOCAL" "$((BASE_PORT + 2))"

failed=0
for index in "${!PIDS[@]}"; do
    if wait "${PIDS[$index]}"; then
        echo "[DONE] ${NAMES[$index]}"
    else
        echo "[ERROR] ${NAMES[$index]} failed: ${LOG_ROOT}/${NAMES[$index]}_launcher.log"
        failed=1
    fi
done
if [ "$failed" -ne 0 ]; then
    exit 1
fi
echo "[DONE] Wave 1 completed: $LOG_ROOT"
