#!/bin/bash
set -euo pipefail

# Controlled 60-epoch fine-tuning from the completed v3a cached-context model:
#   GPU 0,1 -> no deep supervision (continued-training control)
#   GPU 2,3 -> final + one high-resolution auxiliary head
#   GPU 4,5 -> final + two high/mid-resolution auxiliary heads

PROJECT_DIR="${PROJECT_DIR:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk}"
BASE_LAUNCHER="${BASE_LAUNCHER:-${PROJECT_DIR}/run_train_dicom_ablation_8gpu.sh}"
PYTHON_EXE="${PYTHON_EXE:-python}"
GPU_IDS="${GPU_IDS:-}"
TRAIN_CSV="${TRAIN_CSV:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_train_set.xlsx}"
VALID_CSV="${VALID_CSV:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_valid_set.xlsx}"
LLM_REPO="${LLM_REPO:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_seg_jhk/model_custom/llama2/Llama-2-7b-chat-hf}"
SHARED_CACHE="${SHARED_CACHE:-${PROJECT_DIR}/text_feature_cache/llama2_safe_cc_nosoft.sqlite3}"
PRETRAINED="${PRETRAINED:-/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/llama/safe_film_context_v3a_4gpu_parallel/text_safe_no_soft_cached_1gpu/final_model.pth}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/llama/safe_film_context_v34_partial_ds}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/train_logs/llama_safe_film_context_v34_partial_ds_$(date +%Y%m%d_%H%M%S)}"
BASE_PORT="${BASE_PORT:-30000}"
SMOKE_TEST="${SMOKE_TEST:-0}"

read -r -a gpu_list <<< "$GPU_IDS"
if [ "${#gpu_list[@]}" -ne 6 ]; then
    echo "[ERROR] Explicitly set six GPU IDs, e.g. GPU_IDS='0 1 2 3 4 5'"
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
for required_file in "$BASE_LAUNCHER" "$SHARED_CACHE" "$PRETRAINED"; do
    if [ ! -f "$required_file" ]; then
        echo "[ERROR] Required file not found: $required_file"
        exit 2
    fi
done

selected_cuda_devices="$(IFS=,; echo "${gpu_list[*]}")"
CUDA_VISIBLE_DEVICES="$selected_cuda_devices" \
EXPECTED_GPUS=6 \
"$PYTHON_EXE" "${PROJECT_DIR}/verify_runtime_environment.py"

mkdir -p "$LOG_ROOT" "$CHECKPOINT_BASE"

"$PYTHON_EXE" "${PROJECT_DIR}/audit_dataset_contract.py" \
    --train-csv "$TRAIN_CSV" \
    --valid-csv "$VALID_CSV" \
    --output "$LOG_ROOT/dataset_contract_report.json"

"$PYTHON_EXE" "${PROJECT_DIR}/verify_text_feature_cache.py" \
    --csv "$TRAIN_CSV" \
    --csv "$VALID_CSV" \
    --llm-repo "$LLM_REPO" \
    --cache "$SHARED_CACHE" \
    --coverage-only

LOCAL_CACHE_DIR="${LOCAL_CACHE_DIR:-/tmp/llmseg_text_cache}"
TEXT_FEATURE_CACHE="${TEXT_FEATURE_CACHE:-${LOCAL_CACHE_DIR}/$(basename "$SHARED_CACHE")}"
mkdir -p "$(dirname "$TEXT_FEATURE_CACHE")"
echo "[CACHE] Copying cache to local storage: $TEXT_FEATURE_CACHE"
cp -f "$SHARED_CACHE" "$TEXT_FEATURE_CACHE"

# Use an immutable launcher snapshot for all three concurrent jobs.
BASE_LAUNCHER_SNAPSHOT="${LOG_ROOT}/run_train_dicom_ablation_snapshot.sh"
cp "$BASE_LAUNCHER" "$BASE_LAUNCHER_SNAPSHOT"
BASE_LAUNCHER="$BASE_LAUNCHER_SNAPSHOT"

export BATCH_SIZE="${BATCH_SIZE:-4}"
export GRAD_ACCUM="${GRAD_ACCUM:-4}"
export N_ITER_PER_EPOCH="${N_ITER_PER_EPOCH:-64}"
export N_ITER_VALID="${N_ITER_VALID:-13}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export EPOCHS="${EPOCHS:-60}"
export LR="${LR:-3e-6}"
export WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5}"
export TEXT_FUSION_WARMUP_EPOCHS=0
export TEXT_FUSION_TRANSITION_EPOCHS=0

if [ "$SMOKE_TEST" = "1" ]; then
    export EPOCHS="${SMOKE_EPOCHS:-1}"
    export N_ITER_PER_EPOCH="${SMOKE_N_ITER_PER_EPOCH:-4}"
    export N_ITER_VALID="${SMOKE_N_ITER_VALID:-1}"
    export BATCH_SIZE="${SMOKE_BATCH_SIZE:-2}"
    export GRAD_ACCUM="${SMOKE_GRAD_ACCUM:-2}"
    export NUM_WORKERS="${SMOKE_NUM_WORKERS:-2}"
    export CHECKPOINT_INTERVAL=1
    export OVERWRITE_TRAIN=1
    CHECKPOINT_BASE="${CHECKPOINT_BASE_SMOKE:-${PROJECT_DIR}/_debug_ckpt/llama_safe_film_context_v34_partial_ds}"
fi

PIDS=()
NAMES=()

launch_job() {
    local job_name="$1"
    local gpu_pair="$2"
    local deep_supervision="$3"
    local deep_supervision_weights="$4"
    local port="$5"
    local suffix="_v34_${job_name}_ddp2"
    local job_log_root="${LOG_ROOT}/${job_name}"
    local launcher_log="${LOG_ROOT}/${job_name}_launcher.log"
    local train_log="${job_log_root}/text_safe${suffix}.log"

    echo "[LAUNCH] $job_name on GPU pair $gpu_pair -> $launcher_log"
    echo "[TRAIN LOG] $train_log"
    (
        PROJECT_DIR="$PROJECT_DIR" \
        TRAIN_CSV="$TRAIN_CSV" \
        VALID_CSV="$VALID_CSV" \
        LLM_REPO="$LLM_REPO" \
        GPU_PAIRS="$gpu_pair" \
        NUM_PROCESSES_PER_JOB=2 \
        MAX_PARALLEL=1 \
        ONLY_EXPERIMENTS=text_safe \
        SOFT_PROMPT_MODE=disabled \
        TEXT_FEATURE_CACHE="$TEXT_FEATURE_CACHE" \
        PRETRAINED="$PRETRAINED" \
        DEEP_SUPERVISION="$deep_supervision" \
        DEEP_SUPERVISION_WEIGHTS="$deep_supervision_weights" \
        EXPERIMENT_NAME_SUFFIX="$suffix" \
        CHECKPOINT_BASE="$CHECKPOINT_BASE" \
        LOG_ROOT="$job_log_root" \
        BASE_PORT="$port" \
        CHECK_IMAGE_PATHS=0 \
        CHECK_DATASET_CONTRACT=0 \
        AUTO_RESUME="${AUTO_RESUME:-1}" \
        OVERWRITE_TRAIN="${OVERWRITE_TRAIN:-0}" \
        STREAM_LOGS=0 \
        NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" \
        NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}" \
        bash "$BASE_LAUNCHER"
    ) >"$launcher_log" 2>&1 &
    PIDS+=("$!")
    NAMES+=("$job_name")
}

launch_job cached_control "${gpu_list[0]},${gpu_list[1]}" 0 "1.0" "$((BASE_PORT + 0))"
launch_job cached_ds2 "${gpu_list[2]},${gpu_list[3]}" 1 "1.0,0.3" "$((BASE_PORT + 1))"
launch_job cached_ds3 "${gpu_list[4]},${gpu_list[5]}" 1 "1.0,0.3,0.1" "$((BASE_PORT + 2))"

failed=0
for index in "${!PIDS[@]}"; do
    if wait "${PIDS[$index]}"; then
        echo "[DONE] ${NAMES[$index]}"
    else
        echo "[ERROR] ${NAMES[$index]} failed. See ${LOG_ROOT}/${NAMES[$index]}_launcher.log"
        failed=1
    fi
done

if [ "$failed" -ne 0 ]; then
    exit 1
fi

echo "[DONE] v3.4 partial deep-supervision ablation completed: $LOG_ROOT"
