#!/bin/bash
set -euo pipefail

# Run four controlled v3a jobs concurrently, one process per 98 GB GPU.
# This avoids broadcasting the frozen Llama through DDP while keeping the
# effective batch size equal across the four comparisons.

PROJECT_DIR="${PROJECT_DIR:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk}"
BASE_LAUNCHER="${BASE_LAUNCHER:-${PROJECT_DIR}/run_train_dicom_ablation_8gpu.sh}"
PYTHON_EXE="${PYTHON_EXE:-python}"
GPU_IDS="${GPU_IDS:-}"
V3A_4GPU_JOBS="${V3A_4GPU_JOBS:-vision_control,soft_prompt_online,no_soft_online,no_soft_cached}"
TEXT_FEATURE_CACHE="${TEXT_FEATURE_CACHE:-${PROJECT_DIR}/text_feature_cache/llama2_safe_cc_nosoft.sqlite3}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/llama/safe_film_context_v3a_4gpu_parallel}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/train_logs/llama_safe_film_context_v3a_4gpu_$(date +%Y%m%d_%H%M%S)}"
SMOKE_TEST="${SMOKE_TEST:-0}"
BASE_PORT="${BASE_PORT:-29800}"

read -r -a gpu_list <<< "$GPU_IDS"
if [ "${#gpu_list[@]}" -eq 1 ]; then
    case "$V3A_4GPU_JOBS" in
        vision_control|soft_prompt_online|no_soft_online|no_soft_cached) ;;
        *)
            echo "[ERROR] Single-GPU mode requires exactly one known V3A_4GPU_JOBS value."
            exit 2
            ;;
    esac
elif [ "${#gpu_list[@]}" -ne 4 ]; then
    echo "[ERROR] Set either one GPU for one selected job or four GPUs for the full run."
    echo "        Examples: GPU_IDS='3' V3A_4GPU_JOBS=no_soft_cached"
    echo "                  GPU_IDS='0 1 2 3'"
    exit 2
fi
for gpu in "${gpu_list[@]}"; do
    if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] Invalid GPU ID: $gpu"
        exit 2
    fi
done
if [ "$(printf '%s\n' "${gpu_list[@]}" | sort -u | wc -l)" -ne "${#gpu_list[@]}" ]; then
    echo "[ERROR] GPU_IDS must contain distinct devices."
    exit 2
fi
if [ ! -f "$BASE_LAUNCHER" ]; then
    echo "[ERROR] Base launcher not found: $BASE_LAUNCHER"
    exit 2
fi
if [[ ",$V3A_4GPU_JOBS," == *",no_soft_cached,"* ]] \
   && [ ! -f "$TEXT_FEATURE_CACHE" ]; then
    echo "[ERROR] Text feature cache not found: $TEXT_FEATURE_CACHE"
    exit 2
fi

selected_cuda_devices="$(IFS=,; echo "${gpu_list[*]}")"
CUDA_VISIBLE_DEVICES="$selected_cuda_devices" \
EXPECTED_GPUS="${#gpu_list[@]}" \
"$PYTHON_EXE" "${PROJECT_DIR}/verify_runtime_environment.py"

# One GPU still uses the same effective batch as the former 2-GPU
# batch=2/accum=8 setting: 2 * 16 * 1 = 32.
export BATCH_SIZE="${BATCH_SIZE:-2}"
export GRAD_ACCUM="${GRAD_ACCUM:-16}"
export NUM_WORKERS="${NUM_WORKERS:-6}"

if [ "$SMOKE_TEST" = "1" ]; then
    export EPOCHS="${EPOCHS:-1}"
    export N_ITER_PER_EPOCH="${N_ITER_PER_EPOCH:-2}"
    export N_ITER_VALID="${N_ITER_VALID:-1}"
    export BATCH_SIZE="${SMOKE_BATCH_SIZE:-1}"
    export GRAD_ACCUM="${SMOKE_GRAD_ACCUM:-1}"
    export NUM_WORKERS="${SMOKE_NUM_WORKERS:-2}"
    export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1}"
    export OVERWRITE_TRAIN="${OVERWRITE_TRAIN:-1}"
    CHECKPOINT_BASE="${CHECKPOINT_BASE_SMOKE:-${PROJECT_DIR}/_debug_ckpt/llama_safe_film_context_v3a_4gpu}"
    LOG_ROOT="${LOG_ROOT_SMOKE:-${PROJECT_DIR}/train_logs/llama_safe_film_context_v3a_4gpu_smoke_$(date +%Y%m%d_%H%M%S)}"
    export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
fi

mkdir -p "$LOG_ROOT" "$CHECKPOINT_BASE"

PIDS=()
NAMES=()

selected() {
    local name="$1"
    [[ ",$V3A_4GPU_JOBS," == *",$name,"* ]]
}

launch_job() {
    local job_name="$1"
    local gpu="$2"
    local base_experiment="$3"
    local soft_prompt_mode="$4"
    local cache_path="$5"
    local suffix="_${job_name}_1gpu"
    local job_log_root="${LOG_ROOT}/${job_name}"
    local launcher_log="${LOG_ROOT}/${job_name}_launcher.log"
    local port="$6"

    if ! selected "$job_name"; then
        return
    fi

    echo "[LAUNCH] $job_name on GPU $gpu -> $launcher_log"
    (
        PROJECT_DIR="$PROJECT_DIR" \
        GPU_PAIRS="$gpu" \
        NUM_PROCESSES_PER_JOB=1 \
        MAX_PARALLEL=1 \
        ONLY_EXPERIMENTS="$base_experiment" \
        SOFT_PROMPT_MODE="$soft_prompt_mode" \
        TEXT_FEATURE_CACHE="$cache_path" \
        EXPERIMENT_NAME_SUFFIX="$suffix" \
        CHECKPOINT_BASE="$CHECKPOINT_BASE" \
        LOG_ROOT="$job_log_root" \
        BASE_PORT="$port" \
        bash "$BASE_LAUNCHER"
    ) >"$launcher_log" 2>&1 &
    PIDS+=("$!")
    NAMES+=("$job_name")
}

gpu_for_slot() {
    local slot="$1"
    if [ "${#gpu_list[@]}" -eq 1 ]; then
        echo "${gpu_list[0]}"
    else
        echo "${gpu_list[$slot]}"
    fi
}

launch_job vision_control       "$(gpu_for_slot 0)" vision_only learned  ""                    "$((BASE_PORT + 0))"
launch_job soft_prompt_online   "$(gpu_for_slot 1)" text_safe  learned  ""                    "$((BASE_PORT + 1))"
launch_job no_soft_online       "$(gpu_for_slot 2)" text_safe  disabled ""                    "$((BASE_PORT + 2))"
launch_job no_soft_cached       "$(gpu_for_slot 3)" text_safe  disabled "$TEXT_FEATURE_CACHE" "$((BASE_PORT + 3))"

if [ "${#PIDS[@]}" -eq 0 ]; then
    echo "[ERROR] V3A_4GPU_JOBS selected no known jobs: $V3A_4GPU_JOBS"
    exit 2
fi

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

echo "[DONE] All selected 4-GPU parallel v3a jobs completed: $LOG_ROOT"
