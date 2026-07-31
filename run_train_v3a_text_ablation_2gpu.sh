#!/bin/bash
set -euo pipefail

# Safe v3a scheduler for two GPUs. Each training job is single-process; the
# two devices run independent experiments in two waves. Do not change this
# back to one 2-GPU DDP job: online Llama previously stalled in DDP prepare.

PROJECT_DIR="${PROJECT_DIR:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk}"
BASE_LAUNCHER="${BASE_LAUNCHER:-${PROJECT_DIR}/run_train_dicom_ablation_8gpu.sh}"
PYTHON_EXE="${PYTHON_EXE:-python}"
GPU_PAIR="${GPU_PAIR:-0,1}"
V3A_2GPU_JOBS="${V3A_2GPU_JOBS:-vision_control,soft_prompt_online,no_soft_online,no_soft_cached}"
TEXT_FEATURE_CACHE="${TEXT_FEATURE_CACHE:-${PROJECT_DIR}/text_feature_cache/llama2_safe_cc_nosoft.sqlite3}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/llama/safe_film_context_v3a_2gpu_waves}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/train_logs/llama_safe_film_context_v3a_2gpu_$(date +%Y%m%d_%H%M%S)}"
SMOKE_TEST="${SMOKE_TEST:-0}"
BASE_PORT="${BASE_PORT:-29700}"

IFS=',' read -r gpu0 gpu1 extra_gpu <<< "$GPU_PAIR"
if [ -z "${gpu0:-}" ] || [ -z "${gpu1:-}" ] || [ -n "${extra_gpu:-}" ]; then
    echo "[ERROR] GPU_PAIR must contain exactly two IDs, e.g. GPU_PAIR=0,1"
    exit 2
fi
if ! [[ "$gpu0" =~ ^[0-9]+$ && "$gpu1" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] GPU IDs must be non-negative integers: $GPU_PAIR"
    exit 2
fi
if [ "$gpu0" = "$gpu1" ]; then
    echo "[ERROR] GPU_PAIR must contain two distinct GPUs."
    exit 2
fi
if [ ! -f "$BASE_LAUNCHER" ]; then
    echo "[ERROR] Base launcher not found: $BASE_LAUNCHER"
    exit 2
fi
if [[ ",$V3A_2GPU_JOBS," == *",no_soft_cached,"* ]] \
   && [ ! -f "$TEXT_FEATURE_CACHE" ]; then
    echo "[ERROR] Text feature cache not found: $TEXT_FEATURE_CACHE"
    exit 2
fi

CUDA_VISIBLE_DEVICES="${gpu0},${gpu1}" \
EXPECTED_GPUS=2 \
"$PYTHON_EXE" "${PROJECT_DIR}/verify_runtime_environment.py"

# Preserve the former effective batch of 32 without cross-GPU DDP:
# batch_size 2 * grad_accum 16 * one process.
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
    CHECKPOINT_BASE="${CHECKPOINT_BASE_SMOKE:-${PROJECT_DIR}/_debug_ckpt/llama_safe_film_context_v3a_2gpu_waves}"
    LOG_ROOT="${LOG_ROOT_SMOKE:-${PROJECT_DIR}/train_logs/llama_safe_film_context_v3a_2gpu_smoke_$(date +%Y%m%d_%H%M%S)}"
    export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
fi

mkdir -p "$LOG_ROOT" "$CHECKPOINT_BASE"

PIDS=()
NAMES=()

selected() {
    local name="$1"
    [[ ",$V3A_2GPU_JOBS," == *",$name,"* ]]
}

launch_job() {
    local job_name="$1"
    local gpu="$2"
    local base_experiment="$3"
    local soft_prompt_mode="$4"
    local cache_path="$5"
    local port="$6"
    local suffix="_${job_name}_1gpu"
    local job_log_root="${LOG_ROOT}/${job_name}"
    local launcher_log="${LOG_ROOT}/${job_name}_launcher.log"

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

wait_wave() {
    local failed=0
    for index in "${!PIDS[@]}"; do
        if wait "${PIDS[$index]}"; then
            echo "[DONE] ${NAMES[$index]}"
        else
            echo "[ERROR] ${NAMES[$index]} failed. See ${LOG_ROOT}/${NAMES[$index]}_launcher.log"
            failed=1
        fi
    done
    PIDS=()
    NAMES=()
    if [ "$failed" -ne 0 ]; then
        exit 1
    fi
}

echo "[WAVE 1] vision control + learned soft prompt"
launch_job vision_control     "$gpu0" vision_only learned "" "$((BASE_PORT + 0))"
launch_job soft_prompt_online "$gpu1" text_safe  learned "" "$((BASE_PORT + 1))"
if [ "${#PIDS[@]}" -gt 0 ]; then
    wait_wave
fi

echo "[WAVE 2] no-soft online + no-soft cached"
launch_job no_soft_online "$gpu0" text_safe disabled ""                    "$((BASE_PORT + 2))"
launch_job no_soft_cached "$gpu1" text_safe disabled "$TEXT_FEATURE_CACHE" "$((BASE_PORT + 3))"
if [ "${#PIDS[@]}" -gt 0 ]; then
    wait_wave
fi

if ! selected vision_control \
   && ! selected soft_prompt_online \
   && ! selected no_soft_online \
   && ! selected no_soft_cached; then
    echo "[ERROR] V3A_2GPU_JOBS selected no known jobs: $V3A_2GPU_JOBS"
    exit 2
fi

echo "[DONE] All selected 2-GPU wave experiments completed: $LOG_ROOT"
