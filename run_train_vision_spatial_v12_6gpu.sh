#!/bin/bash
set -euo pipefail

# Factorial follow-up to Vision-Balanced-v1.1 (the existing control):
#   GPU 0,1: deeper Z context only
#   GPU 2,3: brain-aware random negatives only
#   GPU 4,5: deeper Z context + brain-aware random negatives

PROJECT_DIR="${PROJECT_DIR:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5}"
SMOKE_TEST="${SMOKE_TEST:-0}"
AUTO_RESUME="${AUTO_RESUME:-0}"
SELECTED_JOBS="${VISION_V12_JOBS:-z64_context brain_random z64_brain}"
MANIFEST="${MANIFEST:-${PROJECT_DIR}/data_manifests/vision_balanced_v11.csv}"
NUM_WORKERS_PER_JOB="${NUM_WORKERS_PER_JOB:-2}"
SCREEN_OPTIMIZER_STEPS="${SCREEN_OPTIMIZER_STEPS:-2500}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/llama/vision_balanced_v12}"

read -r -a gpu_array <<< "$GPU_IDS"
if [ "${#gpu_array[@]}" -ne 6 ]; then
    echo "[ERROR] GPU_IDS must contain six space-separated IDs: $GPU_IDS"
    exit 2
fi
declare -A seen=()
for gpu in "${gpu_array[@]}"; do
    if ! [[ "$gpu" =~ ^[0-9]+$ ]] || [ -n "${seen[$gpu]:-}" ]; then
        echo "[ERROR] GPU_IDS must be six distinct non-negative integers: $GPU_IDS"
        exit 2
    fi
    seen[$gpu]=1
done
if [ ! -f "$MANIFEST" ]; then
    echo "[ERROR] Missing image-aligned manifest: $MANIFEST"
    exit 2
fi

run_kind=full
if [ "$SMOKE_TEST" = "1" ]; then run_kind=smoke; fi
LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/train_logs/vision_balanced_v12_6gpu_${run_kind}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_ROOT"

contains_job() { [[ " $SELECTED_JOBS " == *" $1 "* ]]; }

declare -A pids=()
declare -A logs=()
launch_job() {
    local job="$1" pair="$2" patch="$3" brain="$4" port="$5"
    local checkpoint_dir="${EXPERIMENT_ROOT}/${job}_seed42"
    if [ "$SMOKE_TEST" = "1" ]; then
        checkpoint_dir="${EXPERIMENT_ROOT}/smoke/${job}_seed42"
    fi
    local child_log="${LOG_ROOT}/${job}_launcher.log"
    local child_log_root="${LOG_ROOT}/${job}"
    mkdir -p "$checkpoint_dir" "$child_log_root"
    echo "[LAUNCH] $job GPU=$pair patch=($patch) brain_random=$brain -> $child_log"
    (
        PROJECT_DIR="$PROJECT_DIR" \
        GPU_PAIR="$pair" \
        MANIFEST="$MANIFEST" \
        SMOKE_TEST="$SMOKE_TEST" \
        AUTO_RESUME="$AUTO_RESUME" \
        CHECKPOINT_DIR="$checkpoint_dir" \
        LOG_ROOT="$child_log_root" \
        EXPERIMENT_NAME="vision_balanced_v12_${job}_seed42" \
        PATCH_SIZE="$patch" \
        SW_VALID_PATCH_SIZE="32 224 224" \
        BATCH_SIZE=2 \
        GRAD_ACCUM=8 \
        MAX_OPTIMIZER_STEPS=5000 \
        STOP_OPTIMIZER_STEPS="$SCREEN_OPTIMIZER_STEPS" \
        BRAIN_AWARE_RANDOM_SAMPLING="$brain" \
        NUM_WORKERS="$NUM_WORKERS_PER_JOB" \
        BASE_PORT="$port" \
        bash "${PROJECT_DIR}/run_train_vision_balanced_v1_2gpu.sh"
    ) >"$child_log" 2>&1 &
    pids[$job]=$!
    logs[$job]="$child_log"
}

if contains_job z64_context; then
    launch_job z64_context "${gpu_array[0]},${gpu_array[1]}" "64 192 192" 0 "${PORT_Z64:-30710}"
fi
if contains_job brain_random; then
    launch_job brain_random "${gpu_array[2]},${gpu_array[3]}" "32 224 224" 1 "${PORT_BRAIN:-30720}"
fi
if contains_job z64_brain; then
    launch_job z64_brain "${gpu_array[4]},${gpu_array[5]}" "64 192 192" 1 "${PORT_Z64_BRAIN:-30730}"
fi
if [ "${#pids[@]}" -eq 0 ]; then
    echo "[ERROR] No jobs selected. Valid: z64_context brain_random z64_brain"
    exit 2
fi

failed=0
for job in z64_context brain_random z64_brain; do
    if [ -z "${pids[$job]:-}" ]; then continue; fi
    if wait "${pids[$job]}"; then
        echo "[DONE] $job"
    else
        echo "[ERROR] $job failed. See ${logs[$job]}"
        failed=1
    fi
done
if [ "$failed" -ne 0 ]; then exit 1; fi
echo "[DONE] Vision spatial v1.2 jobs completed: $LOG_ROOT"
