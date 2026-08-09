#!/bin/bash
set -euo pipefail

# Three matched 2-GPU DDP jobs on a six-GPU node:
#   0,1: DICOM-FiLM seed 42
#   2,3: Vision-only seed 43
#   4,5: DICOM-FiLM seed 43
# Vision-only seed 42 is expected to run on the separate two-GPU node.

PROJECT_DIR="${PROJECT_DIR:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk}"
PYTHON_EXE="${PYTHON_EXE:-python}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5}"
TRAIN_CSV="${TRAIN_CSV:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_train_set.xlsx}"
VALID_CSV="${VALID_CSV:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_valid_set.xlsx}"
MANIFEST="${MANIFEST:-${PROJECT_DIR}/data_manifests/vision_balanced_v11.csv}"
MANIFEST_WORKERS="${MANIFEST_WORKERS:-12}"
SMOKE_TEST="${SMOKE_TEST:-0}"
SELECTED_JOBS="${BALANCED_V1_JOBS:-dicom_seed42 vision_seed43 dicom_seed43}"

read -r -a gpu_array <<< "$GPU_IDS"
if [ "${#gpu_array[@]}" -ne 6 ]; then
    echo "[ERROR] GPU_IDS must contain six space-separated IDs: $GPU_IDS"
    exit 2
fi
declare -A seen_gpu=()
for gpu in "${gpu_array[@]}"; do
    if ! [[ "$gpu" =~ ^[0-9]+$ ]] || [ -n "${seen_gpu[$gpu]:-}" ]; then
        echo "[ERROR] GPU_IDS must be six distinct non-negative integers: $GPU_IDS"
        exit 2
    fi
    seen_gpu[$gpu]=1
done

run_kind="full"
auto_resume_default=1
if [ "$SMOKE_TEST" = "1" ]; then
    run_kind="smoke"
    auto_resume_default=0
fi
AUTO_RESUME="${AUTO_RESUME:-$auto_resume_default}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/train_logs/balanced_v1_6gpu_pairs_${run_kind}_$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/llama}"

export LLMSEG_IMAGE_PATH_REWRITE_FROM="${LLMSEG_IMAGE_PATH_REWRITE_FROM:-/mnt/nas100/Brain_ER/data/BrainCT_NIfTIv2}"
export LLMSEG_IMAGE_PATH_REWRITE_TO="${LLMSEG_IMAGE_PATH_REWRITE_TO:-/mnt/nas100/Brain_ER/IDs/kevin/BrainCT_NIfTIv2}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"

cd "$PROJECT_DIR"
mkdir -p "$LOG_ROOT" "$(dirname "$MANIFEST")"

if [ ! -f "$MANIFEST" ]; then
    echo "[MANIFEST] Building once before parallel launch: $MANIFEST"
    "$PYTHON_EXE" -u build_vision_sampling_manifest.py \
        --train-csv "$TRAIN_CSV" \
        --valid-csv "$VALID_CSV" \
        --output "$MANIFEST" \
        --workers "$MANIFEST_WORKERS" \
        2>&1 | tee "${LOG_ROOT}/manifest.log"
else
    echo "[MANIFEST] Reusing: $MANIFEST"
fi

if [ -f "${PROJECT_DIR}/verify_runtime_environment.py" ]; then
    EXPECTED_GPUS=6 "$PYTHON_EXE" "${PROJECT_DIR}/verify_runtime_environment.py"
fi

contains_job() {
    local wanted="$1"
    [[ " $SELECTED_JOBS " == *" $wanted "* ]]
}

checkpoint_for() {
    local job="$1"
    if [ "$SMOKE_TEST" = "1" ]; then
        echo "${EXPERIMENT_ROOT}/balanced_v11_6gpu_smoke/${job}"
        return
    fi
    case "$job" in
        dicom_seed42) echo "${EXPERIMENT_ROOT}/dicom_film_balanced_v11/seed42" ;;
        vision_seed43) echo "${EXPERIMENT_ROOT}/vision_balanced_v11/vision_only_seed43" ;;
        dicom_seed43) echo "${EXPERIMENT_ROOT}/dicom_film_balanced_v11/seed43" ;;
        *) echo "[ERROR] Unknown job: $job" >&2; return 2 ;;
    esac
}

declare -A job_pids=()
declare -A job_logs=()

launch_job() {
    local job="$1"
    local launcher="$2"
    local pair="$3"
    local seed="$4"
    local port="$5"
    local checkpoint_dir
    checkpoint_dir="$(checkpoint_for "$job")"
    local launcher_log="${LOG_ROOT}/${job}_launcher.log"
    local child_log_root="${LOG_ROOT}/${job}"
    mkdir -p "$child_log_root" "$checkpoint_dir"
    echo "[LAUNCH] $job seed=$seed GPU pair $pair -> $launcher_log"
    (
        PROJECT_DIR="$PROJECT_DIR" \
        PYTHON_EXE="$PYTHON_EXE" \
        GPU_PAIR="$pair" \
        TRAIN_CSV="$TRAIN_CSV" \
        VALID_CSV="$VALID_CSV" \
        MANIFEST="$MANIFEST" \
        CHECKPOINT_DIR="$checkpoint_dir" \
        LOG_ROOT="$child_log_root" \
        SMOKE_TEST="$SMOKE_TEST" \
        AUTO_RESUME="$AUTO_RESUME" \
        SEED="$seed" \
        BASE_PORT="$port" \
        bash "$launcher"
    ) >"$launcher_log" 2>&1 &
    job_pids[$job]=$!
    job_logs[$job]="$launcher_log"
}

if contains_job dicom_seed42; then
    launch_job \
        dicom_seed42 \
        "${PROJECT_DIR}/run_train_dicom_film_balanced_v1_2gpu.sh" \
        "${gpu_array[0]},${gpu_array[1]}" 42 "${PORT_DICOM42:-30610}"
fi
if contains_job vision_seed43; then
    launch_job \
        vision_seed43 \
        "${PROJECT_DIR}/run_train_vision_balanced_v1_2gpu.sh" \
        "${gpu_array[2]},${gpu_array[3]}" 43 "${PORT_VISION43:-30620}"
fi
if contains_job dicom_seed43; then
    launch_job \
        dicom_seed43 \
        "${PROJECT_DIR}/run_train_dicom_film_balanced_v1_2gpu.sh" \
        "${gpu_array[4]},${gpu_array[5]}" 43 "${PORT_DICOM43:-30630}"
fi

if [ "${#job_pids[@]}" -eq 0 ]; then
    echo "[ERROR] No jobs selected. Valid jobs: dicom_seed42 vision_seed43 dicom_seed43"
    exit 2
fi

failed=0
for job in dicom_seed42 vision_seed43 dicom_seed43; do
    if [ -z "${job_pids[$job]:-}" ]; then
        continue
    fi
    if wait "${job_pids[$job]}"; then
        echo "[DONE] $job"
    else
        echo "[ERROR] $job failed. See ${job_logs[$job]}"
        failed=1
    fi
done

if [ "$failed" -ne 0 ]; then
    exit 1
fi
echo "[DONE] All selected balanced-v1 jobs completed: $LOG_ROOT"
