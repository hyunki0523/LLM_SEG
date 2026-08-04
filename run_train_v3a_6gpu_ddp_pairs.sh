#!/bin/bash
set -euo pipefail

# Run the three successful v3a branches concurrently with two-GPU DDP:
#   vision_control     -> GPU 0,1
#   soft_prompt_online -> GPU 2,3
#   no_soft_online     -> GPU 4,5
# no_soft_cached is intentionally excluded and has its own two-GPU launcher.

PROJECT_DIR="${PROJECT_DIR:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk}"
BASE_LAUNCHER="${BASE_LAUNCHER:-${PROJECT_DIR}/run_train_dicom_ablation_8gpu.sh}"
PYTHON_EXE="${PYTHON_EXE:-python}"
GPU_IDS="${GPU_IDS:-}"
V3A_6GPU_JOBS="${V3A_6GPU_JOBS:-vision_control,soft_prompt_online,no_soft_online}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/llama/safe_film_context_v3a_4gpu_parallel}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/train_logs/llama_safe_film_context_v3a_6gpu_ddp2_$(date +%Y%m%d_%H%M%S)}"
SMOKE_TEST="${SMOKE_TEST:-0}"
BASE_PORT="${BASE_PORT:-29900}"

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
if [ ! -f "$BASE_LAUNCHER" ]; then
    echo "[ERROR] Base launcher not found: $BASE_LAUNCHER"
    exit 2
fi

selected() {
    local name="$1"
    [[ ",$V3A_6GPU_JOBS," == *",$name,"* ]]
}

selected_count=0
for known_job in vision_control soft_prompt_online no_soft_online; do
    if selected "$known_job"; then
        selected_count=$((selected_count + 1))
    fi
done
if [ "$selected_count" -eq 0 ]; then
    echo "[ERROR] V3A_6GPU_JOBS selected no known jobs: $V3A_6GPU_JOBS"
    exit 2
fi

selected_cuda_devices="$(IFS=,; echo "${gpu_list[*]}")"
CUDA_VISIBLE_DEVICES="$selected_cuda_devices" \
EXPECTED_GPUS=6 \
"$PYTHON_EXE" "${PROJECT_DIR}/verify_runtime_environment.py"

# Preserve the single-GPU schedule:
#   old: 256 iterations / accumulation 16 / one GPU = 16 updates, batch 32
#   new: 128 iterations / accumulation 8  / two GPUs = 16 updates, batch 32
export BATCH_SIZE="${BATCH_SIZE:-2}"
export GRAD_ACCUM="${GRAD_ACCUM:-8}"
export N_ITER_PER_EPOCH="${N_ITER_PER_EPOCH:-128}"
export N_ITER_VALID="${N_ITER_VALID:-25}"
export NUM_WORKERS="${NUM_WORKERS:-4}"

if [ "$SMOKE_TEST" = "1" ]; then
    export EPOCHS="${EPOCHS:-1}"
    export N_ITER_PER_EPOCH="${SMOKE_N_ITER_PER_EPOCH:-2}"
    export N_ITER_VALID="${SMOKE_N_ITER_VALID:-1}"
    export BATCH_SIZE="${SMOKE_BATCH_SIZE:-1}"
    export GRAD_ACCUM="${SMOKE_GRAD_ACCUM:-1}"
    export NUM_WORKERS="${SMOKE_NUM_WORKERS:-2}"
    export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1}"
    export OVERWRITE_TRAIN="${OVERWRITE_TRAIN:-1}"
    CHECKPOINT_BASE="${CHECKPOINT_BASE_SMOKE:-${PROJECT_DIR}/_debug_ckpt/llama_safe_film_context_v3a_6gpu_ddp2}"
    LOG_ROOT="${LOG_ROOT_SMOKE:-${PROJECT_DIR}/train_logs/llama_safe_film_context_v3a_6gpu_ddp2_smoke_$(date +%Y%m%d_%H%M%S)}"
    export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
fi

mkdir -p "$LOG_ROOT" "$CHECKPOINT_BASE"

# Bash may continue reading a script while it runs. Use an immutable snapshot
# so repository updates cannot shift the file offset of active launchers.
BASE_LAUNCHER_SNAPSHOT="${LOG_ROOT}/run_train_dicom_ablation_snapshot.sh"
cp "$BASE_LAUNCHER" "$BASE_LAUNCHER_SNAPSHOT"
BASE_LAUNCHER="$BASE_LAUNCHER_SNAPSHOT"

PIDS=()
NAMES=()

launch_job() {
    local job_name="$1"
    local gpu_pair="$2"
    local base_experiment="$3"
    local soft_prompt_mode="$4"
    local port="$5"
    # Keep the original suffix so AUTO_RESUME finds the existing checkpoint.
    local suffix="_${job_name}_1gpu"
    local job_log_root="${LOG_ROOT}/${job_name}"
    local launcher_log="${LOG_ROOT}/${job_name}_launcher.log"

    if ! selected "$job_name"; then
        return
    fi

    echo "[LAUNCH] $job_name on GPU pair $gpu_pair -> $launcher_log"
    echo "[TRAIN LOG] ${job_log_root}/${base_experiment}${suffix}.log"
    (
        PROJECT_DIR="$PROJECT_DIR" \
        GPU_PAIRS="$gpu_pair" \
        NUM_PROCESSES_PER_JOB=2 \
        MAX_PARALLEL=1 \
        ONLY_EXPERIMENTS="$base_experiment" \
        SOFT_PROMPT_MODE="$soft_prompt_mode" \
        TEXT_FEATURE_CACHE="" \
        EXPERIMENT_NAME_SUFFIX="$suffix" \
        CHECKPOINT_BASE="$CHECKPOINT_BASE" \
        LOG_ROOT="$job_log_root" \
        BASE_PORT="$port" \
        STREAM_LOGS=0 \
        NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" \
        NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}" \
        bash "$BASE_LAUNCHER"
    ) >"$launcher_log" 2>&1 &
    PIDS+=("$!")
    NAMES+=("$job_name")
}

launch_job vision_control \
    "${gpu_list[0]},${gpu_list[1]}" vision_only learned "$((BASE_PORT + 0))"
launch_job soft_prompt_online \
    "${gpu_list[2]},${gpu_list[3]}" text_safe learned "$((BASE_PORT + 1))"
launch_job no_soft_online \
    "${gpu_list[4]},${gpu_list[5]}" text_safe disabled "$((BASE_PORT + 2))"

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

echo "[DONE] All selected 6-GPU DDP-pair jobs completed: $LOG_ROOT"
