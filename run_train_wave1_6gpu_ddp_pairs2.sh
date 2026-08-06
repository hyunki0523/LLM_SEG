#!/bin/bash
set -euo pipefail

# Wave 2: repeat the three DICOM transport branches with learned soft prompts.
#   B / GPU 0,1: CC only + soft prompt
#   D / GPU 2,3: CC + full DICOM text + soft prompt
#   F / GPU 4,5: CC + DICOM FiLM + soft prompt
# Wave 1 and Wave 2 share every training setting except soft_prompt_mode.

PROJECT_DIR="${PROJECT_DIR:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk}"
BASE_LAUNCHER="${BASE_LAUNCHER:-${PROJECT_DIR}/run_train_dicom_ablation_8gpu.sh}"
PYTHON_EXE="${PYTHON_EXE:-python}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5}"
WAVE2_JOBS="${WAVE2_JOBS:-cc_only_soft,dicom_text_full_soft,dicom_film_soft}"
TRAIN_CSV="${TRAIN_CSV:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_train_set.xlsx}"
VALID_CSV="${VALID_CSV:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_valid_set.xlsx}"
LLM_REPO="${LLM_REPO:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_seg_jhk/model_custom/llama2/Llama-2-7b-chat-hf/}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/llama/hybrid_wave2_soft_ema_ds2}"
SMOKE_TEST="${SMOKE_TEST:-0}"
BASE_PORT="${BASE_PORT:-30200}"

DEFAULT_IMAGE_PATH_REWRITE_FROM="/mnt/nas100/Brain_ER/data/BrainCT_NIfTIv2"
DEFAULT_IMAGE_PATH_REWRITE_TO="/mnt/nas100/Brain_ER/IDs/kevin/BrainCT_NIfTIv2"
IMAGE_PATH_REWRITE_FROM="${IMAGE_PATH_REWRITE_FROM:-$DEFAULT_IMAGE_PATH_REWRITE_FROM}"
IMAGE_PATH_REWRITE_TO="${IMAGE_PATH_REWRITE_TO:-$DEFAULT_IMAGE_PATH_REWRITE_TO}"

timestamp="$(date +%Y%m%d_%H%M%S)"
smoke_tag=""
if [ "$SMOKE_TEST" = "1" ]; then
    smoke_tag="_smoke"
fi
LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/train_logs/hybrid_wave2_soft_ema_ds2_6gpu${smoke_tag}_${timestamp}}"

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
if [ ! -f "${LLM_REPO%/}/config.json" ]; then
    echo "[ERROR] Llama config not found: ${LLM_REPO%/}/config.json"
    exit 2
fi
if ! compgen -G "${LLM_REPO%/}/model-*.safetensors" >/dev/null \
   && ! compgen -G "${LLM_REPO%/}/pytorch_model-*.bin" >/dev/null; then
    echo "[ERROR] Llama weight shards not found: $LLM_REPO"
    exit 2
fi

selected() {
    local name="$1"
    [[ ",$WAVE2_JOBS," == *",$name,"* ]]
}

selected_count=0
for known_job in cc_only_soft dicom_text_full_soft dicom_film_soft; do
    if selected "$known_job"; then
        selected_count=$((selected_count + 1))
    fi
done
if [ "$selected_count" -eq 0 ]; then
    echo "[ERROR] WAVE2_JOBS selected no known jobs: $WAVE2_JOBS"
    exit 2
fi

mkdir -p "$LOG_ROOT" "$CHECKPOINT_BASE"
cd "$PROJECT_DIR"

selected_cuda_devices="$(IFS=,; echo "${gpu_list[*]}")"
CUDA_VISIBLE_DEVICES="$selected_cuda_devices" EXPECTED_GPUS=6 \
    "$PYTHON_EXE" "${PROJECT_DIR}/verify_runtime_environment.py"

export LLMSEG_SKIP_MISSING_IMAGE_PATHS="${SKIP_MISSING_IMAGE_PATHS:-1}"
export LLMSEG_IMAGE_PATH_REWRITE_FROM="$IMAGE_PATH_REWRITE_FROM"
export LLMSEG_IMAGE_PATH_REWRITE_TO="$IMAGE_PATH_REWRITE_TO"

echo "[CHECK] Auditing the Wave 2 safe prompt contracts."
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

# Match Wave 1 exactly: 16 optimizer updates and effective global batch 32.
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
export SOFT_PROMPT_MODE=learned
# Keep position IDs equivalent to Wave 1's cached no-soft LLM features. Without
# this switch, left-padding length would change token positions by batch.
export LLMSEG_SOFT_PROMPT_MASK_POSITION_IDS=1
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
    CHECKPOINT_BASE="${CHECKPOINT_BASE_SMOKE:-${PROJECT_DIR}/_debug_ckpt/hybrid_wave2_soft_ema_ds2}"
    mkdir -p "$CHECKPOINT_BASE"
    export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
fi

echo "[PLAN] Wave 2 jobs=$WAVE2_JOBS epochs=$EPOCHS soft=$SOFT_PROMPT_MODE EMA=$USE_EMA DS=$DEEP_SUPERVISION/$DEEP_SUPERVISION_WEIGHTS"
echo "[PLAN] Checkpoints=$CHECKPOINT_BASE"
echo "[PLAN] Logs=$LOG_ROOT"

# Snapshot prevents active jobs from observing later edits to the base script.
BASE_LAUNCHER_SNAPSHOT="${LOG_ROOT}/run_train_dicom_ablation_snapshot.sh"
cp "$BASE_LAUNCHER" "$BASE_LAUNCHER_SNAPSHOT"

PIDS=()
NAMES=()

launch_job() {
    local job_name="$1"
    local gpu_pair="$2"
    local base_experiment="$3"
    local dicom_prompt_mode="$4"
    local port="$5"
    local suffix="_wave2_${job_name}_ema_ds2"
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
        SOFT_PROMPT_MODE=learned TEXT_FEATURE_CACHE="" \
        DICOM_PROMPT_MODE="$dicom_prompt_mode" \
        EXPERIMENT_NAME_SUFFIX="$suffix" \
        CHECKPOINT_BASE="$CHECKPOINT_BASE" LOG_ROOT="$job_log_root" \
        BASE_PORT="$port" STREAM_LOGS=0 \
        CHECK_IMAGE_PATHS=0 CHECK_DATASET_CONTRACT=0 \
        NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" \
        NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}" \
        LLMSEG_SOFT_PROMPT_MASK_POSITION_IDS=1 \
        bash "$BASE_LAUNCHER_SNAPSHOT"
    ) >"$launcher_log" 2>&1 &
    PIDS+=("$!")
    NAMES+=("$job_name")
}

launch_job cc_only_soft \
    "${gpu_list[0]},${gpu_list[1]}" text_safe none "$((BASE_PORT + 0))"
launch_job dicom_text_full_soft \
    "${gpu_list[2]},${gpu_list[3]}" text_safe full "$((BASE_PORT + 1))"
launch_job dicom_film_soft \
    "${gpu_list[4]},${gpu_list[5]}" dicom_text_safe none "$((BASE_PORT + 2))"

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
echo "[DONE] Wave 2 completed: $LOG_ROOT"
