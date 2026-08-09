#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk}"
PYTHON_EXE="${PYTHON_EXE:-python}"
GPU_PAIR="${GPU_PAIR:-0,1}"
TRAIN_CSV="${TRAIN_CSV:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_train_set.xlsx}"
VALID_CSV="${VALID_CSV:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_valid_set.xlsx}"
MANIFEST="${MANIFEST:-${PROJECT_DIR}/data_manifests/vision_balanced_v1.csv}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/llama/vision_balanced_v1/vision_only_balanced_v1}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/train_logs/vision_balanced_v1_2gpu_$(date +%Y%m%d_%H%M%S)}"

PATCH_SIZE="${PATCH_SIZE:-32 224 224}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
N_ITER_PER_EPOCH="${N_ITER_PER_EPOCH:-128}"
MAX_OPTIMIZER_STEPS="${MAX_OPTIMIZER_STEPS:-5000}"
WARMUP_OPTIMIZER_STEPS="${WARMUP_OPTIMIZER_STEPS:-250}"
EPOCHS="${EPOCHS:-313}"
LR="${LR:-1e-5}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SW_VALID_INTERVAL_STEPS="${SW_VALID_INTERVAL_STEPS:-500}"
SW_VALID_POSITIVE_CASES="${SW_VALID_POSITIVE_CASES:-128}"
SW_VALID_NORMAL_CASES="${SW_VALID_NORMAL_CASES:-128}"
SW_VALID_BATCH_SIZE="${SW_VALID_BATCH_SIZE:-4}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-32}"
MANIFEST_WORKERS="${MANIFEST_WORKERS:-12}"
SMOKE_TEST="${SMOKE_TEST:-0}"
MANIFEST_ONLY="${MANIFEST_ONLY:-0}"
AUTO_RESUME="${AUTO_RESUME:-1}"

export LLMSEG_IMAGE_PATH_REWRITE_FROM="${LLMSEG_IMAGE_PATH_REWRITE_FROM:-/mnt/nas100/Brain_ER/data/BrainCT_NIfTIv2}"
export LLMSEG_IMAGE_PATH_REWRITE_TO="${LLMSEG_IMAGE_PATH_REWRITE_TO:-/mnt/nas100/Brain_ER/IDs/kevin/BrainCT_NIfTIv2}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"

if [ -f "${PROJECT_DIR}/wandb_key.txt" ]; then
    export WANDB_API_KEY="$(tr -d '\r\n' < "${PROJECT_DIR}/wandb_key.txt")"
fi

cd "$PROJECT_DIR"
mkdir -p "$(dirname "$MANIFEST")" "$CHECKPOINT_DIR" "$LOG_ROOT"

if [ ! -f "$MANIFEST" ]; then
    echo "[MANIFEST] Building supervision manifest: $MANIFEST"
    "$PYTHON_EXE" build_vision_sampling_manifest.py \
        --train-csv "$TRAIN_CSV" \
        --valid-csv "$VALID_CSV" \
        --output "$MANIFEST" \
        --workers "$MANIFEST_WORKERS" \
        2>&1 | tee "${LOG_ROOT}/manifest.log"
else
    echo "[MANIFEST] Reusing: $MANIFEST"
fi

if [ "$MANIFEST_ONLY" = "1" ]; then
    echo "[DONE] Manifest-only run completed."
    exit 0
fi

if [ -f "${PROJECT_DIR}/verify_runtime_environment.py" ]; then
    "$PYTHON_EXE" "${PROJECT_DIR}/verify_runtime_environment.py"
fi

if [ "$SMOKE_TEST" = "1" ]; then
    N_ITER_PER_EPOCH="${SMOKE_N_ITER_PER_EPOCH:-16}"
    MAX_OPTIMIZER_STEPS="${SMOKE_MAX_OPTIMIZER_STEPS:-2}"
    WARMUP_OPTIMIZER_STEPS="${SMOKE_WARMUP_OPTIMIZER_STEPS:-1}"
    EPOCHS="${SMOKE_EPOCHS:-1}"
    SW_VALID_INTERVAL_STEPS="${SMOKE_SW_VALID_INTERVAL_STEPS:-2}"
    SW_VALID_POSITIVE_CASES="${SMOKE_SW_VALID_POSITIVE_CASES:-2}"
    SW_VALID_NORMAL_CASES="${SMOKE_SW_VALID_NORMAL_CASES:-2}"
    CHECKPOINT_INTERVAL=1
fi

resume_args=()
if [ "$AUTO_RESUME" = "1" ]; then
    latest_checkpoint="$(find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name 'model_epoch_*.pth' -printf '%f\n' 2>/dev/null | sort -V | tail -n 1 || true)"
    if [ -n "$latest_checkpoint" ]; then
        resume_args=(--resume "${CHECKPOINT_DIR}/${latest_checkpoint}")
        echo "[RESUME] ${resume_args[1]}"
    fi
fi

IFS=',' read -r -a gpu_array <<< "$GPU_PAIR"
if [ "${#gpu_array[@]}" -ne 2 ]; then
    echo "[ERROR] GPU_PAIR must contain exactly two GPU IDs: $GPU_PAIR"
    exit 2
fi

run_log="${LOG_ROOT}/vision_balanced_v1.log"
echo "[LAUNCH] Vision-Balanced-v1 on GPUs $GPU_PAIR"
echo "[LOG] $run_log"

CUDA_VISIBLE_DEVICES="$GPU_PAIR" \
accelerate launch \
    --num_processes 2 \
    --main_process_port "${BASE_PORT:-30600}" \
    --mixed_precision bf16 \
    --dynamo_backend no \
    train.py \
    --no-context \
    --no-use_dicom \
    --soft_prompt_mode disabled \
    --train_csv "$TRAIN_CSV" \
    --valid_csv "$VALID_CSV" \
    --vision_manifest "$MANIFEST" \
    --balanced_sampling \
    --filter_invalid_supervision \
    --labeled_validation_only \
    --sw_validation \
    --sw_valid_interval_steps "$SW_VALID_INTERVAL_STEPS" \
    --sw_valid_positive_cases "$SW_VALID_POSITIVE_CASES" \
    --sw_valid_normal_cases "$SW_VALID_NORMAL_CASES" \
    --sw_valid_step_size 0.5 \
    --sw_valid_batch_size "$SW_VALID_BATCH_SIZE" \
    --patch_size $PATCH_SIZE \
    --batch_size "$BATCH_SIZE" \
    --grad_accum "$GRAD_ACCUM" \
    --n_iter_per_epoch "$N_ITER_PER_EPOCH" \
    --n_iter_valid 1 \
    --val_interval 0 \
    --epochs "$EPOCHS" \
    --max_optimizer_steps "$MAX_OPTIMIZER_STEPS" \
    --warmup_optimizer_steps "$WARMUP_OPTIMIZER_STEPS" \
    --lr "$LR" \
    --num_workers "$NUM_WORKERS" \
    --loss_fct tversky \
    --deep_supervision \
    --deep_supervision_weights "1.0,0.3" \
    --use_ema \
    --ema_decay 0.999 \
    --checkpoint_interval "$CHECKPOINT_INTERVAL" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --experiment_name vision_balanced_v1 \
    --mixed_precision bf16 \
    "${resume_args[@]}" \
    2>&1 | tee "$run_log"

echo "[DONE] Vision-Balanced-v1: $CHECKPOINT_DIR"
