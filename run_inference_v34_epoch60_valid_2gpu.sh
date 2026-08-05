#!/bin/bash
set -euo pipefail

# Full validation comparison for the three completed v3.4 epoch-60 models.
# This is the model-selection run; the held-out test set is not used here.

PROJECT_DIR="${PROJECT_DIR:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk}"
PYTHON_EXE="${PYTHON_EXE:-python}"
GPU_PAIR="${GPU_PAIR:-0,1}"
VALID_CSV="${VALID_CSV:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_valid_set.xlsx}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/llama/safe_film_context_v34_partial_ds}"
LLM_REPO="${LLM_REPO:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_seg_jhk/model_custom/llama2/Llama-2-7b-chat-hf}"
SOURCE_TEXT_CACHE="${TEXT_FEATURE_CACHE:-${PROJECT_DIR}/text_feature_cache/llama2_safe_cc_nosoft.sqlite3}"
LOCAL_CACHE_DIR="${LOCAL_CACHE_DIR:-/tmp/llmseg_text_cache}"
LOCAL_TEXT_CACHE="${LOCAL_TEXT_CACHE:-${LOCAL_CACHE_DIR}/$(basename "$SOURCE_TEXT_CACHE")}"
RESULT_ROOT="${RESULT_ROOT:-/mnt/nas125/forGPU2/lhyunki/llmseg/inference_result/FUdata/v34_epoch60_full_valid_$(date +%Y%m%d_%H%M%S)}"
SW_BATCH_SIZE="${SW_BATCH_SIZE:-4}"
SAVE_PRED="${SAVE_PRED:-0}"
PROB_THRESHOLD="${PROB_THRESHOLD:-0.5}"
MIN_COMPONENT_VOXELS="${MIN_COMPONENT_VOXELS:-0}"
BASE_PORT="${BASE_PORT:-29980}"

IFS=',' read -r -a gpu_list <<< "$GPU_PAIR"
if [ "${#gpu_list[@]}" -ne 2 ] \
   || [ "${gpu_list[0]}" = "${gpu_list[1]}" ] \
   || ! [[ "${gpu_list[0]}" =~ ^[0-9]+$ ]] \
   || ! [[ "${gpu_list[1]}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] GPU_PAIR must contain two distinct IDs, e.g. GPU_PAIR=0,1"
    exit 2
fi

NAMES=(v34_cached_control_epoch60 v34_cached_ds2_epoch60 v34_cached_ds3_epoch60)
CHECKPOINTS=(
    "${CHECKPOINT_BASE}/text_safe_v34_cached_control_ddp2/model_epoch_60.pth"
    "${CHECKPOINT_BASE}/text_safe_v34_cached_ds2_ddp2/model_epoch_60.pth"
    "${CHECKPOINT_BASE}/text_safe_v34_cached_ds3_ddp2/model_epoch_60.pth"
)
for required in "$VALID_CSV" "$SOURCE_TEXT_CACHE" "${CHECKPOINTS[@]}"; do
    if [ ! -f "$required" ]; then
        echo "[ERROR] Required file not found: $required"
        exit 2
    fi
done

CUDA_VISIBLE_DEVICES="$GPU_PAIR" EXPECTED_GPUS=2 \
    "$PYTHON_EXE" "${PROJECT_DIR}/verify_runtime_environment.py"

mkdir -p "$LOCAL_CACHE_DIR" "$RESULT_ROOT"
"$PYTHON_EXE" "${PROJECT_DIR}/verify_text_feature_cache.py" \
    --csv "$VALID_CSV" --llm-repo "$LLM_REPO" \
    --cache "$SOURCE_TEXT_CACHE" --coverage-only
echo "[CACHE] Copying validation cache to local storage: $LOCAL_TEXT_CACHE"
cp -f "$SOURCE_TEXT_CACHE" "$LOCAL_TEXT_CACHE"

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export LLMSEG_FORCE_MATH_SDP="${LLMSEG_FORCE_MATH_SDP:-1}"
export LLMSEG_FORCE_FP32_MHA="${LLMSEG_FORCE_FP32_MHA:-1}"

save_args=(--no-save_pred)
if [ "$SAVE_PRED" = "1" ]; then
    save_args=(--save_pred)
fi

for index in "${!NAMES[@]}"; do
    name="${NAMES[$index]}"
    checkpoint="${CHECKPOINTS[$index]}"
    model_result="${RESULT_ROOT}/${name}"
    log_path="${RESULT_ROOT}/${name}.log"
    mkdir -p "$model_result"

    echo "=========================================================="
    echo "[FULL VALIDATION] $name on GPUs $GPU_PAIR"
    echo "Checkpoint: $checkpoint"
    echo "Save predictions: $SAVE_PRED"
    echo "Log: $log_path"
    echo "=========================================================="
    CUDA_VISIBLE_DEVICES="$GPU_PAIR" accelerate launch \
        --num_processes 2 --num_machines 1 --mixed_precision bf16 \
        --dynamo_backend no --main_process_port "$((BASE_PORT + index))" \
        "${PROJECT_DIR}/inference_eval.py" \
        --model_path "$checkpoint" --csv_path "$VALID_CSV" \
        --save_root "$model_result" --patch_size 32 224 224 \
        --mixed_precision bf16 --sw_batch_size "$SW_BATCH_SIZE" \
        --prob_threshold "$PROB_THRESHOLD" \
        --min_component_voxels "$MIN_COMPONENT_VOXELS" \
        "${save_args[@]}" \
        --context --soft_prompt_mode disabled --llm_repo "$LLM_REPO" \
        --text_feature_cache "$LOCAL_TEXT_CACHE" \
        --include_cc --include_chief_complaint \
        --overwrite_annotation_csv 2>&1 | tee "$log_path"
done

"$PYTHON_EXE" "${PROJECT_DIR}/compare_inference_summaries.py" --root "$RESULT_ROOT"
echo "[DONE] Epoch-60 full-validation comparison: $RESULT_ROOT"
