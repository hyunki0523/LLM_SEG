#!/bin/bash
set -euo pipefail

# V7 Frozen Post-Hoc Suppressor
#   1) Fit calibrated cached-LLM and TF-IDF normal classifiers.
#   2) Run the frozen vision checkpoint exactly once and retain FP16 probability maps.
#   3) Sweep negative-only band-pass suppression on CPU without rerunning STU-Net.

PROJECT_DIR="${PROJECT_DIR:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk}"
PYTHON_EXE="${PYTHON_EXE:-python}"
GPU_PAIR="${GPU_PAIR:-0,1}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
TRAIN_CSV="${TRAIN_CSV:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_train_set.xlsx}"
VALID_CSV="${VALID_CSV:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_valid_set.xlsx}"
VISION_CHECKPOINT="${VISION_CHECKPOINT:-/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/llama/safe_film_context_v3a_4gpu_parallel/vision_only_vision_control_1gpu/model_epoch_300.pth}"
LLM_REPO="${LLM_REPO:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_seg_jhk/model_custom/llama2/Llama-2-7b-chat-hf}"
SOURCE_TEXT_CACHE="${TEXT_FEATURE_CACHE:-${PROJECT_DIR}/text_feature_cache/llama2_safe_cc_nosoft_deterministic.sqlite3}"
LOCAL_CACHE_DIR="${LOCAL_CACHE_DIR:-/tmp/llmseg_text_cache}"
LOCAL_TEXT_CACHE="${LOCAL_TEXT_CACHE:-${LOCAL_CACHE_DIR}/$(basename "$SOURCE_TEXT_CACHE")}"
RESULT_ROOT="${RESULT_ROOT:-/mnt/nas125/forGPU2/lhyunki/llmseg/inference_result/FUdata/posthoc_suppression_v7_$(date +%Y%m%d_%H%M%S)}"
USE_EMA="${USE_EMA:-0}"
EMA_DECAY="${EMA_DECAY:-0.999}"
SW_BATCH_SIZE="${SW_BATCH_SIZE:-4}"
BASE_PORT="${BASE_PORT:-30700}"
P_MIN="${P_MIN:-0.2}"
P_PROTECT="${P_PROTECT:-0.85}"
BETAS="${BETAS:-0.25,0.5,1,2,3}"
TEXT_THRESHOLDS="${TEXT_THRESHOLDS:-0}"
MAX_SENSITIVITY_DROP="${MAX_SENSITIVITY_DROP:-0.01}"
SKIP_CLASSIFIER="${SKIP_CLASSIFIER:-0}"
SKIP_VISION_INFERENCE="${SKIP_VISION_INFERENCE:-0}"

gpu_tokens="${GPU_PAIR//,/ }"
read -r -a gpu_list <<< "$gpu_tokens"
if ! [[ "$NUM_PROCESSES" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] NUM_PROCESSES must be a positive integer."
    exit 2
fi
if [ "${#gpu_list[@]}" -ne "$NUM_PROCESSES" ]; then
    echo "[ERROR] GPU_PAIR must contain exactly $NUM_PROCESSES GPU IDs."
    exit 2
fi
for gpu in "${gpu_list[@]}"; do
    if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] Invalid GPU ID: $gpu"
        exit 2
    fi
done
if [ "$(printf '%s\n' "${gpu_list[@]}" | sort -u | wc -l)" -ne "$NUM_PROCESSES" ]; then
    echo "[ERROR] GPU_PAIR must contain $NUM_PROCESSES distinct GPU IDs."
    exit 2
fi
CUDA_DEVICE_LIST="$(IFS=,; echo "${gpu_list[*]}")"
for required in "$TRAIN_CSV" "$VALID_CSV" "$VISION_CHECKPOINT" "$SOURCE_TEXT_CACHE"; do
    if [ ! -f "$required" ]; then
        echo "[ERROR] Required file not found: $required"
        exit 2
    fi
done

mkdir -p "$RESULT_ROOT" "$LOCAL_CACHE_DIR"
CLASSIFIER_DIR="${RESULT_ROOT}/normal_classifier"
VISION_RESULT="${RESULT_ROOT}/vision_probability"
SWEEP_RESULT="${RESULT_ROOT}/suppression_sweep"
mkdir -p "$CLASSIFIER_DIR" "$VISION_RESULT" "$SWEEP_RESULT"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_LIST" EXPECTED_GPUS="$NUM_PROCESSES" \
    "$PYTHON_EXE" "${PROJECT_DIR}/verify_runtime_environment.py"

echo "[CACHE] Verifying CC-only cache coverage"
"$PYTHON_EXE" "${PROJECT_DIR}/verify_text_feature_cache.py" \
    --csv "$TRAIN_CSV" --csv "$VALID_CSV" \
    --llm-repo "$LLM_REPO" \
    --cache "$SOURCE_TEXT_CACHE" --coverage-only
echo "[CACHE] Copying cache to local storage: $LOCAL_TEXT_CACHE"
cp -f "$SOURCE_TEXT_CACHE" "$LOCAL_TEXT_CACHE"

if [ "$SKIP_CLASSIFIER" != "1" ]; then
    echo "[FIT] Cached-LLM normal suppressor and TF-IDF control"
    "$PYTHON_EXE" "${PROJECT_DIR}/fit_posthoc_normal_suppressor.py" \
        --train-csv "$TRAIN_CSV" \
        --valid-csv "$VALID_CSV" \
        --text-feature-cache "$LOCAL_TEXT_CACHE" \
        --output-dir "$CLASSIFIER_DIR" \
        2>&1 | tee "${RESULT_ROOT}/fit_normal_classifier.log"
fi

SCORES_CSV="${CLASSIFIER_DIR}/valid_normal_scores.csv"
if [ ! -f "$SCORES_CSV" ]; then
    echo "[ERROR] Missing classifier score file: $SCORES_CSV"
    exit 2
fi

if [ "$SKIP_VISION_INFERENCE" != "1" ]; then
    ema_args=(--no-use_ema)
    if [ "$USE_EMA" = "1" ]; then
        ema_args=(--use_ema --ema_decay "$EMA_DECAY")
    fi
    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
    export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
    export LLMSEG_FORCE_MATH_SDP="${LLMSEG_FORCE_MATH_SDP:-1}"
    echo "[INFERENCE] Frozen vision probability extraction on GPUs $CUDA_DEVICE_LIST"
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE_LIST" accelerate launch \
        --num_processes "$NUM_PROCESSES" --num_machines 1 --mixed_precision bf16 \
        --dynamo_backend no --main_process_port "$BASE_PORT" \
        "${PROJECT_DIR}/inference_eval.py" \
        --model_path "$VISION_CHECKPOINT" \
        --csv_path "$VALID_CSV" \
        --save_root "$VISION_RESULT" \
        --patch_size 32 224 224 \
        --mixed_precision bf16 \
        --sw_batch_size "$SW_BATCH_SIZE" \
        --no-context \
        --no-save_pred \
        --save_probabilities \
        --save_labeled_probabilities_only \
        --probability_dtype float16 \
        --prob_threshold 0.5 \
        --fail_on_missing_images \
        --overwrite_annotation_csv \
        "${ema_args[@]}" \
        2>&1 | tee "${RESULT_ROOT}/vision_probability_inference.log"
fi

ANNOTATION_CSV="${VISION_RESULT}/annotation.csv"
if [ ! -f "$ANNOTATION_CSV" ]; then
    echo "[ERROR] Missing vision annotation: $ANNOTATION_CSV"
    exit 2
fi

echo "[SWEEP] Frozen V7 band-pass suppression"
"$PYTHON_EXE" "${PROJECT_DIR}/evaluate_posthoc_suppression_v7.py" \
    --annotation-csv "$ANNOTATION_CSV" \
    --normal-scores-csv "$SCORES_CSV" \
    --output-dir "$SWEEP_RESULT" \
    --p-min "$P_MIN" \
    --p-protect "$P_PROTECT" \
    --betas "$BETAS" \
    --text-thresholds "$TEXT_THRESHOLDS" \
    --max-sensitivity-drop "$MAX_SENSITIVITY_DROP" \
    2>&1 | tee "${RESULT_ROOT}/suppression_sweep.log"

echo "[DONE] V7 post-hoc suppression: $RESULT_ROOT"
