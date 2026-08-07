#!/bin/bash
set -euo pipefail

# Paired V7 five-condition experiment using all six GPUs:
#   GPU 0,1,2 -> frozen Vision-only probability extraction
#   GPU 3,4,5 -> frozen true DICOM-FiLM-only probability extraction
# The three V7 CC controls are subsequently applied to the exact same frozen
# DICOM-FiLM probability volumes on CPU.

PROJECT_DIR="${PROJECT_DIR:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk}"
PYTHON_EXE="${PYTHON_EXE:-python}"
GPU_IDS="${GPU_IDS:-}"
TRAIN_CSV="${TRAIN_CSV:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_train_set.xlsx}"
VALID_CSV="${VALID_CSV:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_valid_set.xlsx}"
VISION_CHECKPOINT="${VISION_CHECKPOINT:-/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/llama/safe_film_context_v3a_4gpu_parallel/vision_only_vision_control_1gpu/model_epoch_300.pth}"
DICOM_FILM_CHECKPOINT="${DICOM_FILM_CHECKPOINT:-}"
LLM_REPO="${LLM_REPO:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_seg_jhk/model_custom/llama2/Llama-2-7b-chat-hf}"
SOURCE_TEXT_CACHE="${TEXT_FEATURE_CACHE:-${PROJECT_DIR}/text_feature_cache/llama2_safe_cc_nosoft_deterministic.sqlite3}"
LOCAL_CACHE_DIR="${LOCAL_CACHE_DIR:-/tmp/llmseg_text_cache}"
LOCAL_TEXT_CACHE="${LOCAL_TEXT_CACHE:-${LOCAL_CACHE_DIR}/$(basename "$SOURCE_TEXT_CACHE")}"
RESULT_ROOT="${RESULT_ROOT:-/mnt/nas125/forGPU2/lhyunki/llmseg/inference_result/FUdata/posthoc_suppression_v7_matrix_6gpu_$(date +%Y%m%d_%H%M%S)}"
VISION_USE_EMA="${VISION_USE_EMA:-0}"
DICOM_USE_EMA="${DICOM_USE_EMA:-0}"
EMA_DECAY="${EMA_DECAY:-0.999}"
SW_BATCH_SIZE="${SW_BATCH_SIZE:-4}"
BASE_PORT="${BASE_PORT:-30800}"
P_MIN="${P_MIN:-0.2}"
P_PROTECT="${P_PROTECT:-0.85}"
BETAS="${BETAS:-0.25,0.5,1,2,3}"
TEXT_THRESHOLDS="${TEXT_THRESHOLDS:-0}"
MAX_SENSITIVITY_DROP="${MAX_SENSITIVITY_DROP:-0.01}"
SKIP_CLASSIFIER="${SKIP_CLASSIFIER:-0}"
SKIP_VISION_INFERENCE="${SKIP_VISION_INFERENCE:-0}"
SKIP_DICOM_INFERENCE="${SKIP_DICOM_INFERENCE:-0}"

if [ -z "$GPU_IDS" ]; then
    echo "[ERROR] Explicitly set six GPU IDs, e.g. GPU_IDS='0 1 2 3 4 5'"
    exit 2
fi
gpu_tokens="${GPU_IDS//,/ }"
read -r -a gpu_list <<< "$gpu_tokens"
if [ "${#gpu_list[@]}" -ne 6 ]; then
    echo "[ERROR] GPU_IDS must contain exactly six GPU IDs."
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
if [ -z "$DICOM_FILM_CHECKPOINT" ]; then
    echo "[ERROR] Set DICOM_FILM_CHECKPOINT to a true context=False, use_dicom=True checkpoint."
    echo "[ERROR] Do not use the Wave1 dicom_film job; that job trained context+FiLM."
    exit 2
fi
for required in \
    "$TRAIN_CSV" "$VALID_CSV" "$VISION_CHECKPOINT" \
    "$DICOM_FILM_CHECKPOINT" "$SOURCE_TEXT_CACHE"; do
    if [ ! -f "$required" ]; then
        echo "[ERROR] Required file not found: $required"
        exit 2
    fi
done

all_devices="$(IFS=,; echo "${gpu_list[*]}")"
vision_devices="${gpu_list[0]},${gpu_list[1]},${gpu_list[2]}"
dicom_devices="${gpu_list[3]},${gpu_list[4]},${gpu_list[5]}"
CUDA_VISIBLE_DEVICES="$all_devices" EXPECTED_GPUS=6 \
    "$PYTHON_EXE" "${PROJECT_DIR}/verify_runtime_environment.py"
"$PYTHON_EXE" "${PROJECT_DIR}/verify_checkpoint_contract.py" \
    --checkpoint "$VISION_CHECKPOINT" \
    --expect-context false --expect-dicom false --name vision_only
"$PYTHON_EXE" "${PROJECT_DIR}/verify_checkpoint_contract.py" \
    --checkpoint "$DICOM_FILM_CHECKPOINT" \
    --expect-context false --expect-dicom true --name dicom_film_only \
    --reference-checkpoint "$VISION_CHECKPOINT"

mkdir -p "$RESULT_ROOT" "$LOCAL_CACHE_DIR"
CLASSIFIER_DIR="${RESULT_ROOT}/normal_classifier"
VISION_RESULT="${RESULT_ROOT}/vision_only_probability"
DICOM_RESULT="${RESULT_ROOT}/dicom_film_probability"
MATRIX_RESULT="${RESULT_ROOT}/five_condition_matrix"
mkdir -p "$CLASSIFIER_DIR" "$VISION_RESULT" "$DICOM_RESULT" "$MATRIX_RESULT"

echo "[CACHE] Verifying deterministic CC-only cache coverage"
"$PYTHON_EXE" "${PROJECT_DIR}/verify_text_feature_cache.py" \
    --csv "$TRAIN_CSV" --csv "$VALID_CSV" \
    --llm-repo "$LLM_REPO" \
    --cache "$SOURCE_TEXT_CACHE" --coverage-only
echo "[CACHE] Copying cache to local storage: $LOCAL_TEXT_CACHE"
cp -f "$SOURCE_TEXT_CACHE" "$LOCAL_TEXT_CACHE"

if [ "$SKIP_CLASSIFIER" != "1" ]; then
    "$PYTHON_EXE" "${PROJECT_DIR}/fit_posthoc_normal_suppressor.py" \
        --train-csv "$TRAIN_CSV" \
        --valid-csv "$VALID_CSV" \
        --text-feature-cache "$LOCAL_TEXT_CACHE" \
        --output-dir "$CLASSIFIER_DIR" \
        2>&1 | tee "${RESULT_ROOT}/fit_normal_classifier.log"
fi
SCORES_CSV="${CLASSIFIER_DIR}/valid_normal_scores.csv"
if [ ! -f "$SCORES_CSV" ]; then
    echo "[ERROR] Missing classifier scores: $SCORES_CSV"
    exit 2
fi

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export LLMSEG_FORCE_MATH_SDP="${LLMSEG_FORCE_MATH_SDP:-1}"

PIDS=()
NAMES=()
launch_inference() {
    local name="$1"
    local devices="$2"
    local checkpoint="$3"
    local result_dir="$4"
    local use_dicom="$5"
    local use_ema="$6"
    local port="$7"
    local skip="$8"
    if [ "$skip" = "1" ]; then
        echo "[SKIP] $name inference; reusing $result_dir"
        return
    fi
    local dicom_args=(--no-use_dicom)
    if [ "$use_dicom" = "1" ]; then
        dicom_args=(--use_dicom)
    fi
    local ema_args=(--no-use_ema)
    if [ "$use_ema" = "1" ]; then
        ema_args=(--use_ema --ema_decay "$EMA_DECAY")
    fi
    local log_path="${RESULT_ROOT}/${name}_inference.log"
    echo "[LAUNCH] $name on GPUs $devices -> $log_path"
    (
        set -o pipefail
        CUDA_VISIBLE_DEVICES="$devices" accelerate launch \
            --num_processes 3 --num_machines 1 --mixed_precision bf16 \
            --dynamo_backend no --main_process_port "$port" \
            "${PROJECT_DIR}/inference_eval.py" \
            --model_path "$checkpoint" \
            --csv_path "$VALID_CSV" \
            --save_root "$result_dir" \
            --patch_size 32 224 224 \
            --mixed_precision bf16 \
            --sw_batch_size "$SW_BATCH_SIZE" \
            --no-context \
            "${dicom_args[@]}" \
            "${ema_args[@]}" \
            --no-save_pred \
            --save_probabilities \
            --save_labeled_probabilities_only \
            --probability_dtype float16 \
            --prob_threshold 0.5 \
            --fail_on_missing_images \
            --overwrite_annotation_csv \
            2>&1 | tee "$log_path"
    ) &
    PIDS+=("$!")
    NAMES+=("$name")
}

launch_inference vision_only "$vision_devices" "$VISION_CHECKPOINT" \
    "$VISION_RESULT" 0 "$VISION_USE_EMA" "$BASE_PORT" "$SKIP_VISION_INFERENCE"
launch_inference dicom_film_only "$dicom_devices" "$DICOM_FILM_CHECKPOINT" \
    "$DICOM_RESULT" 1 "$DICOM_USE_EMA" "$((BASE_PORT + 1))" "$SKIP_DICOM_INFERENCE"

failed=0
for index in "${!PIDS[@]}"; do
    if wait "${PIDS[$index]}"; then
        echo "[DONE] ${NAMES[$index]}"
    else
        echo "[ERROR] ${NAMES[$index]} inference failed."
        failed=1
    fi
done
if [ "$failed" -ne 0 ]; then
    exit 1
fi

VISION_ANNOTATION="${VISION_RESULT}/annotation.csv"
DICOM_ANNOTATION="${DICOM_RESULT}/annotation.csv"
for required in "$VISION_ANNOTATION" "$DICOM_ANNOTATION"; do
    if [ ! -f "$required" ]; then
        echo "[ERROR] Missing inference annotation: $required"
        exit 2
    fi
done

echo "[MATRIX] Vision | DICOM FiLM | DICOM+Empty | DICOM+Shuffled | DICOM+Real"
"$PYTHON_EXE" "${PROJECT_DIR}/evaluate_posthoc_suppression_v7_matrix.py" \
    --vision-annotation-csv "$VISION_ANNOTATION" \
    --dicom-annotation-csv "$DICOM_ANNOTATION" \
    --normal-scores-csv "$SCORES_CSV" \
    --output-dir "$MATRIX_RESULT" \
    --p-min "$P_MIN" \
    --p-protect "$P_PROTECT" \
    --betas "$BETAS" \
    --text-thresholds "$TEXT_THRESHOLDS" \
    --max-sensitivity-drop "$MAX_SENSITIVITY_DROP" \
    2>&1 | tee "${RESULT_ROOT}/five_condition_matrix.log"

echo "[DONE] V7 paired five-condition experiment: $RESULT_ROOT"
