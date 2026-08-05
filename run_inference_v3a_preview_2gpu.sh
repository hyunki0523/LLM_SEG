#!/bin/bash
set -euo pipefail

# Fair interim comparison: all primary v3a models use the same epoch and cases.
# The four models run sequentially, while each inference job uses both GPUs.

PROJECT_DIR="${PROJECT_DIR:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk}"
PYTHON_EXE="${PYTHON_EXE:-python}"
GPU_PAIR="${GPU_PAIR:-0,1}"
EVAL_EPOCH="${EVAL_EPOCH:-auto}"
POSITIVE_CASES="${POSITIVE_CASES:-16}"
NORMAL_CASES="${NORMAL_CASES:-16}"
SUBSET_SEED="${SUBSET_SEED:-42}"
SW_BATCH_SIZE="${SW_BATCH_SIZE:-4}"
BASE_PORT="${BASE_PORT:-29920}"
OVERWRITE_PRED="${OVERWRITE_PRED:-0}"
INCLUDE_CACHED_FINAL="${INCLUDE_CACHED_FINAL:-0}"

VALID_CSV="${VALID_CSV:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_valid_set.xlsx}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/llama/safe_film_context_v3a_4gpu_parallel}"
LLM_REPO="${LLM_REPO:-${PROJECT_DIR}/model_custom/llama2/Llama-2-7b-chat-hf}"
SOURCE_TEXT_CACHE="${TEXT_FEATURE_CACHE:-${PROJECT_DIR}/text_feature_cache/llama2_safe_cc_nosoft.sqlite3}"
LOCAL_CACHE_DIR="${LOCAL_CACHE_DIR:-/tmp/llmseg_text_cache}"
LOCAL_TEXT_CACHE="${LOCAL_TEXT_CACHE:-${LOCAL_CACHE_DIR}/$(basename "$SOURCE_TEXT_CACHE")}"

if [ "$EVAL_EPOCH" = "auto" ]; then
    EVAL_EPOCH=""
    while IFS= read -r candidate_epoch; do
        if [ -f "${CHECKPOINT_BASE}/text_safe_soft_prompt_online_1gpu/model_epoch_${candidate_epoch}.pth" ] \
           && [ -f "${CHECKPOINT_BASE}/text_safe_no_soft_online_1gpu/model_epoch_${candidate_epoch}.pth" ] \
           && [ -f "${CHECKPOINT_BASE}/text_safe_no_soft_cached_1gpu/model_epoch_${candidate_epoch}.pth" ]; then
            EVAL_EPOCH="$candidate_epoch"
            break
        fi
    done < <(
        find "${CHECKPOINT_BASE}/vision_only_vision_control_1gpu" \
            -maxdepth 1 -type f -name 'model_epoch_*.pth' -printf '%f\n' \
            | sed -E 's/^model_epoch_([0-9]+)\.pth$/\1/' \
            | sort -rn
    )
    if [ -z "$EVAL_EPOCH" ]; then
        echo "[ERROR] No common checkpoint epoch exists across the four v3a models."
        exit 2
    fi
    echo "[CHECKPOINT] Latest common epoch: $EVAL_EPOCH"
elif ! [[ "$EVAL_EPOCH" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] EVAL_EPOCH must be a positive integer or 'auto'."
    exit 2
fi

SUBSET_CSV="${SUBSET_CSV:-${PROJECT_DIR}/_eval_subsets/v3a_valid_p${POSITIVE_CASES}_n${NORMAL_CASES}_seed${SUBSET_SEED}.csv}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_DIR}/inference_result/v3a_preview_epoch${EVAL_EPOCH}_$(date +%Y%m%d_%H%M%S)}"

IFS=',' read -r -a gpu_list <<< "$GPU_PAIR"
if [ "${#gpu_list[@]}" -ne 2 ]; then
    echo "[ERROR] GPU_PAIR must contain exactly two comma-separated GPU IDs."
    exit 2
fi
for gpu in "${gpu_list[@]}"; do
    if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] Invalid GPU ID: $gpu"
        exit 2
    fi
done
if [ "${gpu_list[0]}" = "${gpu_list[1]}" ]; then
    echo "[ERROR] GPU_PAIR must contain two distinct devices."
    exit 2
fi

for required in "$VALID_CSV" "$SOURCE_TEXT_CACHE"; do
    if [ ! -f "$required" ]; then
        echo "[ERROR] Required file not found: $required"
        exit 2
    fi
done

mkdir -p "$(dirname "$SUBSET_CSV")" "$RESULT_ROOT" "$LOCAL_CACHE_DIR"
if [ ! -f "$SUBSET_CSV" ] || [ "${REBUILD_SUBSET:-0}" = "1" ]; then
    "$PYTHON_EXE" "${PROJECT_DIR}/make_inference_preview_subset.py" \
        --input "$VALID_CSV" \
        --output "$SUBSET_CSV" \
        --positive "$POSITIVE_CASES" \
        --normal "$NORMAL_CASES" \
        --seed "$SUBSET_SEED"
fi

if [ ! -f "$LOCAL_TEXT_CACHE" ] || ! cmp -s "$SOURCE_TEXT_CACHE" "$LOCAL_TEXT_CACHE"; then
    echo "[CACHE] Copying SQLite cache to local storage: $LOCAL_TEXT_CACHE"
    cp "$SOURCE_TEXT_CACHE" "$LOCAL_TEXT_CACHE"
fi
"$PYTHON_EXE" "${PROJECT_DIR}/verify_text_feature_cache.py" \
    --csv "$SUBSET_CSV" \
    --llm-repo "$LLM_REPO" \
    --cache "$LOCAL_TEXT_CACHE" \
    --coverage-only

VISION_CKPT="${CHECKPOINT_BASE}/vision_only_vision_control_1gpu/model_epoch_${EVAL_EPOCH}.pth"
SOFT_CKPT="${CHECKPOINT_BASE}/text_safe_soft_prompt_online_1gpu/model_epoch_${EVAL_EPOCH}.pth"
NO_SOFT_ONLINE_CKPT="${CHECKPOINT_BASE}/text_safe_no_soft_online_1gpu/model_epoch_${EVAL_EPOCH}.pth"
NO_SOFT_CACHED_CKPT="${CHECKPOINT_BASE}/text_safe_no_soft_cached_1gpu/model_epoch_${EVAL_EPOCH}.pth"
CACHED_FINAL_CKPT="${CHECKPOINT_BASE}/text_safe_no_soft_cached_1gpu/final_model.pth"

NAMES=(vision_epoch${EVAL_EPOCH} soft_prompt_epoch${EVAL_EPOCH} no_soft_online_epoch${EVAL_EPOCH} no_soft_cached_epoch${EVAL_EPOCH})
CHECKPOINTS=("$VISION_CKPT" "$SOFT_CKPT" "$NO_SOFT_ONLINE_CKPT" "$NO_SOFT_CACHED_CKPT")
MODES=(vision soft cached cached)
if [ "$INCLUDE_CACHED_FINAL" = "1" ]; then
    NAMES+=(no_soft_cached_final)
    CHECKPOINTS+=("$CACHED_FINAL_CKPT")
    MODES+=(cached)
fi

for checkpoint in "${CHECKPOINTS[@]}"; do
    if [ ! -f "$checkpoint" ]; then
        echo "[ERROR] Checkpoint not found: $checkpoint"
        exit 2
    fi
done

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export LLMSEG_FORCE_MATH_SDP="${LLMSEG_FORCE_MATH_SDP:-1}"

for index in "${!NAMES[@]}"; do
    name="${NAMES[$index]}"
    checkpoint="${CHECKPOINTS[$index]}"
    mode="${MODES[$index]}"
    model_result="${RESULT_ROOT}/${name}"
    log_path="${RESULT_ROOT}/${name}.log"
    mkdir -p "$model_result"

    mode_args=()
    case "$mode" in
        vision)
            mode_args+=(--no-context --soft_prompt_mode learned)
            ;;
        soft)
            mode_args+=(--context --soft_prompt_mode learned --llm_repo "$LLM_REPO")
            ;;
        cached)
            mode_args+=(--context --soft_prompt_mode disabled --llm_repo "$LLM_REPO" --text_feature_cache "$LOCAL_TEXT_CACHE")
            ;;
    esac
    overwrite_args=(--no-overwrite_pred)
    if [ "$OVERWRITE_PRED" = "1" ]; then
        overwrite_args=(--overwrite_pred)
    fi

    echo "=========================================================="
    echo "[INFERENCE] $name on GPUs $GPU_PAIR"
    echo "Checkpoint: $checkpoint"
    echo "Log: $log_path"
    echo "=========================================================="
    CUDA_VISIBLE_DEVICES="$GPU_PAIR" accelerate launch \
        --num_processes 2 \
        --num_machines 1 \
        --mixed_precision bf16 \
        --dynamo_backend no \
        --main_process_port "$((BASE_PORT + index))" \
        "${PROJECT_DIR}/inference_eval.py" \
        --model_path "$checkpoint" \
        --csv_path "$SUBSET_CSV" \
        --save_root "$model_result" \
        --patch_size 32 224 224 \
        --mixed_precision bf16 \
        --sw_batch_size "$SW_BATCH_SIZE" \
        --prob_threshold 0.5 \
        --min_component_voxels 20 \
        --save_pred \
        --include_cc \
        --include_chief_complaint \
        "${overwrite_args[@]}" \
        "${mode_args[@]}" 2>&1 | tee "$log_path"
done

"$PYTHON_EXE" "${PROJECT_DIR}/compare_inference_summaries.py" --root "$RESULT_ROOT"
echo "[DONE] Preview inference completed: $RESULT_ROOT"
