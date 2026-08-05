#!/bin/bash
set -euo pipefail

# Compare the v3.4 deep-supervision branches at their latest common epoch.
# Each checkpoint runs sequentially; each inference uses both selected GPUs.

PROJECT_DIR="${PROJECT_DIR:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk}"
PYTHON_EXE="${PYTHON_EXE:-python}"
GPU_PAIR="${GPU_PAIR:-0,1}"
EVAL_EPOCH="${EVAL_EPOCH:-auto}"
POSITIVE_CASES="${POSITIVE_CASES:-16}"
NORMAL_CASES="${NORMAL_CASES:-16}"
SUBSET_SEED="${SUBSET_SEED:-42}"
SW_BATCH_SIZE="${SW_BATCH_SIZE:-4}"
BASE_PORT="${BASE_PORT:-29940}"
OVERWRITE_PRED="${OVERWRITE_PRED:-0}"

VALID_CSV="${VALID_CSV:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_valid_set.xlsx}"
V3A_BASE="${V3A_BASE:-/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/llama/safe_film_context_v3a_4gpu_parallel}"
V34_BASE="${V34_BASE:-/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/llama/safe_film_context_v34_partial_ds}"
LLM_REPO="${LLM_REPO:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_seg_jhk/model_custom/llama2/Llama-2-7b-chat-hf}"
SOURCE_TEXT_CACHE="${TEXT_FEATURE_CACHE:-${PROJECT_DIR}/text_feature_cache/llama2_safe_cc_nosoft.sqlite3}"
LOCAL_CACHE_DIR="${LOCAL_CACHE_DIR:-/tmp/llmseg_text_cache}"
LOCAL_TEXT_CACHE="${LOCAL_TEXT_CACHE:-${LOCAL_CACHE_DIR}/$(basename "$SOURCE_TEXT_CACHE")}"

CONTROL_DIR="${V34_BASE}/text_safe_v34_cached_control_ddp2"
DS2_DIR="${V34_BASE}/text_safe_v34_cached_ds2_ddp2"
DS3_DIR="${V34_BASE}/text_safe_v34_cached_ds3_ddp2"
if [ "$EVAL_EPOCH" = "auto" ]; then
    EVAL_EPOCH=""
    while IFS= read -r candidate_epoch; do
        if [ -f "${DS2_DIR}/model_epoch_${candidate_epoch}.pth" ] \
           && [ -f "${DS3_DIR}/model_epoch_${candidate_epoch}.pth" ]; then
            EVAL_EPOCH="$candidate_epoch"
            break
        fi
    done < <(
        find "$CONTROL_DIR" -maxdepth 1 -type f -name 'model_epoch_*.pth' -printf '%f\n' \
            | sed -E 's/^model_epoch_([0-9]+)\.pth$/\1/' \
            | sort -rn
    )
    if [ -z "$EVAL_EPOCH" ]; then
        echo "[ERROR] No common checkpoint epoch exists across the v3.4 branches."
        exit 2
    fi
    echo "[CHECKPOINT] Latest v3.4 common epoch: $EVAL_EPOCH"
elif ! [[ "$EVAL_EPOCH" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] EVAL_EPOCH must be a positive integer or 'auto'."
    exit 2
fi

SUBSET_CSV="${SUBSET_CSV:-${PROJECT_DIR}/_eval_subsets/v34_valid_p${POSITIVE_CASES}_n${NORMAL_CASES}_seed${SUBSET_SEED}.csv}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_DIR}/inference_result/v34_preview_epoch${EVAL_EPOCH}_$(date +%Y%m%d_%H%M%S)}"

IFS=',' read -r -a gpu_list <<< "$GPU_PAIR"
if [ "${#gpu_list[@]}" -ne 2 ] \
   || [ "${gpu_list[0]}" = "${gpu_list[1]}" ] \
   || ! [[ "${gpu_list[0]}" =~ ^[0-9]+$ ]] \
   || ! [[ "${gpu_list[1]}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] GPU_PAIR must contain two distinct IDs, e.g. GPU_PAIR=0,1"
    exit 2
fi

BASELINE_CKPT="${V3A_BASE}/text_safe_no_soft_cached_1gpu/final_model.pth"
NAMES=(v3a_pretrained_final v34_control_epoch${EVAL_EPOCH} v34_ds2_epoch${EVAL_EPOCH} v34_ds3_epoch${EVAL_EPOCH})
CHECKPOINTS=(
    "$BASELINE_CKPT"
    "${CONTROL_DIR}/model_epoch_${EVAL_EPOCH}.pth"
    "${DS2_DIR}/model_epoch_${EVAL_EPOCH}.pth"
    "${DS3_DIR}/model_epoch_${EVAL_EPOCH}.pth"
)
for required in "$VALID_CSV" "$SOURCE_TEXT_CACHE" "${CHECKPOINTS[@]}"; do
    if [ ! -f "$required" ]; then
        echo "[ERROR] Required file not found: $required"
        exit 2
    fi
done

mkdir -p "$(dirname "$SUBSET_CSV")" "$RESULT_ROOT" "$LOCAL_CACHE_DIR"
if [ ! -f "$SUBSET_CSV" ] || [ "${REBUILD_SUBSET:-0}" = "1" ]; then
    "$PYTHON_EXE" "${PROJECT_DIR}/make_inference_preview_subset.py" \
        --input "$VALID_CSV" --output "$SUBSET_CSV" \
        --positive "$POSITIVE_CASES" --normal "$NORMAL_CASES" --seed "$SUBSET_SEED"
fi

if [ ! -f "$LOCAL_TEXT_CACHE" ] || ! cmp -s "$SOURCE_TEXT_CACHE" "$LOCAL_TEXT_CACHE"; then
    echo "[CACHE] Copying SQLite cache to local storage: $LOCAL_TEXT_CACHE"
    cp "$SOURCE_TEXT_CACHE" "$LOCAL_TEXT_CACHE"
fi
"$PYTHON_EXE" "${PROJECT_DIR}/verify_text_feature_cache.py" \
    --csv "$SUBSET_CSV" --llm-repo "$LLM_REPO" \
    --cache "$LOCAL_TEXT_CACHE" --coverage-only

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export LLMSEG_FORCE_MATH_SDP="${LLMSEG_FORCE_MATH_SDP:-1}"
export LLMSEG_FORCE_FP32_MHA="${LLMSEG_FORCE_FP32_MHA:-1}"

for index in "${!NAMES[@]}"; do
    name="${NAMES[$index]}"
    checkpoint="${CHECKPOINTS[$index]}"
    model_result="${RESULT_ROOT}/${name}"
    log_path="${RESULT_ROOT}/${name}.log"
    mkdir -p "$model_result"
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
        --num_processes 2 --num_machines 1 --mixed_precision bf16 \
        --dynamo_backend no --main_process_port "$((BASE_PORT + index))" \
        "${PROJECT_DIR}/inference_eval.py" \
        --model_path "$checkpoint" --csv_path "$SUBSET_CSV" \
        --save_root "$model_result" --patch_size 32 224 224 \
        --mixed_precision bf16 --sw_batch_size "$SW_BATCH_SIZE" \
        --prob_threshold 0.5 --min_component_voxels 20 --save_pred \
        --context --soft_prompt_mode disabled --llm_repo "$LLM_REPO" \
        --text_feature_cache "$LOCAL_TEXT_CACHE" \
        --include_cc --include_chief_complaint \
        "${overwrite_args[@]}" 2>&1 | tee "$log_path"
done

"$PYTHON_EXE" "${PROJECT_DIR}/compare_inference_summaries.py" --root "$RESULT_ROOT"
echo "[DONE] v3.4 preview inference completed: $RESULT_ROOT"
