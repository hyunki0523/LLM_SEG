#!/bin/bash
set -euo pipefail

# Encode deterministic prompt shards on separate GPUs and merge their SQLite
# databases afterwards. Never let multiple encoders write the same DB.

PROJECT_DIR="${PROJECT_DIR:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk}"
PYTHON_EXE="${PYTHON_EXE:-python}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5}"
TRAIN_CSV="${TRAIN_CSV:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_train_set.xlsx}"
VALID_CSV="${VALID_CSV:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_valid_set.xlsx}"
LLM_REPO="${LLM_REPO:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_seg_jhk/model_custom/llama2/Llama-2-7b-chat-hf/}"
DICOM_PROMPT_MODE="${DICOM_PROMPT_MODE:-full}"
TARGET_CACHE="${TARGET_CACHE:-${PROJECT_DIR}/text_feature_cache/llama2_safe_cc_dicom_full_nosoft_deterministic.sqlite3}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-8}"
CACHE_COMMIT_EVERY="${CACHE_COMMIT_EVERY:-16}"
STARTUP_STAGGER_SECONDS="${STARTUP_STAGGER_SECONDS:-5}"

read -r -a gpu_list <<< "$GPU_IDS"
if [ "${#gpu_list[@]}" -lt 2 ]; then
    echo "[ERROR] GPU_IDS must contain at least two IDs, e.g. GPU_IDS='0 1 2 3 4 5'"
    exit 2
fi
for gpu in "${gpu_list[@]}"; do
    if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] Invalid GPU ID: $gpu"
        exit 2
    fi
done
if [ "$(printf '%s\n' "${gpu_list[@]}" | sort -u | wc -l)" -ne "${#gpu_list[@]}" ]; then
    echo "[ERROR] GPU_IDS must contain distinct devices."
    exit 2
fi

num_shards="${#gpu_list[@]}"
target_name="$(basename "$TARGET_CACHE" .sqlite3)"
PART_DIR="${PART_DIR:-${PROJECT_DIR}/text_feature_cache/parallel_parts/${target_name}_${num_shards}way}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/train_logs/text_cache_${DICOM_PROMPT_MODE}_${num_shards}gpu_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$PART_DIR" "$LOG_ROOT" "$(dirname "$TARGET_CACHE")"
cd "$PROJECT_DIR"

selected_cuda_devices="$(IFS=,; echo "${gpu_list[*]}")"
CUDA_VISIBLE_DEVICES="$selected_cuda_devices" EXPECTED_GPUS="$num_shards" \
    "$PYTHON_EXE" "${PROJECT_DIR}/verify_runtime_environment.py"

PY_SITE=$($PYTHON_EXE -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$PY_SITE/nvidia/nvjitlink/lib:$PY_SITE/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TRANSFORMERS_NO_TORCHAUDIO="${TRANSFORMERS_NO_TORCHAUDIO:-1}"
export TRANSFORMERS_NO_AUDIO="${TRANSFORMERS_NO_AUDIO:-1}"
export LLMSEG_DISABLE_TORCHAUDIO="${LLMSEG_DISABLE_TORCHAUDIO:-1}"

skip_args=()
if [ -f "$TARGET_CACHE" ]; then
    skip_args=(--skip-cache "$TARGET_CACHE")
fi

PIDS=()
SHARDS=()
for index in "${!gpu_list[@]}"; do
    gpu="${gpu_list[$index]}"
    shard_path="${PART_DIR}/shard_${index}_of_${num_shards}.sqlite3"
    shard_log="${LOG_ROOT}/shard_${index}_gpu_${gpu}.log"
    SHARDS+=("$shard_path")
    echo "[LAUNCH] shard=$index/$num_shards GPU=$gpu log=$shard_log"
    (
        PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_EXE" -u precompute_text_features.py \
            --csv "$TRAIN_CSV" --csv "$VALID_CSV" \
            --llm-repo "$LLM_REPO" \
            --output "$shard_path" \
            --dicom-prompt-mode "$DICOM_PROMPT_MODE" \
            --batch-size "$CACHE_BATCH_SIZE" \
            --commit-every "$CACHE_COMMIT_EVERY" \
            --num-shards "$num_shards" \
            --shard-index "$index" \
            "${skip_args[@]}" \
            --device cuda:0
    ) >"$shard_log" 2>&1 &
    PIDS+=("$!")
    if [ "$STARTUP_STAGGER_SECONDS" -gt 0 ] && [ "$index" -lt $((num_shards - 1)) ]; then
        sleep "$STARTUP_STAGGER_SECONDS"
    fi
done

failed=0
for index in "${!PIDS[@]}"; do
    if wait "${PIDS[$index]}"; then
        echo "[DONE] shard=$index path=${SHARDS[$index]}"
    else
        echo "[ERROR] shard=$index failed: ${LOG_ROOT}/shard_${index}_gpu_${gpu_list[$index]}.log"
        failed=1
    fi
done
if [ "$failed" -ne 0 ]; then
    echo "[ERROR] At least one shard failed; completed shards remain resumable in $PART_DIR"
    exit 1
fi

merge_args=()
for shard in "${SHARDS[@]}"; do
    merge_args+=(--source "$shard")
done
"$PYTHON_EXE" merge_text_feature_caches.py \
    --target "$TARGET_CACHE" "${merge_args[@]}"

"$PYTHON_EXE" verify_text_feature_cache.py \
    --csv "$TRAIN_CSV" --csv "$VALID_CSV" \
    --llm-repo "$LLM_REPO" \
    --cache "$TARGET_CACHE" \
    --dicom-prompt-mode "$DICOM_PROMPT_MODE" \
    --coverage-only

echo "[DONE] Parallel cache complete: $TARGET_CACHE"
echo "[INFO] Shards retained for audit/resume: $PART_DIR"
