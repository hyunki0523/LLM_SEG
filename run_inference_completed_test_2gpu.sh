#!/bin/bash
set -euo pipefail

# Full held-out test inference for completed checkpoints only.
# Included: v3a cached final, v3.4 control final, v3.4 DS2 final.
# Excluded: all still-running 6-GPU jobs and v3.4 DS3.

PROJECT_DIR="${PROJECT_DIR:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk}"
PYTHON_EXE="${PYTHON_EXE:-python}"
GPU_PAIR="${GPU_PAIR:-0,1}"
TEST_CSV="${TEST_CSV:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_test_set_test2.xlsx}"
V3A_BASE="${V3A_BASE:-/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/llama/safe_film_context_v3a_4gpu_parallel}"
V34_BASE="${V34_BASE:-/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/llama/safe_film_context_v34_partial_ds}"
LLM_REPO="${LLM_REPO:-${PROJECT_DIR}/model_custom/llama2/Llama-2-7b-chat-hf}"
TEST_TEXT_CACHE="${TEST_TEXT_CACHE:-${PROJECT_DIR}/text_feature_cache/llama2_safe_cc_nosoft_test.sqlite3}"
LOCAL_CACHE_DIR="${LOCAL_CACHE_DIR:-/tmp/llmseg_text_cache}"
LOCAL_TEXT_CACHE="${LOCAL_TEXT_CACHE:-${LOCAL_CACHE_DIR}/$(basename "$TEST_TEXT_CACHE")}"
RESULT_ROOT="${RESULT_ROOT:-/mnt/nas125/forGPU2/lhyunki/llmseg/inference_result/FUdata/completed_context_test_$(date +%Y%m%d_%H%M%S)}"
SW_BATCH_SIZE="${SW_BATCH_SIZE:-8}"
PROB_THRESHOLD="${PROB_THRESHOLD:-0.5}"
MIN_COMPONENT_VOXELS="${MIN_COMPONENT_VOXELS:-0}"
BASE_PORT="${BASE_PORT:-29960}"
OVERWRITE_PRED="${OVERWRITE_PRED:-0}"

IFS=',' read -r -a gpu_list <<< "$GPU_PAIR"
if [ "${#gpu_list[@]}" -ne 2 ] \
   || [ "${gpu_list[0]}" = "${gpu_list[1]}" ] \
   || ! [[ "${gpu_list[0]}" =~ ^[0-9]+$ ]] \
   || ! [[ "${gpu_list[1]}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] GPU_PAIR must contain two distinct IDs, e.g. GPU_PAIR=0,1"
    exit 2
fi

NAMES=(v3a_no_soft_cached_final v34_cached_control_final v34_cached_ds2_final)
CHECKPOINTS=(
    "${V3A_BASE}/text_safe_no_soft_cached_1gpu/final_model.pth"
    "${V34_BASE}/text_safe_v34_cached_control_ddp2/final_model.pth"
    "${V34_BASE}/text_safe_v34_cached_ds2_ddp2/final_model.pth"
)
for required in "$TEST_CSV" "${CHECKPOINTS[@]}"; do
    if [ ! -f "$required" ]; then
        echo "[ERROR] Required file not found: $required"
        exit 2
    fi
done

CUDA_VISIBLE_DEVICES="$GPU_PAIR" EXPECTED_GPUS=2 \
    "$PYTHON_EXE" "${PROJECT_DIR}/verify_runtime_environment.py"

mkdir -p "$(dirname "$TEST_TEXT_CACHE")" "$LOCAL_CACHE_DIR" "$RESULT_ROOT"

# The train/valid cache does not cover every held-out test prompt. Build a
# separate test-only cache once, then reuse it without changing the source cache.
cache_ready=0
if [ -f "$TEST_TEXT_CACHE" ]; then
    if "$PYTHON_EXE" "${PROJECT_DIR}/verify_text_feature_cache.py" \
        --csv "$TEST_CSV" --llm-repo "$LLM_REPO" \
        --cache "$TEST_TEXT_CACHE" --coverage-only; then
        cache_ready=1
    fi
fi
if [ "$cache_ready" -ne 1 ]; then
    echo "[CACHE] Building/appending the held-out test text cache on GPU ${gpu_list[0]}."
    CUDA_VISIBLE_DEVICES="${gpu_list[0]}" "$PYTHON_EXE" \
        "${PROJECT_DIR}/precompute_text_features.py" \
        --csv "$TEST_CSV" --llm-repo "$LLM_REPO" \
        --output "$TEST_TEXT_CACHE" --batch-size "${CACHE_BATCH_SIZE:-8}" \
        --max-length 128 --device cuda:0 --commit-every 32
fi
"$PYTHON_EXE" "${PROJECT_DIR}/verify_text_feature_cache.py" \
    --csv "$TEST_CSV" --llm-repo "$LLM_REPO" \
    --cache "$TEST_TEXT_CACHE" --coverage-only

echo "[CACHE] Copying the test cache to local storage: $LOCAL_TEXT_CACHE"
cp -f "$TEST_TEXT_CACHE" "$LOCAL_TEXT_CACHE"

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export LLMSEG_FORCE_MATH_SDP="${LLMSEG_FORCE_MATH_SDP:-1}"

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
    echo "[TEST INFERENCE] $name on GPUs $GPU_PAIR"
    echo "Checkpoint: $checkpoint"
    echo "Log: $log_path"
    echo "=========================================================="
    CUDA_VISIBLE_DEVICES="$GPU_PAIR" accelerate launch \
        --num_processes 2 --num_machines 1 --mixed_precision bf16 \
        --dynamo_backend no --main_process_port "$((BASE_PORT + index))" \
        "${PROJECT_DIR}/inference_eval.py" \
        --model_path "$checkpoint" --csv_path "$TEST_CSV" \
        --save_root "$model_result" --patch_size 32 224 224 \
        --mixed_precision bf16 --sw_batch_size "$SW_BATCH_SIZE" \
        --prob_threshold "$PROB_THRESHOLD" \
        --min_component_voxels "$MIN_COMPONENT_VOXELS" --save_pred \
        --context --soft_prompt_mode disabled --llm_repo "$LLM_REPO" \
        --text_feature_cache "$LOCAL_TEXT_CACHE" \
        --include_cc --include_chief_complaint \
        "${overwrite_args[@]}" 2>&1 | tee "$log_path"
done

"$PYTHON_EXE" "${PROJECT_DIR}/compare_inference_summaries.py" --root "$RESULT_ROOT"
echo "[DONE] Completed-model test inference: $RESULT_ROOT"
