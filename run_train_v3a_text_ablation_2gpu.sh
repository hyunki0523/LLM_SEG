#!/bin/bash
set -euo pipefail

# Controlled v3a text experiment on one 2-GPU pair:
#   1. learned soft prompt, online Llama
#   2. disabled soft prompt, online Llama
#   3. disabled soft prompt, cached frozen-Llama features

PROJECT_DIR="${PROJECT_DIR:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk}"
BASE_LAUNCHER="${BASE_LAUNCHER:-${PROJECT_DIR}/run_train_dicom_ablation_8gpu.sh}"
GPU_PAIR="${GPU_PAIR:-0,1}"
V3A_EXPERIMENTS="${V3A_EXPERIMENTS:-soft_prompt_online,no_soft_online,no_soft_cached}"
TEXT_FEATURE_CACHE="${TEXT_FEATURE_CACHE:-${PROJECT_DIR}/text_feature_cache/llama2_safe_cc_nosoft.sqlite3}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/llama/safe_film_context_v3a}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/train_logs/llama_safe_film_context_v3a_$(date +%Y%m%d_%H%M%S)}"
SMOKE_TEST="${SMOKE_TEST:-0}"

IFS=',' read -r gpu0 gpu1 extra_gpu <<< "$GPU_PAIR"
if [ -z "${gpu0:-}" ] || [ -z "${gpu1:-}" ] || [ -n "${extra_gpu:-}" ]; then
    echo "[ERROR] GPU_PAIR must contain exactly two GPU IDs, e.g. 0,1"
    exit 2
fi
if [ ! -f "$BASE_LAUNCHER" ]; then
    echo "[ERROR] Base launcher not found: $BASE_LAUNCHER"
    exit 2
fi

if [ "$SMOKE_TEST" = "1" ]; then
    export EPOCHS="${EPOCHS:-1}"
    export N_ITER_PER_EPOCH="${N_ITER_PER_EPOCH:-2}"
    export N_ITER_VALID="${N_ITER_VALID:-1}"
    export BATCH_SIZE="${BATCH_SIZE:-1}"
    export GRAD_ACCUM="${GRAD_ACCUM:-1}"
    export NUM_WORKERS="${NUM_WORKERS:-2}"
    export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1}"
    export OVERWRITE_TRAIN="${OVERWRITE_TRAIN:-1}"
    CHECKPOINT_BASE="${CHECKPOINT_BASE_SMOKE:-${PROJECT_DIR}/_debug_ckpt/llama_safe_film_context_v3a}"
    LOG_ROOT="${LOG_ROOT_SMOKE:-${PROJECT_DIR}/train_logs/llama_safe_film_context_v3a_smoke_$(date +%Y%m%d_%H%M%S)}"
    export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
fi

mkdir -p "$LOG_ROOT" "$CHECKPOINT_BASE"

run_variant() {
    local variant="$1"
    local soft_prompt_mode="$2"
    local cache_path="$3"
    local suffix="_${variant}"

    echo "=========================================================="
    echo "[V3A] $variant"
    echo "GPU pair     : $GPU_PAIR"
    echo "Soft prompt  : $soft_prompt_mode"
    echo "Text cache   : ${cache_path:-<online>}"
    echo "=========================================================="

    PROJECT_DIR="$PROJECT_DIR" \
    GPU_PAIRS="$GPU_PAIR" \
    MAX_PARALLEL=1 \
    ONLY_EXPERIMENTS=text_safe \
    SOFT_PROMPT_MODE="$soft_prompt_mode" \
    TEXT_FEATURE_CACHE="$cache_path" \
    EXPERIMENT_NAME_SUFFIX="$suffix" \
    CHECKPOINT_BASE="$CHECKPOINT_BASE" \
    LOG_ROOT="$LOG_ROOT" \
    bash "$BASE_LAUNCHER"
}

IFS=',' read -r -a selected <<< "$V3A_EXPERIMENTS"
for variant in "${selected[@]}"; do
    case "$variant" in
        soft_prompt_online)
            run_variant "$variant" learned ""
            ;;
        no_soft_online)
            run_variant "$variant" disabled ""
            ;;
        no_soft_cached)
            if [ ! -f "$TEXT_FEATURE_CACHE" ]; then
                echo "[ERROR] Cache not found: $TEXT_FEATURE_CACHE"
                echo "[HINT] Run precompute_text_features.py first."
                exit 2
            fi
            run_variant "$variant" disabled "$TEXT_FEATURE_CACHE"
            ;;
        *)
            echo "[ERROR] Unknown V3A experiment: $variant"
            exit 2
            ;;
    esac
done

echo "[DONE] v3a text ablation completed. Logs: $LOG_ROOT"
