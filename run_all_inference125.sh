#!/bin/bash
set -euo pipefail

# ==========================================
# 1. Common paths and options
# ==========================================
PROJECT_DIR="${PROJECT_DIR:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_seg_jhk}"
CSV_PATH="${CSV_PATH:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_test_set_test2.xlsx}"
PATCH_SIZE="${PATCH_SIZE:-16 224 224}"

# Keep post-processed outputs separate from old final_0616 predictions.
POSTPROC_TAG="${POSTPROC_TAG:-cc20}"
BASE_SAVE_ROOT="${BASE_SAVE_ROOT:-/mnt/nas125/forGPU2/lhyunki/llmseg/inference_result/final_0616_${POSTPROC_TAG}}"
METRIC_ROOT="${METRIC_ROOT:-/mnt/nas125/forGPU2/lhyunki/llmseg/metric_results_0616/final_test_set_${POSTPROC_TAG}}"

MIXED_PRECISION="${MIXED_PRECISION:---mixed_precision fp16}"
SAVE_OPT="${SAVE_OPT:---save_pred}"
SW_BATCH="${SW_BATCH:---sw_batch_size 4}"
POSTPROC_OPT="${POSTPROC_OPT:---min_component_voxels 20}"
CFG_OPT="${CFG_OPT:---cfg_scale 1}"
OVERWRITE_OPT="${OVERWRITE_OPT:-}"

RUN_METRICS_AFTER="${RUN_METRICS_AFTER:-1}"
OVERWRITE_METRICS="${OVERWRITE_METRICS:-0}"
PYTHON_EXE="${PYTHON_EXE:-python}"
DOCKER_BIN="${DOCKER_BIN:-docker}"

CONTAINER_NAMES=(
    "lhyunki_llmseg"
    "lhyunki_llmseg5"
    "lhyunki_llmseg6"
    "lhyunki_llmseg7"
)

GPU_ID_LIST="${GPU_ID_LIST:-4 5 6 7}"
read -r -a GPU_IDS <<< "$GPU_ID_LIST"
MAX_PARALLEL="${MAX_PARALLEL:-4}"

LOG_ROOT="${LOG_ROOT:-./inference_logs/final_0616_${POSTPROC_TAG}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_ROOT"

# ==========================================
# 2. Models (path|context options)
# ==========================================
MODELS=(
    "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/llama/bs2_cfg/model_epoch_300.pth|--context"
    "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/llama/bs2_dicom/model_epoch_300.pth|--context --no-include_emr"
    "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/llama/bs2_lora/model_epoch_300.pth|--context"
    "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/llama/bs4_cfg/model_epoch_300.pth|--context"

    "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/qwen/bs2_cfg/model_epoch_300.pth|--context"
    "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/qwen/bs2_cfg_dicom_cc/model_epoch_300.pth|--context --no-include_emr --include_cc"
    "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/qwen/bs2_context_lora/model_epoch_300.pth|--context"
    "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/qwen/bs2_dicom/model_epoch_300.pth|--context --no-include_emr"

    "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/visionly/model_epoch_300.pth|--no-context"
)

if [ "${#CONTAINER_NAMES[@]}" -ne 4 ] || [ "${#GPU_IDS[@]}" -ne 4 ]; then
    echo "[ERROR] Containers and GPUs count must be exactly 4."
    exit 1
fi

make_arg() {
    local model_path="$1"
    local dir_level_1 dir_level_2 dir_level_3
    dir_level_1=$(dirname "$model_path")
    dir_level_2=$(dirname "$dir_level_1")
    dir_level_3=$(dirname "$dir_level_2")
    echo "$(basename "$dir_level_3")_$(basename "$dir_level_2")_$(basename "$dir_level_1")"
}

run_command_in_container() {
    local container="$1"
    local gpu_id="$2"
    local log_path="$3"
    local command_body="$4"

    echo "[INFO] Container=${container}, CUDA_VISIBLE_DEVICES=${gpu_id}" | tee "$log_path"

    "$DOCKER_BIN" start "$container" >/dev/null 2>&1 || true
    if [ "$("$DOCKER_BIN" inspect -f '{{.State.Running}}' "$container" 2>/dev/null || echo false)" = "true" ]; then
        "$DOCKER_BIN" exec \
            -e CUDA_VISIBLE_DEVICES="$gpu_id" \
            -w "$PROJECT_DIR" \
            "$container" \
            bash -lc "$command_body" >>"$log_path" 2>&1
        return $?
    fi

    {
        printf 'export CUDA_VISIBLE_DEVICES=%q\n' "$gpu_id"
        printf '%s\n' "$command_body"
        printf 'exit\n'
    } | "$DOCKER_BIN" start -ai "$container" >>"$log_path" 2>&1
}

build_inference_command() {
    local model_path="$1"
    local context_flag="$2"
    local save_root="$3"

    # [수정] nibabel 설치 과정을 컨테이너 실행 명령 내부(진입 시점)로 포함
    cat <<EOF
set -euo pipefail
cd "$PROJECT_DIR"

echo "📦 Installing nibabel inside container..."
$PYTHON_EXE -m pip install --text-quiet nibabel || $PYTHON_EXE -m pip install nibabel
$PYTHON_EXE -m pip peft --text-quiet peft || $PYTHON_EXE -m pip install peft

PY_SITE=\$($PYTHON_EXE -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="\$PY_SITE/nvidia/nvjitlink/lib:\$PY_SITE/nvidia/cuda_runtime/lib:\${LD_LIBRARY_PATH:-}"

accelerate launch --num_processes 1 --dynamo_backend no inference_eval.py \\
    --model_path "$model_path" \\
    --csv_path "$CSV_PATH" \\
    --save_root "$save_root" \\
    --patch_size "$PATCH_SIZE" \\
    $context_flag \\
    $MIXED_PRECISION \\
    $SAVE_OPT \\
    $SW_BATCH \\
    $POSTPROC_OPT \\
    $CFG_OPT \\
    $OVERWRITE_OPT
EOF
}

run_metric_for_result() {
    local save_root="$1"
    local arg="$2"
    local save_csv="${METRIC_ROOT}/${arg}_metrics.csv"
    local log_path="${LOG_ROOT}/metric_${arg}.log"
    local container="${CONTAINER_NAMES[0]}"
    local gpu_id="${GPU_IDS[0]}"

    if [ "$OVERWRITE_METRICS" != "1" ] && [ -f "$save_csv" ]; then
        echo "[SKIP] Metric already exists: $save_csv"
        return 0
    fi

    # [수정] 평가지표용 메인 컨테이너 진입 시에도 nibabel 설치 보장
    local metric_cmd
    metric_cmd=$(cat <<EOF
set -euo pipefail
cd "$PROJECT_DIR"
$PYTHON_EXE -m pip install --text-quiet nibabel || $PYTHON_EXE -m pip install nibabel
mkdir -p "$METRIC_ROOT"
$PYTHON_EXE compute_metrics.py --pred_root "$save_root" --csv_path "$CSV_PATH" --output_csv "$save_csv"
EOF
)

    echo "=========================================================="
    echo "[START] Metrics: $arg"
    echo "Log: $log_path"
    echo "=========================================================="
    run_command_in_container "$container" "$gpu_id" "$log_path" "$metric_cmd"
}

wait_batch() {
    local fail=0
    local pid
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            fail=1
        fi
    done
    PIDS=()

    if [ "$fail" -ne 0 ]; then
        echo "[ERROR] At least one inference job failed. Check logs in: $LOG_ROOT"
        exit 1
    fi
}

# ==========================================
# 3. Run inference in 4-container batches
# ==========================================
PIDS=()
RESULT_DIRS=()
RESULT_ARGS=()

for idx in "${!MODELS[@]}"; do
    item="${MODELS[$idx]}"
    model_path="${item%%|*}"
    context_flag="${item##*|}"
    arg="$(make_arg "$model_path")"
    save_root="${BASE_SAVE_ROOT}/${arg}"

    slot=$((idx % MAX_PARALLEL))
    container="${CONTAINER_NAMES[$slot]}"
    gpu_id="${GPU_IDS[$slot]}"
    log_path="${LOG_ROOT}/${arg}.log"

    RESULT_DIRS+=("$save_root")
    RESULT_ARGS+=("$arg")

    echo "=========================================================="
    echo "[START] Inference: $arg"
    echo "Container : $container"
    echo "GPU       : $gpu_id"
    echo "Log       : $log_path"
    echo "=========================================================="

    cmd="$(build_inference_command "$model_path" "$context_flag" "$save_root")"
    run_command_in_container "$container" "$gpu_id" "$log_path" "$cmd" &
    PIDS+=("$!")

    if [ "${#PIDS[@]}" -ge "$MAX_PARALLEL" ]; then
        wait_batch
    fi
done

if [ "${#PIDS[@]}" -gt 0 ]; then
    wait_batch
fi

echo "[DONE] All inference jobs completed."

# ==========================================
# 4. Run metrics after inference
# ==========================================
if [ "$RUN_METRICS_AFTER" = "1" ]; then
    echo "=========================================================="
    echo "[START] Metrics after inference"
    echo "Metric Root: $METRIC_ROOT"
    echo "=========================================================="

    for idx in "${!RESULT_DIRS[@]}"; do
        run_metric_for_result "${RESULT_DIRS[$idx]}" "${RESULT_ARGS[$idx]}"
    done

    overview_log="${LOG_ROOT}/metric_overview.log"
    overview_cmd=$(cat <<EOF
set -euo pipefail
cd "$PROJECT_DIR"
$PYTHON_EXE -m pip install --text-quiet nibabel || $PYTHON_EXE -m pip install nibabel
$PYTHON_EXE generate_metric_overview.py "$METRIC_ROOT"
EOF
)
    run_command_in_container "${CONTAINER_NAMES[0]}" "${GPU_IDS[0]}" "$overview_log" "$overview_cmd"
    echo "[DONE] Metrics and overview completed."
fi

echo "[DONE] Pipeline finished. Logs: $LOG_ROOT"
