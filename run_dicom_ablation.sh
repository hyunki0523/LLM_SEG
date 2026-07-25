#!/bin/bash
set -euo pipefail

# DICOM prompt ablation with CFG disabled.
# Default target: strongest current DICOM-context checkpoint (Qwen bs2_dicom).

PYTHON_EXE="${PYTHON_EXE:-python}"

PY_SITE=$($PYTHON_EXE -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$PY_SITE/nvidia/nvjitlink/lib:$PY_SITE/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

$PYTHON_EXE - <<'PY'
import sys
import torch

print(f"[CUDA CHECK] torch={torch.__version__}, torch_cuda={torch.version.cuda}, available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("[ERROR] CUDA is not available. Stop before falling back to CPU.")
PY

CSV_PATH="${CSV_PATH:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_test_set_test2.xlsx}"
PATCH_SIZE="${PATCH_SIZE:-32 224 224}"
BASE_SAVE_ROOT="${BASE_SAVE_ROOT:-/mnt/nas125/forGPU2/lhyunki/llmseg/inference_result/FUdata/dicom_ablation_0722}"
METRIC_ROOT="${METRIC_ROOT:-/mnt/nas125/forGPU2/lhyunki/llmseg/inference_result/FUdata/dicom_ablation_0722_metrics}"

MIXED_PRECISION_MODE="${MIXED_PRECISION_MODE:-bf16}"
SAVE_OPT="${SAVE_OPT:---save_pred}"
SW_BATCH="${SW_BATCH:---sw_batch_size 16}"
POSTPROC_OPT="${POSTPROC_OPT:---min_component_voxels 20}"
CFG_OPT="${CFG_OPT:---cfg_scale 1}"
OVERWRITE_OPT="${OVERWRITE_OPT:-}"
RUN_METRICS_AFTER="${RUN_METRICS_AFTER:-1}"
OVERWRITE_METRICS="${OVERWRITE_METRICS:-0}"

MODEL_SPECS=(
    "qwen_bs2_dicom|/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/qwen/bs2_dicom/model_epoch_300.pth|--llm_repo /mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_seg_jhk/model_custom/qwen3/Qwen3-8B-Base"
    # "llama_bs2_dicom|/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/llama/bs2_dicom/model_epoch_300.pth|"
)

MODE_SPECS=(
    "dicom_full_demo|full|1"
    "dicom_full|full|0"
    "dicom_limited|limited|0"
    "dicom_geometry|geometry|0"
    "dicom_kernel_only|kernel_only|0"
    "dicom_spacing_only|spacing_only|0"
    "dicom_scanner_only|scanner_only|0"
    "dicom_protocol_only|protocol_only|0"
    "dicom_none|none|0"
)

mkdir -p "$BASE_SAVE_ROOT" "$METRIC_ROOT"

echo "[INFO] CSV_PATH=$CSV_PATH"
echo "[INFO] BASE_SAVE_ROOT=$BASE_SAVE_ROOT"
echo "[INFO] METRIC_ROOT=$METRIC_ROOT"
echo "[INFO] CFG_OPT=$CFG_OPT"
echo "[INFO] RUN_METRICS_AFTER=$RUN_METRICS_AFTER"

for model_spec in "${MODEL_SPECS[@]}"; do
    IFS='|' read -r MODEL_TAG MODEL_PATH LLM_REPO_OPT <<< "$model_spec"

    for mode_spec in "${MODE_SPECS[@]}"; do
        IFS='|' read -r MODE_TAG DICOM_MODE INCLUDE_DEMO <<< "$mode_spec"

        if [ "$INCLUDE_DEMO" = "1" ]; then
            DEMO_OPT="--include_demographics"
        else
            DEMO_OPT="--no-include_demographics"
        fi

        SAVE_ROOT="${BASE_SAVE_ROOT}/${MODEL_TAG}_${MODE_TAG}"

        echo "=========================================================="
        echo "[START] DICOM Ablation: ${MODEL_TAG}_${MODE_TAG}"
        echo "Model Path       : $MODEL_PATH"
        echo "DICOM mode       : $DICOM_MODE"
        echo "Demographics     : $INCLUDE_DEMO"
        echo "CFG              : $CFG_OPT"
        echo "Save Root        : $SAVE_ROOT"
        echo "=========================================================="

        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" accelerate launch \
            --num_processes 1 \
            --num_machines 1 \
            --mixed_precision "$MIXED_PRECISION_MODE" \
            --dynamo_backend no \
            inference_eval.py \
            --model_path "$MODEL_PATH" \
            --csv_path "$CSV_PATH" \
            --save_root "$SAVE_ROOT" \
            --patch_size $PATCH_SIZE \
            --context \
            --no-include_cc \
            --no-include_emr \
            $DEMO_OPT \
            --dicom_prompt_mode "$DICOM_MODE" \
            --mixed_precision "$MIXED_PRECISION_MODE" \
            $SAVE_OPT \
            $SW_BATCH \
            $POSTPROC_OPT \
            $CFG_OPT \
            $OVERWRITE_OPT \
            $LLM_REPO_OPT

        echo "[DONE] Inference: ${MODEL_TAG}_${MODE_TAG}"

        if [ "$RUN_METRICS_AFTER" = "1" ]; then
            SAVE_CSV="${METRIC_ROOT}/${MODEL_TAG}_${MODE_TAG}_metrics.csv"
            if [ -f "$SAVE_CSV" ] && [ "$OVERWRITE_METRICS" != "1" ]; then
                echo "[SKIP] Existing metrics: $SAVE_CSV"
            else
                echo "[START] Metrics: ${MODEL_TAG}_${MODE_TAG}"
                $PYTHON_EXE compute_metrics.py \
                    --pred_root "$SAVE_ROOT" \
                    --csv_path "$CSV_PATH" \
                    --output_csv "$SAVE_CSV" \
                    --hemo_dice_class_filter \
                    --exclude_class_hemo_without_mask_volume \
                    --require_image_mask_pair
                echo "[DONE] Metrics: ${MODEL_TAG}_${MODE_TAG}"
            fi
        fi

        echo ""
    done
done

if [ "$RUN_METRICS_AFTER" = "1" ]; then
    echo "=========================================================="
    echo "[AUTO OVERVIEW] Generate combined overview"
    echo "=========================================================="
    $PYTHON_EXE generate_metric_overview.py "$METRIC_ROOT"
fi

echo "[DONE] DICOM ablation completed."
