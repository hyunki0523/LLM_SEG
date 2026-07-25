#!/bin/bash
set -euo pipefail

PYTHON_EXE="${PYTHON_EXE:-python3}"

# Use the final pair-only test2 file. It already contains FUtest_data image_path
# and mask_path columns, including generated null masks for Normal cases.
CSV_PATH="${CSV_PATH:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_test_set_test2.xlsx}"

INFERENCE_BASE="${INFERENCE_BASE:-/mnt/nas125/forGPU2/lhyunki/llmseg/inference_result/FUdata/test0630}"
METRIC_ROOT="${METRIC_ROOT:-/mnt/nas125/forGPU2/lhyunki/llmseg/inference_result/FUdata/test0630/final_test_set_test2_metrics_0721}"

# Set OVERWRITE_METRICS=1 to recompute files that already exist.
OVERWRITE_METRICS="${OVERWRITE_METRICS:-0}"

# Set HEMO_DICE_CLASS_FILTER=0 to fall back to mask-positive-only hemo Dice.
HEMO_DICE_CLASS_FILTER="${HEMO_DICE_CLASS_FILTER:-1}"

# Exclude class-hemo cases if their GT mask is missing or has zero foreground.
EXCLUDE_CLASS_HEMO_WITHOUT_MASK_VOLUME="${EXCLUDE_CLASS_HEMO_WITHOUT_MASK_VOLUME:-1}"

# final_test_set_test2.xlsx is pair-only and has physical masks, so missing
# Normal masks should not be synthesized during metrics by default.
TREAT_MISSING_NORMAL_AS_EMPTY="${TREAT_MISSING_NORMAL_AS_EMPTY:-0}"

# Require the final CSV rows to have both FUtest_data image and mask files.
REQUIRE_IMAGE_MASK_PAIR="${REQUIRE_IMAGE_MASK_PAIR:-1}"

mkdir -p "$METRIC_ROOT"

FILTER_OPT=()
if [ "$HEMO_DICE_CLASS_FILTER" = "1" ]; then
    FILTER_OPT+=(--hemo_dice_class_filter)
fi
if [ "$EXCLUDE_CLASS_HEMO_WITHOUT_MASK_VOLUME" = "1" ]; then
    FILTER_OPT+=(--exclude_class_hemo_without_mask_volume)
fi
if [ "$TREAT_MISSING_NORMAL_AS_EMPTY" = "1" ]; then
    FILTER_OPT+=(--treat_missing_normal_as_empty)
fi
if [ "$REQUIRE_IMAGE_MASK_PAIR" = "1" ]; then
    FILTER_OPT+=(--require_image_mask_pair)
else
    FILTER_OPT+=(--no-require_image_mask_pair)
fi

echo "[INFO] CSV_PATH=$CSV_PATH"
echo "[INFO] INFERENCE_BASE=$INFERENCE_BASE"
echo "[INFO] METRIC_ROOT=$METRIC_ROOT"
echo "[INFO] HEMO_DICE_CLASS_FILTER=$HEMO_DICE_CLASS_FILTER"
echo "[INFO] EXCLUDE_CLASS_HEMO_WITHOUT_MASK_VOLUME=$EXCLUDE_CLASS_HEMO_WITHOUT_MASK_VOLUME"
echo "[INFO] TREAT_MISSING_NORMAL_AS_EMPTY=$TREAT_MISSING_NORMAL_AS_EMPTY"
echo "[INFO] REQUIRE_IMAGE_MASK_PAIR=$REQUIRE_IMAGE_MASK_PAIR"

metric_root_abs=$(readlink -f "$METRIC_ROOT" 2>/dev/null || echo "$METRIC_ROOT")

mapfile -t INFERENCE_DIRS < <(
    find "$INFERENCE_BASE" -mindepth 1 -maxdepth 1 -type d | while read -r dir; do
        dir_abs=$(readlink -f "$dir" 2>/dev/null || echo "$dir")
        base=$(basename "$dir")
        if [ "$dir_abs" = "$metric_root_abs" ]; then
            continue
        fi
        case "$base" in
            final_test_set_0703*|metric_results*)
                continue
                ;;
        esac
        echo "$dir"
    done | sort
)

for SAVE_ROOT in "${INFERENCE_DIRS[@]}"; do
    ARG=$(basename "$SAVE_ROOT")
    SAVE_CSV="${METRIC_ROOT}/${ARG}_metrics.csv"

    if [ -f "$SAVE_CSV" ] && [ "$OVERWRITE_METRICS" != "1" ]; then
        echo "[SKIP] Existing metrics: $ARG"
        continue
    fi

    echo "=========================================================="
    echo "[START] Metrics Calculation: $ARG"
    echo "Target Root : $SAVE_ROOT"
    echo "Output CSV  : $SAVE_CSV"
    echo "=========================================================="

    "$PYTHON_EXE" compute_metrics.py \
        --pred_root "$SAVE_ROOT" \
        --csv_path "$CSV_PATH" \
        --output_csv "$SAVE_CSV" \
        "${FILTER_OPT[@]}"

    echo "[DONE] Finished Metrics for $ARG"
    echo ""
done

echo "=========================================================="
echo "[AUTO OVERVIEW] Generate combined overview"
echo "=========================================================="
"$PYTHON_EXE" generate_metric_overview.py "$METRIC_ROOT"
