#!/bin/bash
set -euo pipefail

# Wave V7.2: CPU-only safety calibration from cached paired probabilities.
# This wave does not train or run the segmentation models again.

PROJECT_DIR="${PROJECT_DIR:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk}"
PYTHON_EXE="${PYTHON_EXE:-python}"
RESULT_ROOT="${RESULT_ROOT:-/mnt/nas125/forGPU2/lhyunki/llmseg/inference_result/FUdata/posthoc_suppression_v7_matrix_shared_20260807}"
VISION_ANNOTATION="${VISION_ANNOTATION:-${RESULT_ROOT}/vision_probability/annotation.csv}"
DICOM_ANNOTATION="${DICOM_ANNOTATION:-${RESULT_ROOT}/dicom_film_probability/annotation.csv}"
NORMAL_SCORES="${NORMAL_SCORES:-${RESULT_ROOT}/normal_classifier/valid_normal_scores.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULT_ROOT}/wave_v72_safety_calibration}"
MATRIX_DIR="${OUTPUT_DIR}/fine_matrix"
AUDIT_DIR="${OUTPUT_DIR}/tail_safety_audit"
HOLDOUT_DIR="${OUTPUT_DIR}/calibration_holdout"

P_MIN="${P_MIN:-0.2}"
P_PROTECT="${P_PROTECT:-0.85}"
BETAS="${BETAS:-0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.24,0.245,0.25}"
TEXT_THRESHOLDS="${TEXT_THRESHOLDS:-0,0.5,0.7,0.9,0.95}"
MAX_MEAN_DROP="${MAX_MEAN_DROP:-0.01}"
MAX_Q95_INCREMENTAL_DROP="${MAX_Q95_INCREMENTAL_DROP:-0.01}"
MAX_CATASTROPHIC_RATE="${MAX_CATASTROPHIC_RATE:-0.001}"
MAX_NEWLY_MISSED="${MAX_NEWLY_MISSED:-0}"
OVERWRITE_MATRIX="${OVERWRITE_MATRIX:-0}"

for required in "$VISION_ANNOTATION" "$DICOM_ANNOTATION" "$NORMAL_SCORES"; do
    if [ ! -f "$required" ]; then
        echo "[ERROR] Missing required cached artifact: $required"
        exit 2
    fi
done

mkdir -p "$MATRIX_DIR" "$AUDIT_DIR" "$HOLDOUT_DIR"
PER_CASE_CSV="${MATRIX_DIR}/per_case_v7_five_condition_metrics.csv"

echo "[V7.2] CPU-only fine safety calibration"
echo "[V7.2] p_protect=${P_PROTECT}; no adaptive high-confidence override"
echo "[V7.2] beta=${BETAS}"
echo "[V7.2] text_threshold=${TEXT_THRESHOLDS}"

if [ ! -f "$PER_CASE_CSV" ] || [ "$OVERWRITE_MATRIX" = "1" ]; then
    "$PYTHON_EXE" "${PROJECT_DIR}/evaluate_posthoc_suppression_v7_matrix.py" \
        --vision-annotation-csv "$VISION_ANNOTATION" \
        --dicom-annotation-csv "$DICOM_ANNOTATION" \
        --normal-scores-csv "$NORMAL_SCORES" \
        --output-dir "$MATRIX_DIR" \
        --p-min "$P_MIN" \
        --p-protect "$P_PROTECT" \
        --betas "$BETAS" \
        --text-thresholds "$TEXT_THRESHOLDS" \
        --max-sensitivity-drop "$MAX_MEAN_DROP" \
        --fast-screen \
        2>&1 | tee "${OUTPUT_DIR}/fine_matrix.log"
else
    echo "[SKIP] Reusing completed fine matrix: $PER_CASE_CSV"
fi

"$PYTHON_EXE" "${PROJECT_DIR}/audit_posthoc_suppression_v7.py" \
    --per-case-csv "$PER_CASE_CSV" \
    --normal-scores-csv "$NORMAL_SCORES" \
    --output-dir "$AUDIT_DIR" \
    --max-mean-sensitivity-drop-vs-vision "$MAX_MEAN_DROP" \
    --max-q95-incremental-drop-vs-dicom "$MAX_Q95_INCREMENTAL_DROP" \
    --max-catastrophic-drop-rate "$MAX_CATASTROPHIC_RATE" \
    --max-newly-missed-cases "$MAX_NEWLY_MISSED" \
    2>&1 | tee "${OUTPUT_DIR}/tail_safety_audit.log"

"$PYTHON_EXE" "${PROJECT_DIR}/select_v7_operating_point_holdout.py" \
    --per-case-csv "$PER_CASE_CSV" \
    --output-dir "$HOLDOUT_DIR" \
    --max-mean-sensitivity-drop-vs-vision "$MAX_MEAN_DROP" \
    --max-q95-incremental-drop-vs-dicom "$MAX_Q95_INCREMENTAL_DROP" \
    --max-catastrophic-drop-rate "$MAX_CATASTROPHIC_RATE" \
    --max-newly-missed-cases "$MAX_NEWLY_MISSED" \
    2>&1 | tee "${OUTPUT_DIR}/calibration_holdout.log"

echo "[DONE] Wave V7.2 safety calibration: $OUTPUT_DIR"
echo "[NEXT] Review the locked holdout table before any p_protect>0.85 experiment."
