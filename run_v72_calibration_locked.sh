#!/bin/bash
set -euo pipefail

# CPU calibration/locked-validation stage. Probability extraction must finish
# first; no segmentation checkpoint is tuned on the locked cohort.

PROJECT_DIR="${PROJECT_DIR:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk}"
PYTHON_EXE="${PYTHON_EXE:-python}"
RESULT_ROOT="${RESULT_ROOT:-}"
VISION_MANIFEST="${VISION_MANIFEST:-${PROJECT_DIR}/data_manifests/vision_balanced_v11.csv}"
P_MIN_VALUES="${P_MIN_VALUES:-0.1,0.2,0.3}"
P_PROTECT_VALUES="${P_PROTECT_VALUES:-0.85,0.90,0.95}"
BETAS="${BETAS:-0.1,0.25,0.5,0.75,1,1.5,2,3}"
TEXT_THRESHOLDS="${TEXT_THRESHOLDS:-0,0.5,0.7,0.9}"
MAX_SENSITIVITY_DROP="${MAX_SENSITIVITY_DROP:-0.01}"
SPLIT_SEED="${SPLIT_SEED:-20260809}"

if [ -z "$RESULT_ROOT" ]; then
    echo "[ERROR] Set RESULT_ROOT to the completed paired probability directory."
    exit 2
fi
VISION_ANNOTATION="${VISION_ANNOTATION:-${RESULT_ROOT}/vision_only_probability/annotation.csv}"
DICOM_ANNOTATION="${DICOM_ANNOTATION:-${RESULT_ROOT}/dicom_film_probability/annotation.csv}"
NORMAL_SCORES="${NORMAL_SCORES:-${RESULT_ROOT}/normal_classifier/valid_normal_scores.csv}"
V72_ROOT="${V72_ROOT:-${RESULT_ROOT}/v72_calibration_locked}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${V72_ROOT}/v72_split.csv}"
CALIBRATION_ROOT="${V72_ROOT}/calibration"
LOCKED_ROOT="${V72_ROOT}/locked_selected"

for required in \
    "$VISION_ANNOTATION" "$DICOM_ANNOTATION" "$NORMAL_SCORES" "$VISION_MANIFEST"; do
    if [ ! -f "$required" ]; then
        echo "[ERROR] Required artifact not found: $required"
        exit 2
    fi
done
mkdir -p "$CALIBRATION_ROOT" "$LOCKED_ROOT"

if [ ! -f "$SPLIT_MANIFEST" ]; then
    "$PYTHON_EXE" "${PROJECT_DIR}/build_v72_calibration_split.py" \
        --annotation-csv "$VISION_ANNOTATION" \
        --vision-manifest "$VISION_MANIFEST" \
        --output "$SPLIT_MANIFEST" \
        --calibration-fraction 0.5 \
        --seed "$SPLIT_SEED"
else
    echo "[SPLIT] Reusing locked split: $SPLIT_MANIFEST"
fi

echo "[CALIBRATION] Loading each probability volume once for the full band sweep"
"$PYTHON_EXE" "${PROJECT_DIR}/evaluate_posthoc_suppression_v7_matrix.py" \
    --vision-annotation-csv "$VISION_ANNOTATION" \
    --dicom-annotation-csv "$DICOM_ANNOTATION" \
    --normal-scores-csv "$NORMAL_SCORES" \
    --split-manifest "$SPLIT_MANIFEST" \
    --cohort-part calibration \
    --output-dir "$CALIBRATION_ROOT" \
    --p-mins "$P_MIN_VALUES" \
    --p-protects "$P_PROTECT_VALUES" \
    --betas "$BETAS" \
    --text-thresholds "$TEXT_THRESHOLDS" \
    --max-sensitivity-drop "$MAX_SENSITIVITY_DROP" \
    --fast-screen \
    2>&1 | tee "${CALIBRATION_ROOT}/calibration.log"

SELECTED_JSON="${V72_ROOT}/selected_calibration_candidate.json"
SELECTED_ENV="${V72_ROOT}/selected_calibration_candidate.env"
"$PYTHON_EXE" "${PROJECT_DIR}/select_v72_calibration_candidate.py" \
    --summary "${CALIBRATION_ROOT}/v7_five_condition_summary.csv" \
    --output-json "$SELECTED_JSON" \
    --output-env "$SELECTED_ENV"
# The generated file contains numeric assignments only.
# shellcheck disable=SC1090
source "$SELECTED_ENV"

echo "[LOCKED] Evaluating the single selected operating point"
"$PYTHON_EXE" "${PROJECT_DIR}/evaluate_posthoc_suppression_v7_matrix.py" \
    --vision-annotation-csv "$VISION_ANNOTATION" \
    --dicom-annotation-csv "$DICOM_ANNOTATION" \
    --normal-scores-csv "$NORMAL_SCORES" \
    --split-manifest "$SPLIT_MANIFEST" \
    --cohort-part locked \
    --output-dir "$LOCKED_ROOT" \
    --p-min "$SELECTED_P_MIN" \
    --p-protect "$SELECTED_P_PROTECT" \
    --betas "$SELECTED_BETA" \
    --text-thresholds "$SELECTED_TEXT_THRESHOLD" \
    --max-sensitivity-drop "$MAX_SENSITIVITY_DROP" \
    2>&1 | tee "${LOCKED_ROOT}/locked_validation.log"

"$PYTHON_EXE" "${PROJECT_DIR}/audit_posthoc_suppression_v7.py" \
    --per-case-csv "${LOCKED_ROOT}/per_case_v7_five_condition_metrics.csv" \
    --normal-scores-csv "$NORMAL_SCORES" \
    --output-dir "${LOCKED_ROOT}/tail_safety" \
    --max-mean-sensitivity-drop-vs-vision "$MAX_SENSITIVITY_DROP" \
    2>&1 | tee "${LOCKED_ROOT}/tail_safety.log"

echo "[DONE] V7.2 calibration/locked validation: $V72_ROOT"
