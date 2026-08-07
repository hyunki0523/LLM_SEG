#!/bin/bash
set -euo pipefail

# Wave V7.1: matched Wave0 Vision/DICOM-FiLM five-condition fast validation.
# The DICOM probability extraction and normal classifier are expected to have
# completed in the shared result root. All six GPUs are then assigned to the
# remaining Wave0 Vision probability extraction before the CPU matrix screen.

PROJECT_DIR="${PROJECT_DIR:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk}"
BASE_LAUNCHER="${BASE_LAUNCHER:-${PROJECT_DIR}/run_posthoc_suppression_v7_6gpu.sh}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5}"
RESULT_ROOT="${RESULT_ROOT:-/mnt/nas125/forGPU2/lhyunki/llmseg/inference_result/FUdata/posthoc_suppression_v7_matrix_shared_20260807}"
VISION_CHECKPOINT="${VISION_CHECKPOINT:-/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/llama/hybrid_wave0_vision_ema_ds2/vision_only_wave0_vision_ema_ds2/model_epoch_120.pth}"
DICOM_FILM_CHECKPOINT="${DICOM_FILM_CHECKPOINT:-/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/llama/hybrid_wave0_dicom_film_ema_ds2/dicom_film_wave0_dicom_film_ema_ds2/model_epoch_120.pth}"
TEXT_FEATURE_CACHE="${TEXT_FEATURE_CACHE:-${PROJECT_DIR}/text_feature_cache/llama2_safe_cc_nosoft_deterministic.sqlite3}"

CLASSIFIER_SCORES="${RESULT_ROOT}/normal_classifier/valid_normal_scores.csv"
DICOM_ANNOTATION="${RESULT_ROOT}/dicom_film_probability/annotation.csv"
if [ ! -f "$CLASSIFIER_SCORES" ]; then
    echo "[ERROR] Normal classifier is not complete: $CLASSIFIER_SCORES"
    exit 2
fi
if [ ! -f "$DICOM_ANNOTATION" ]; then
    echo "[ERROR] DICOM full-validation inference is still incomplete: $DICOM_ANNOTATION"
    echo "[ERROR] Wait for the active DICOM inference to finish before Wave V7.1."
    exit 2
fi

echo "[WAVE V7.1] Matched five-condition fast screen"
echo "[WAVE V7.1] Vision probability: run/recompute on all six GPUs"
echo "[WAVE V7.1] DICOM probability: reuse completed artifact"
echo "[WAVE V7.1] beta=0.25,0.5,1,2,3,4,5 tau=0"

PROJECT_DIR="$PROJECT_DIR" \
GPU_IDS="$GPU_IDS" \
RESULT_ROOT="$RESULT_ROOT" \
VISION_CHECKPOINT="$VISION_CHECKPOINT" \
DICOM_FILM_CHECKPOINT="$DICOM_FILM_CHECKPOINT" \
TEXT_FEATURE_CACHE="$TEXT_FEATURE_CACHE" \
VISION_USE_EMA=0 DICOM_USE_EMA=0 \
SKIP_CLASSIFIER=1 \
SKIP_DICOM_INFERENCE=1 \
SKIP_VISION_INFERENCE="${SKIP_VISION_INFERENCE:-0}" \
BETAS="${BETAS:-0.25,0.5,1,2,3,4,5}" \
TEXT_THRESHOLDS="${TEXT_THRESHOLDS:-0}" \
FAST_SCREEN=1 \
bash "$BASE_LAUNCHER"
