#!/bin/bash
set -euo pipefail

# Phase 3 for the new balanced checkpoints. This extracts paired FP16
# probabilities and classifier scores; calibration remains a separate CPU job.

PROJECT_DIR="${PROJECT_DIR:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5}"
VISION_CHECKPOINT="${VISION_CHECKPOINT:-/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/llama/vision_balanced_v1/vision_only_balanced_v1/best_sw_model.pth}"
DICOM_FILM_CHECKPOINT="${DICOM_FILM_CHECKPOINT:-/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/llama/dicom_film_balanced_v1/seed42/best_sw_model.pth}"
VISION_MANIFEST="${VISION_MANIFEST:-${PROJECT_DIR}/data_manifests/vision_balanced_v1.csv}"
RESULT_ROOT="${RESULT_ROOT:-/mnt/nas125/forGPU2/lhyunki/llmseg/inference_result/FUdata/v72_balanced_probability_$(date +%Y%m%d_%H%M%S)}"

for required in "$VISION_CHECKPOINT" "$DICOM_FILM_CHECKPOINT" "$VISION_MANIFEST"; do
    if [ ! -f "$required" ]; then
        echo "[ERROR] Required balanced-v1 artifact not found: $required"
        exit 2
    fi
done

PROJECT_DIR="$PROJECT_DIR" \
GPU_IDS="$GPU_IDS" \
VISION_CHECKPOINT="$VISION_CHECKPOINT" \
DICOM_FILM_CHECKPOINT="$DICOM_FILM_CHECKPOINT" \
VISION_MANIFEST="$VISION_MANIFEST" \
RESULT_ROOT="$RESULT_ROOT" \
VISION_USE_EMA=0 \
DICOM_USE_EMA=0 \
SKIP_MATRIX=1 \
bash "${PROJECT_DIR}/run_posthoc_suppression_v7_6gpu.sh"

echo "[NEXT] CPU calibration:"
echo "RESULT_ROOT='$RESULT_ROOT' bash '${PROJECT_DIR}/run_v72_calibration_locked.sh'"
