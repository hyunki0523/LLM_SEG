#!/bin/bash
set -euo pipefail

# CFG scale 2 sweep for the strongest context checkpoints.

PY_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$PY_SITE/nvidia/nvjitlink/lib:$PY_SITE/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

python - <<'PY'
import sys
import torch

print(f"[CUDA CHECK] torch={torch.__version__}, torch_cuda={torch.version.cuda}, available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("[ERROR] CUDA is not available. Stop before falling back to CPU.")
PY

CSV_PATH="${CSV_PATH:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_test_set_test2_hy.xlsx}"
PATCH_SIZE="32 224 224"
BASE_SAVE_ROOT="${BASE_SAVE_ROOT:-/mnt/nas125/forGPU2/lhyunki/llmseg/inference_result/FUdata/test0630_cfg2}"

MIXED_PRECISION_MODE="bf16"
SAVE_OPT="${SAVE_OPT:---save_pred}"
SW_BATCH="${SW_BATCH:---sw_batch_size 16}"
POSTPROC_OPT="${POSTPROC_OPT:---min_component_voxels 20}"
CFG_OPT="--cfg_scale 2"
OVERWRITE_OPT="${OVERWRITE_OPT:-}"

MODELS=(
    "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/qwen/bs2_dicom/model_epoch_300.pth|--context --no-include_emr"
    "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/llama/bs2_cfg/model_epoch_300.pth|--context"
    "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/qwen/bs2_cfg/model_epoch_300.pth|--context"
    "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/llama/bs4_cfg/model_epoch_300.pth|--context"
)

for item in "${MODELS[@]}"; do
    MODEL_PATH="${item%%|*}"
    CONTEXT_FLAG="${item##*|}"

    DIR_LEVEL_1=$(dirname "$MODEL_PATH")
    DIR_NAME_MINUS_2=$(basename "$DIR_LEVEL_1")
    DIR_LEVEL_2=$(dirname "$DIR_LEVEL_1")
    DIR_NAME_MINUS_3=$(basename "$DIR_LEVEL_2")
    DIR_LEVEL_3=$(dirname "$DIR_LEVEL_2")
    DIR_NAME_MINUS_4=$(basename "$DIR_LEVEL_3")

    ARG="${DIR_NAME_MINUS_4}_${DIR_NAME_MINUS_3}_${DIR_NAME_MINUS_2}_cfg2"
    SAVE_ROOT="${BASE_SAVE_ROOT}/${ARG}"

    LLM_REPO_OPT=""
    MODEL_PATH_LOWER="${MODEL_PATH,,}"
    if [[ "$MODEL_PATH_LOWER" == *"qwen"* ]]; then
        LLM_REPO_OPT="--llm_repo /mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_seg_jhk/model_custom/qwen3/Qwen3-8B-Base"
    fi

    echo "=========================================================="
    echo "[START] Inference: $ARG"
    echo "Model Path : $MODEL_PATH"
    echo "Context    : $CONTEXT_FLAG"
    echo "CFG        : $CFG_OPT"
    echo "CSV        : $CSV_PATH"
    echo "Save Root  : $SAVE_ROOT"
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
        $CONTEXT_FLAG \
        --mixed_precision "$MIXED_PRECISION_MODE" \
        $SAVE_OPT \
        $SW_BATCH \
        $POSTPROC_OPT \
        $CFG_OPT \
        $OVERWRITE_OPT \
        $LLM_REPO_OPT

    echo "[DONE] Finished Inference for $ARG"
    echo ""
done

echo "[DONE] All CFG2 inference jobs completed."
