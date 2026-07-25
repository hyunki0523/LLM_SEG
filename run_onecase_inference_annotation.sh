#!/bin/bash
set -euo pipefail

PY_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=$PY_SITE/nvidia/nvjitlink/lib:$PY_SITE/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}

python - <<'PY'
import sys
import torch

print(f"[CUDA CHECK] torch={torch.__version__}, torch_cuda={torch.version.cuda}, available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("[ERROR] CUDA is not available. Stop before falling back to CPU.")
PY

MODEL_PATH="${MODEL_PATH:-/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/vision/bs2_cls005/model_epoch_300.pth}"
CSV_PATH="${CSV_PATH:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_test_set_test2_hy_multilabel_onecase.xlsx}"
SAVE_ROOT="${SAVE_ROOT:-/mnt/nas125/forGPU2/lhyunki/llmseg/inference_result/FUdata/test0630/onecase_annotation_smoke}"

echo "=========================================================="
echo "[START] One-case inference annotation smoke"
echo "Model   : $MODEL_PATH"
echo "CSV     : $CSV_PATH"
echo "SaveRoot: $SAVE_ROOT"
echo "=========================================================="

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    inference_eval.py \
    --model_path "$MODEL_PATH" \
    --csv_path "$CSV_PATH" \
    --save_root "$SAVE_ROOT" \
    --patch_size 32 224 224 \
    --no-context \
    --mixed_precision bf16 \
    --save_pred \
    --overwrite_pred \
    --sw_batch_size 16 \
    --min_component_voxels 20 \
    --save_annotation_csv \
    --annotation_csv_name annotation.csv

echo "=========================================================="
echo "[DONE] Outputs:"
echo "  $SAVE_ROOT/CT001188/pred_hemo_bin.nii.gz"
echo "  $SAVE_ROOT/annotation.csv"
echo "=========================================================="
