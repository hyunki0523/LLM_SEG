#!/bin/bash

# ==============================================================================
# Qwen3-8B-Base Ablation Study Execution Script (Tversky Loss, patch 32 224 224)
# ONLY the remaining 2 runs needed, as Scenario 2 and Scenario 4 are already done.
# ==============================================================================

# CUDA 12/13 library path safety for bitsandbytes
PY_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=$PY_SITE/nvidia/nvjitlink/lib:$PY_SITE/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

# --- 1. Vision Only (No Text Context) ---
echo "========================================="
echo "🚀 [START] Ablation 1/2: Vision Only"
echo "========================================="
accelerate launch --num_processes 2 --mixed_precision bf16 --dynamo_backend no train.py \
    --mixed_precision bf16 \
    --patch_size 32 224 224 \
    --batch_size 2 \
    --no-context \
    --checkpoint_dir "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/qwen/bs2_cfg_visiononly/" \
    --experiment_name "vision_only" \
    --train_csv "/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_train_set.xlsx" \
    --valid_csv "/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_valid_set.xlsx" \
    --epochs 300 \
    --positive_prob 0.80 \
    --loss_fct tversky

echo "✅ [DONE] Vision Only Experiment finished."
echo "-----------------------------------------"

# --- 2. DICOM + CC Only (Decoupled CC, No EMR history) ---
echo "========================================="
echo "🚀 [START] Ablation 2/2: DICOM + CC Only"
echo "========================================="
accelerate launch --num_processes 2 --mixed_precision bf16 --dynamo_backend no train.py \
    --mixed_precision bf16 \
    --patch_size 32 224 224 \
    --batch_size 2 \
    --llm_repo "/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_seg_jhk/model_custom/qwen3/Qwen3-8B-Base/" \
    --checkpoint_dir "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/qwen/bs2_cfg_dicom_cc/" \
    --experiment_name "dicom_cc" \
    --train_csv "/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_train_set.xlsx" \
    --valid_csv "/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_valid_set.xlsx" \
    --epochs 300 \
    --positive_prob 0.80 \
    --loss_fct tversky \
    --include_cc True \
    --include_emr False

echo "✅ [DONE] DICOM + CC Only Experiment finished."
echo "========================================="
echo "🎉 All remaining ablation experiments completed!"
