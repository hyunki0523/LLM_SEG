#!/bin/bash
set -euo pipefail

# ==========================================
# 1. 공통 경로 및 옵션 설정
# ==========================================
# [중요] bitsandbytes와 RTX 6000 Ada(Sm89) 호환을 위한 라이브러리 지정
# 시스템 CUDA(/usr/local/cuda) 대신 PyTorch 내장 라이브러리를 우선하도록 설정합니다.
PY_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=$PY_SITE/nvidia/nvjitlink/lib:$PY_SITE/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}

python - <<'PY'
import sys
import torch

print(f"[CUDA CHECK] torch={torch.__version__}, torch_cuda={torch.version.cuda}, available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("[ERROR] CUDA is not available. Stop before falling back to CPU.")
PY

CSV_PATH="${CSV_PATH:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_test_set_test2.xlsx}"

PATCH_SIZE="32 224 224"
#PATCH_SIZE="16 224 224"
BASE_SAVE_ROOT="/mnt/nas125/forGPU2/lhyunki/llmseg/inference_result/FUdata/test0628"

# Qwen 모델의 NaN 방지를 위해 기본적으로 bf16 사용
MIXED_PRECISION_MODE="bf16"
MIXED_PRECISION="--mixed_precision ${MIXED_PRECISION_MODE}"

# 파일 저장을 생략하여 매우 빠른 평가만 확인하려면 "--no-save_pred", 개별 환자 3D 예측 마스크 파일을 전부 저장하려면 "--save_pred"
SAVE_OPT="--save_pred"

# 슬라이딩 윈도우 패치 배치 사이즈 (GPU 메모리에 따라 조절: 97GB -> sw_batch_size=16 이상 권장)
SW_BATCH="--sw_batch_size 16"
POSTPROC_OPT="--min_component_voxels 20"
CFG_OPT="${CFG_OPT:---cfg_scale 1}"

# ==========================================
# 2. 실행할 모델 리스트 (경로|컨텍스트_옵션)
# ==========================================
MODELS=(

    # "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/qwen/bs2_cfg_dicom_cc/model_epoch_300.pth|--context --no-include_emr --include_cc"
    # "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/qwen/bs2_context_lora/model_epoch_300.pth|--context"
    
    # "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/visionly/model_epoch_300.pth|--no-context"
    # "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/qwen/bs2_dicom/model_epoch_300.pth|--context --no-include_emr"

    "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/vision/bs2_cls005/model_epoch_300.pth|--no-context"
    

    #"/mnt/nas206/forGPU/lhyunki/NeuroCAD/llmseg_BGE_m3/experiments/llama2_dicom_only/model_epoch_300.pth|--context --no-include_emr"
    # 1. Qwen3 Context 모델 
    #"/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/softprompt/Qwen3/context_baseline/model_epoch_300.pth|--context"
    
    # 2. Vision Only 모델
    # "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/visiononly/model_epoch_300.pth|--no-context"
    #"/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/visiononly_pretrained/model_epoch_300.pth|--no-context"
    #"/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/visionly/model_epoch_300.pth|--no-context" 
    #
    
    # 3. Llama2 Softprompt 모델
    #"/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/softprompt/llama2/softprompt/batch2/model_epoch_300.pth|--context"
    #"/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/softprompt/llama2/softprompt/pretrained_bs1/model_epoch_300.pth|--context"
    #"/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/softprompt/llama2/softprompt/pretrained_bs2/model_epoch_300.pth|--context"
    #"/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/softprompt/llama2/softprompt/ema/model_epoch_300.pth|--context"

    # 3.1 Llama2 prefix 모델
    # "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/300ep_batch4_prepix/llama2/context/model_epoch_300.pth|--context"

    # "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/softprompt/llama/softprompt/ema/model_epoch_300.pth|--context"
    # "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/softprompt/llama/softprompt/ema/model_epoch_300.pth|--context"

    
    #"/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/softprompt/llama/context_pretrained_lr/tv_bs/model_epoch_300.pth|--context"     
    #"/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/softprompt/llama/context_pretrained_lr/model_epoch_300.pth|--context"  

    # 4. Qwen Softprompt 모델
    #"/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/softprompt/qwen/softprompt/model_epoch_300.pth|--context"
    #"/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/softprompt/qwen3/context_pretrained_lr/model_epoch_300.pth|--context"

    # 4.1 Qwen prefix 모델
        

    # 5. Bio-mistral 모델
    #"/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/softprompt/mistral/softprompt/model_epoch_300.pth|--context"

    # 6. Final parameter_ablation
    
    # "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/final/mistral/context/model_epoch_300.pth| --context"
    # "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/final/mistral/dicom/model_epoch_300.pth| --no-include_emr --context"
    #"/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/final/mistral/scratch/context/model_epoch_300.pth| --context"

    # "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/final/qwen/dicom/model_epoch_300.pth| --no-include_emr --context"
    # "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/final/qwen/context/model_epoch_300.pth| --context"    
    # "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/final/qwen/scratch_context/model_epoch_300.pth| --context"
    #"/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/visionly/use_ema_bs4/model_epoch_300.pth| --no-context"
    #"/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/final/visiononly/context_bs4/model_epoch_300.pth| --no-context" 
    #"/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/final/llama/context_bs4/model_epoch_300.pth| --context"
    
    # 7. FU dataset 
    # "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata/visionly/bs2/model_epoch_300.pth| --no-context"
    # "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata/llama/context_bs2/model_epoch_300.pth| --context"
    # "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata/gemma4/context_bs2/model_epoch_300.pth| --context"
    # "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata/llama/context_bs2_lora/model_epoch_300.pth| --context"
    # "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata/llama/context_bs2/model_epoch_300.pth| --context"
    # "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata/mistral/context_bs2/model_epoch_300.pth| --context"
    # "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata/qwen/context_bs2/model_epoch_300.pth| --context"
    # "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata/visionly/bs2/model_epoch_300.pth| --no-context"
    #"/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata/llama/bs2_cfg/model_epoch_300.pth| --context"
    # 8. Gemma 모델 (경로는 실제 사용하시는 모델 경로로 변경해 주세요)
    # "/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata/gemma/context_bs2/model_epoch_300.pth| --context"
)

# ==========================================
# 3. 루프 실행
# ==========================================
for item in "${MODELS[@]}"; do
    # 구분자('|')를 기준으로 모델 경로와 옵션을 분리
    MODEL_PATH="${item%%|*}"
    CONTEXT_FLAG="${item##*|}"
    
    # 폴더명을 뒤에서부터 자르기 (list[-3]와 list[-2] 추출)
    DIR_LEVEL_1=$(dirname "$MODEL_PATH")          # 예: .../Qwen3/context_baseline
    DIR_NAME_MINUS_2=$(basename "$DIR_LEVEL_1")   # 예: context_baseline
    
    DIR_LEVEL_2=$(dirname "$DIR_LEVEL_1")         # 예: .../Qwen3
    DIR_NAME_MINUS_3=$(basename "$DIR_LEVEL_2")   # 예: Qwen3

    DIR_LEVEL_3=$(dirname "$DIR_LEVEL_2")         # 예: .../Qwen3
    DIR_NAME_MINUS_4=$(basename "$DIR_LEVEL_3")   # 예: Qwen3
    
    # 최종 ARG 및 저장 위치 결합
    ARG="${DIR_NAME_MINUS_4}_${DIR_NAME_MINUS_3}_${DIR_NAME_MINUS_2}"
    SAVE_ROOT="${BASE_SAVE_ROOT}/${ARG}"

    # LLM_REPO_OPT 설정: 모델 경로(MODEL_PATH)에 따라 적절한 LLM 리포지토리를 지정합니다.
    LLM_REPO_OPT=""
    MODEL_PATH_LOWER="${MODEL_PATH,,}"
    if [[ "$MODEL_PATH" == *"gemma"* || "$MODEL_PATH" == *"Gemma"* ]]; then
        LLM_REPO_OPT="--llm_repo /mnt/nas252/forGPU2/HeeYeon/00_Resources/llm_weight/models--google--gemma-4-E4B-it/snapshots/83df0a889143b1dbfc61b591bbc639540fd9ce4c"
    elif [[ "$MODEL_PATH_LOWER" == *"qwen"* ]]; then
        LLM_REPO_OPT="--llm_repo /mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_seg_jhk/model_custom/qwen3/Qwen3-8B-Base"
    fi

    echo "=========================================================="
    echo "🚀 [START] Inference: $ARG"
    echo "▶ Model Path: $MODEL_PATH"
    echo "▶ Context   : $CONTEXT_FLAG"
    echo "▶ Save Root : $SAVE_ROOT"
    echo "=========================================================="
    
    # python 추론 스크립트 실행 (PCI bus 3D 에 해당하는 gpu index 1 만 사용)
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
        $MIXED_PRECISION \
        $SAVE_OPT \
        $SW_BATCH \
        $POSTPROC_OPT \
        $CFG_OPT \
        $LLM_REPO_OPT
        
    echo "✅ [DONE] Finished Inference for $ARG"
    echo ""
done

echo "🎉 모든 인퍼런스 작업이 완료되었습니다!"
