#!/bin/bash
set -euo pipefail

# Train modality-safe ablations with four 2-GPU jobs:
#   slot0 -> GPU 0,1
#   slot1 -> GPU 2,3
#   slot2 -> GPU 4,5
#   slot3 -> GPU 6,7
#
# LLM text is restricted to extracted_cc, chief_complaint, and an optional
# allow-list of DICOM acquisition fields. FiLM remains a separate path. Reports,
# refined EMR, labels, demographics and CFG are disabled.

PROJECT_DIR="${PROJECT_DIR:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk}"
PYTHON_EXE="${PYTHON_EXE:-python}"

if [ -f "${PROJECT_DIR}/wandb_key.txt" ]; then
    # Local credential file; excluded from Git.
    WANDB_API_KEY="$(tr -d '\r\n' < "${PROJECT_DIR}/wandb_key.txt")"
    export WANDB_API_KEY
fi

TRAIN_CSV="${TRAIN_CSV:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_train_set.xlsx}"
VALID_CSV="${VALID_CSV:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_valid_set.xlsx}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-/mnt/nas125/forGPU2/lhyunki/llmseg/experiments/fudata_final/llama/safe_film_context_v2}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_DIR}/train_logs/llama_safe_film_context_v2_$(date +%Y%m%d_%H%M%S)}"

# The hk copy currently contains only Llama config/tokenizer/index files.
# Reuse the complete read-only weight directory from jhk unless LLM_REPO is overridden.
LLM_REPO="${LLM_REPO:-/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_seg_jhk/model_custom/llama2/Llama-2-7b-chat-hf/}"
PRETRAINED="${PRETRAINED:-}"

PATCH_SIZE="${PATCH_SIZE:-32 224 224}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-1e-5}"
EPOCHS="${EPOCHS:-300}"
N_ITER_PER_EPOCH="${N_ITER_PER_EPOCH:-256}"
N_ITER_VALID="${N_ITER_VALID:-50}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-10}"
POSITIVE_PROB="${POSITIVE_PROB:-0.80}"
LOSS_FCT="${LOSS_FCT:-tversky}"
MIXED_PRECISION_MODE="${MIXED_PRECISION_MODE:-bf16}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
NUM_WORKERS="${NUM_WORKERS:-6}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
TEXT_FUSION_WARMUP_EPOCHS="${TEXT_FUSION_WARMUP_EPOCHS:-30}"
TEXT_FUSION_TRANSITION_EPOCHS="${TEXT_FUSION_TRANSITION_EPOCHS:-30}"
CONTEXT_INITIAL_ALPHA="${CONTEXT_INITIAL_ALPHA:-0.10}"
CONTEXT_MIN_ALPHA="${CONTEXT_MIN_ALPHA:-0.05}"
RAW_FUSED_SEG_WEIGHT="${RAW_FUSED_SEG_WEIGHT:-0.5}"
SOFT_PROMPT_MODE="${SOFT_PROMPT_MODE:-learned}"
TEXT_FEATURE_CACHE="${TEXT_FEATURE_CACHE:-}"
DICOM_PROMPT_MODE="${DICOM_PROMPT_MODE:-none}"
EXPERIMENT_NAME_SUFFIX="${EXPERIMENT_NAME_SUFFIX:-}"
STREAM_LOGS="${STREAM_LOGS:-0}"
DEEP_SUPERVISION="${DEEP_SUPERVISION:-0}"
DEEP_SUPERVISION_WEIGHTS="${DEEP_SUPERVISION_WEIGHTS:-1.0,0.3,0.1}"
USE_EMA="${USE_EMA:-1}"
EMA_DECAY="${EMA_DECAY:-0.999}"
CFG_SCALE=1
BASE_PORT="${BASE_PORT:-29600}"
CHECK_IMAGE_PATHS="${CHECK_IMAGE_PATHS:-1}"
CHECK_DATASET_CONTRACT="${CHECK_DATASET_CONTRACT:-1}"
SKIP_MISSING_IMAGE_PATHS="${SKIP_MISSING_IMAGE_PATHS:-1}"
DEFAULT_IMAGE_PATH_REWRITE_FROM="/mnt/nas100/Brain_ER/data/BrainCT_NIfTIv2"
DEFAULT_IMAGE_PATH_REWRITE_TO="/mnt/nas100/Brain_ER/IDs/kevin/BrainCT_NIfTIv2"
IMAGE_PATH_REWRITE_FROM="${IMAGE_PATH_REWRITE_FROM:-${LLMSEG_IMAGE_PATH_REWRITE_FROM:-$DEFAULT_IMAGE_PATH_REWRITE_FROM}}"
IMAGE_PATH_REWRITE_TO="${IMAGE_PATH_REWRITE_TO:-${LLMSEG_IMAGE_PATH_REWRITE_TO:-$DEFAULT_IMAGE_PATH_REWRITE_TO}}"

if [ -n "${GPU_PAIRS:-}" ]; then
    read -r -a GPU_PAIR_LIST <<< "$GPU_PAIRS"
else
    GPU_PAIR_LIST=("0,1" "2,3" "4,5" "6,7")
fi
MAX_PARALLEL="${MAX_PARALLEL:-4}"
NUM_PROCESSES_PER_JOB="${NUM_PROCESSES_PER_JOB:-2}"
RUN_EXTRA_MODES="${RUN_EXTRA_MODES:-0}"
OVERWRITE_TRAIN="${OVERWRITE_TRAIN:-0}"
ONLY_EXPERIMENTS="${ONLY_EXPERIMENTS:-}"
AUTO_RESUME="${AUTO_RESUME:-1}"

if ! [[ "$NUM_PROCESSES_PER_JOB" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] NUM_PROCESSES_PER_JOB must be a positive integer."
    exit 2
fi

CORE_EXPERIMENTS=(
    "vision_only|False|False|False"
    "dicom_film|True|False|False"
    "text_safe|False|True|False"
    "dicom_text_safe|True|True|False"
)

EXTRA_EXPERIMENTS=(
    "dicom_film_frozen|True|False|True"
    "dicom_text_safe_frozen|True|True|True"
)

EXPERIMENTS=("${CORE_EXPERIMENTS[@]}")
if [ "$RUN_EXTRA_MODES" = "1" ]; then
    EXPERIMENTS+=("${EXTRA_EXPERIMENTS[@]}")
fi
if [ -n "$ONLY_EXPERIMENTS" ]; then
    FILTERED_EXPERIMENTS=()
    for experiment in "${EXPERIMENTS[@]}"; do
        exp_name="${experiment%%|*}"
        if [[ ",$ONLY_EXPERIMENTS," == *",$exp_name,"* ]]; then
            FILTERED_EXPERIMENTS+=("$experiment")
        fi
    done
    EXPERIMENTS=("${FILTERED_EXPERIMENTS[@]}")
    if [ "${#EXPERIMENTS[@]}" -eq 0 ]; then
        echo "[ERROR] ONLY_EXPERIMENTS matched no configured experiments: $ONLY_EXPERIMENTS"
        exit 1
    fi
fi

NEEDS_LLM=0
for experiment in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r check_name check_dicom check_context check_freeze <<< "$experiment"
    if [ "$check_context" = "True" ] && [ -z "$TEXT_FEATURE_CACHE" ]; then
        NEEDS_LLM=1
        break
    fi
done
if [ "$NEEDS_LLM" = "1" ]; then
    if [ ! -f "${LLM_REPO%/}/config.json" ]; then
        echo "[ERROR] Llama config not found: ${LLM_REPO%/}/config.json"
        exit 1
    fi
    if ! compgen -G "${LLM_REPO%/}/model-*.safetensors" > /dev/null \
       && ! compgen -G "${LLM_REPO%/}/pytorch_model-*.bin" > /dev/null; then
        echo "[ERROR] Llama weight shards not found under: $LLM_REPO"
        exit 1
    fi
fi
if [ -n "$TEXT_FEATURE_CACHE" ]; then
    if [ "$SOFT_PROMPT_MODE" != "disabled" ]; then
        echo "[ERROR] TEXT_FEATURE_CACHE requires SOFT_PROMPT_MODE=disabled"
        exit 1
    fi
    if [ ! -f "$TEXT_FEATURE_CACHE" ]; then
        echo "[ERROR] Text feature cache not found: $TEXT_FEATURE_CACHE"
        exit 1
    fi
fi

mkdir -p "$LOG_ROOT" "$CHECKPOINT_BASE"
cd "$PROJECT_DIR"

PY_SITE=$($PYTHON_EXE -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$PY_SITE/nvidia/nvjitlink/lib:$PY_SITE/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export USE_TORCHAUDIO="${USE_TORCHAUDIO:-0}"
export TRANSFORMERS_NO_TORCHAUDIO="${TRANSFORMERS_NO_TORCHAUDIO:-1}"
export TRANSFORMERS_NO_AUDIO="${TRANSFORMERS_NO_AUDIO:-1}"
export LLMSEG_DISABLE_TORCHAUDIO="${LLMSEG_DISABLE_TORCHAUDIO:-1}"
# PyTorch SDPA backward intermittently raised cudaErrorIllegalAddress during
# sustained Llama-2 soft-prompt training on sm_120. Hugging Face eager
# attention is slower but stable; callers can explicitly override this after
# validating another backend on their installed Torch/CUDA stack.
export LLM_ATTN_IMPLEMENTATION="${LLM_ATTN_IMPLEMENTATION:-eager}"
# The grounded concept branch uses torch.nn.MultiheadAttention independently
# of the Hugging Face Llama attention implementation. Disable fused
# flash/memory-efficient SDPA there as well on the current Torch 2.12/cu132
# Blackwell stack.
export LLMSEG_FORCE_MATH_SDP="${LLMSEG_FORCE_MATH_SDP:-1}"
# Even the non-fused BF16 torch.bmm path in MultiheadAttention can fail with
# CUBLAS_STATUS_INTERNAL_ERROR on this cu132 Blackwell build. Limit FP32 to the
# small concept-query and spatial-to-concept attention operations.
export LLMSEG_FORCE_FP32_MHA="${LLMSEG_FORCE_FP32_MHA:-1}"
# This stack was stable with CUDA P2P and CUDA-memory transport disabled.
# Single-GPU v3a jobs are unaffected; multi-GPU callers may override after
# separately validating NCCL on their exact driver/runtime combination.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export TRAIN_CSV VALID_CSV
export LLMSEG_SKIP_MISSING_IMAGE_PATHS="$SKIP_MISSING_IMAGE_PATHS"
export LLMSEG_IMAGE_PATH_REWRITE_FROM="$IMAGE_PATH_REWRITE_FROM"
export LLMSEG_IMAGE_PATH_REWRITE_TO="$IMAGE_PATH_REWRITE_TO"

echo "[INFO] Checking required Python packages..."
if ! "$PYTHON_EXE" -c 'import nibabel; print(f"[CHECK] nibabel={nibabel.__version__}")'; then
    $PYTHON_EXE -m pip install --quiet nibabel || $PYTHON_EXE -m pip install nibabel
    "$PYTHON_EXE" -c 'import nibabel; print(f"[CHECK] nibabel={nibabel.__version__}")'
fi

if [ "$CHECK_DATASET_CONTRACT" = "1" ]; then
    echo "[INFO] Auditing multimodal data contract..."
    $PYTHON_EXE audit_dataset_contract.py \
        --train-csv "$TRAIN_CSV" \
        --valid-csv "$VALID_CSV" \
        --dicom-prompt-mode "$DICOM_PROMPT_MODE" \
        --output "$LOG_ROOT/dataset_contract_report.json"
fi

echo "[INFO] PROJECT_DIR=$PROJECT_DIR"
echo "[INFO] TRAIN_CSV=$TRAIN_CSV"
echo "[INFO] VALID_CSV=$VALID_CSV"
echo "[INFO] CHECKPOINT_BASE=$CHECKPOINT_BASE"
echo "[INFO] PRETRAINED=${PRETRAINED:-<none>}"
echo "[INFO] LOG_ROOT=$LOG_ROOT"
echo "[INFO] GPU_PAIRS=${GPU_PAIR_LIST[*]}"
echo "[INFO] MAX_PARALLEL=$MAX_PARALLEL"
echo "[INFO] NUM_PROCESSES_PER_JOB=$NUM_PROCESSES_PER_JOB"
echo "[INFO] LR=$LR"
echo "[INFO] WARMUP_EPOCHS=$WARMUP_EPOCHS"
echo "[INFO] RUN_EXTRA_MODES=$RUN_EXTRA_MODES"
echo "[INFO] ONLY_EXPERIMENTS=${ONLY_EXPERIMENTS:-<all>}"
echo "[INFO] AUTO_RESUME=$AUTO_RESUME"
echo "[INFO] TEXT_FUSION_WARMUP_EPOCHS=$TEXT_FUSION_WARMUP_EPOCHS"
echo "[INFO] TEXT_FUSION_TRANSITION_EPOCHS=$TEXT_FUSION_TRANSITION_EPOCHS"
echo "[INFO] CONTEXT_INITIAL_ALPHA=$CONTEXT_INITIAL_ALPHA"
echo "[INFO] CONTEXT_MIN_ALPHA=$CONTEXT_MIN_ALPHA"
echo "[INFO] RAW_FUSED_SEG_WEIGHT=$RAW_FUSED_SEG_WEIGHT"
echo "[INFO] USE_EMA=$USE_EMA"
echo "[INFO] EMA_DECAY=$EMA_DECAY"
echo "[INFO] SOFT_PROMPT_MODE=$SOFT_PROMPT_MODE"
echo "[INFO] DICOM_PROMPT_MODE=$DICOM_PROMPT_MODE"
echo "[INFO] TEXT_FEATURE_CACHE=${TEXT_FEATURE_CACHE:-<online>}"
echo "[INFO] EXPERIMENT_NAME_SUFFIX=${EXPERIMENT_NAME_SUFFIX:-<none>}"
echo "[INFO] DEEP_SUPERVISION=$DEEP_SUPERVISION"
echo "[INFO] DEEP_SUPERVISION_WEIGHTS=$DEEP_SUPERVISION_WEIGHTS"
echo "[INFO] LLM_ATTN_IMPLEMENTATION=$LLM_ATTN_IMPLEMENTATION"
echo "[INFO] LLMSEG_FORCE_MATH_SDP=$LLMSEG_FORCE_MATH_SDP"
echo "[INFO] LLMSEG_FORCE_FP32_MHA=$LLMSEG_FORCE_FP32_MHA"
echo "[INFO] NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"
echo "[INFO] NCCL_CUMEM_ENABLE=$NCCL_CUMEM_ENABLE"
echo "[INFO] CHECK_IMAGE_PATHS=$CHECK_IMAGE_PATHS"
echo "[INFO] CHECK_DATASET_CONTRACT=$CHECK_DATASET_CONTRACT"
echo "[INFO] SKIP_MISSING_IMAGE_PATHS=$SKIP_MISSING_IMAGE_PATHS"
if [ -n "$LLMSEG_IMAGE_PATH_REWRITE_FROM" ] || [ -n "$LLMSEG_IMAGE_PATH_REWRITE_TO" ]; then
    echo "[INFO] IMAGE_PATH_REWRITE_FROM=$LLMSEG_IMAGE_PATH_REWRITE_FROM"
    echo "[INFO] IMAGE_PATH_REWRITE_TO=$LLMSEG_IMAGE_PATH_REWRITE_TO"
fi

if [ "$CHECK_IMAGE_PATHS" = "1" ]; then
    echo "[INFO] Checking train/valid image paths before launching GPUs..."
    path_check_args=(
        --train-csv "$TRAIN_CSV"
        --valid-csv "$VALID_CSV"
        --rewrite-from "$LLMSEG_IMAGE_PATH_REWRITE_FROM"
        --rewrite-to "$LLMSEG_IMAGE_PATH_REWRITE_TO"
    )
    if [ "$SKIP_MISSING_IMAGE_PATHS" = "1" ]; then
        path_check_args+=(--skip-missing)
    fi
    "$PYTHON_EXE" check_image_paths.py "${path_check_args[@]}"
fi

nvidia-smi || true

PIDS=()

wait_batch() {
    local fail=0
    local pid
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            fail=1
        fi
    done
    PIDS=()

    if [ "$fail" -ne 0 ]; then
        echo "[ERROR] At least one training job failed. Check logs in: $LOG_ROOT"
        exit 1
    fi
}

run_one() {
    local exp_idx="$1"
    local exp_name="$2"
    local use_dicom="$3"
    local use_context="$4"
    local freeze_vision="$5"

    local effective_name="${exp_name}${EXPERIMENT_NAME_SUFFIX}"
    local context_args
    if [ "$use_context" = "True" ]; then
        context_args=(
            --context
            --llm_repo "$LLM_REPO"
            --soft_prompt_mode "$SOFT_PROMPT_MODE"
        )
        if [ -n "$TEXT_FEATURE_CACHE" ]; then
            context_args+=(--text_feature_cache "$TEXT_FEATURE_CACHE")
        fi
    else
        context_args=(--no-context)
    fi
    local dicom_args
    if [ "$use_dicom" = "True" ]; then
        dicom_args=(--use_dicom)
    else
        dicom_args=(--no-use_dicom)
    fi
    local freeze_args
    if [ "$freeze_vision" = "True" ]; then
        if [ -z "$PRETRAINED" ]; then
            echo "[ERROR] $exp_name requires PRETRAINED for a frozen-vision phase."
            return 1
        fi
        freeze_args=(--freeze_vision)
    else
        freeze_args=(--no-freeze_vision)
    fi

    local pretrained_args=()
    if [ -n "$PRETRAINED" ]; then
        pretrained_args=(--pretrained "$PRETRAINED")
    fi
    local deep_supervision_args
    if [ "$DEEP_SUPERVISION" = "1" ]; then
        deep_supervision_args=(--deep_supervision)
    else
        deep_supervision_args=(--no-deep_supervision)
    fi
    local ema_args
    if [ "$USE_EMA" = "1" ]; then
        ema_args=(--use_ema --ema_decay "$EMA_DECAY")
    else
        ema_args=(--no-use_ema --ema_decay "$EMA_DECAY")
    fi

    local slot=$((exp_idx % ${#GPU_PAIR_LIST[@]}))
    local gpu_pair="${GPU_PAIR_LIST[$slot]}"
    local port=$((BASE_PORT + exp_idx))
    local checkpoint_dir="${CHECKPOINT_BASE}/${effective_name}"
    local log_path="${LOG_ROOT}/${effective_name}.log"
    local resume_args=()

    # The first run has no experiment directory yet. Create it before the
    # auto-resume `find`; otherwise `set -euo pipefail` aborts before logging.
    mkdir -p "$checkpoint_dir"

    if [ -f "${checkpoint_dir}/final_model.pth" ] && [ "$OVERWRITE_TRAIN" != "1" ]; then
        echo "[SKIP] final_model.pth already exists: $checkpoint_dir"
        return 0
    fi
    if [ "$AUTO_RESUME" = "1" ] && [ "$OVERWRITE_TRAIN" != "1" ]; then
        local latest_resume
        latest_resume="$(
            find "$checkpoint_dir" -maxdepth 1 -type f -name 'model_epoch_*.pth' \
                -printf '%f\n' 2>/dev/null | sort -V | tail -n 1
        )"
        if [ -n "$latest_resume" ]; then
            resume_args=(--resume "${checkpoint_dir}/${latest_resume}")
        fi
    fi

    # Each run executes in its own background subshell. Redirect that subshell
    # either to the log only or through tee for a single foreground-style job.
    if [ "$STREAM_LOGS" = "1" ]; then
        exec > >(tee "$log_path") 2>&1
    else
        exec >"$log_path" 2>&1
    fi

    {
        echo "=========================================================="
        echo "[START] Train: $effective_name"
        echo "GPU pair          : $gpu_pair"
        echo "main_process_port : $port"
        echo "DICOM FiLM        : $use_dicom"
        echo "Safe text         : $use_context"
        echo "Soft prompt       : $SOFT_PROMPT_MODE"
        echo "DICOM text mode   : $DICOM_PROMPT_MODE"
        echo "Text cache        : ${TEXT_FEATURE_CACHE:-<online>}"
        echo "Freeze vision     : $freeze_vision"
        echo "Checkpoint        : $checkpoint_dir"
        echo "Resume            : ${resume_args[*]:-<none>}"
        echo "=========================================================="

        CUDA_VISIBLE_DEVICES="$gpu_pair" accelerate launch \
            --num_processes "$NUM_PROCESSES_PER_JOB" \
            --num_machines 1 \
            --mixed_precision "$MIXED_PRECISION_MODE" \
            --main_process_port "$port" \
            --dynamo_backend no \
            train.py \
            --mixed_precision "$MIXED_PRECISION_MODE" \
            --patch_size $PATCH_SIZE \
            --batch_size "$BATCH_SIZE" \
            --lr "$LR" \
            --grad_accum "$GRAD_ACCUM" \
            --num_workers "$NUM_WORKERS" \
            "${context_args[@]}" \
            "${dicom_args[@]}" \
            "${freeze_args[@]}" \
            "${pretrained_args[@]}" \
            "${resume_args[@]}" \
            "${deep_supervision_args[@]}" \
            "${ema_args[@]}" \
            --deep_supervision_weights "$DEEP_SUPERVISION_WEIGHTS" \
            --checkpoint_dir "$checkpoint_dir" \
            --experiment_name "$effective_name" \
            --train_csv "$TRAIN_CSV" \
            --valid_csv "$VALID_CSV" \
            --epochs "$EPOCHS" \
            --n_iter_per_epoch "$N_ITER_PER_EPOCH" \
            --n_iter_valid "$N_ITER_VALID" \
            --checkpoint_interval "$CHECKPOINT_INTERVAL" \
            --positive_prob "$POSITIVE_PROB" \
            --loss_fct "$LOSS_FCT" \
            --warmup_epochs "$WARMUP_EPOCHS" \
            --text_fusion_warmup_epochs "$TEXT_FUSION_WARMUP_EPOCHS" \
            --text_fusion_transition_epochs "$TEXT_FUSION_TRANSITION_EPOCHS" \
            --context_initial_alpha "$CONTEXT_INITIAL_ALPHA" \
            --context_min_alpha "$CONTEXT_MIN_ALPHA" \
            --raw_fused_seg_weight "$RAW_FUSED_SEG_WEIGHT" \
            --include_clinical False \
            --include_findings False \
            --include_cc "$use_context" \
            --include_chief_complaint "$use_context" \
            --include_emr False \
            --include_demographics False \
            --dicom_prompt_mode "$DICOM_PROMPT_MODE" \
            --cfg_scale "$CFG_SCALE"

        echo "[DONE] Train: $effective_name"
    }
}

for idx in "${!EXPERIMENTS[@]}"; do
    IFS='|' read -r exp_name use_dicom use_context freeze_vision <<< "${EXPERIMENTS[$idx]}"

    echo "=========================================================="
    effective_name="${exp_name}${EXPERIMENT_NAME_SUFFIX}"
    echo "[LAUNCH] $effective_name on GPU pair ${GPU_PAIR_LIST[$((idx % ${#GPU_PAIR_LIST[@]}))]}"
    echo "Log: ${LOG_ROOT}/${effective_name}.log"
    echo "=========================================================="

    run_one "$idx" "$exp_name" "$use_dicom" "$use_context" "$freeze_vision" &
    PIDS+=("$!")

    if [ "${#PIDS[@]}" -ge "$MAX_PARALLEL" ]; then
        wait_batch
    fi
done

if [ "${#PIDS[@]}" -gt 0 ]; then
    wait_batch
fi

echo "[DONE] DICOM training ablation completed. Logs: $LOG_ROOT"
