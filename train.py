import os, json, argparse, sys, site

# [BUGFIX] bitsandbytes CUDA 12/13 library (libnvJitLink.so) 오류 방지용 동적 경로 주입
try:
    for sp in site.getsitepackages():
        nv_jit_path = os.path.join(sp, 'nvidia', 'nvjitlink', 'lib')
        if os.path.isdir(nv_jit_path):
            current_ld = os.environ.get('LD_LIBRARY_PATH', '')
            if nv_jit_path not in current_ld:
                os.environ['LD_LIBRARY_PATH'] = nv_jit_path + (':' + current_ld if current_ld else '')
except Exception:
    pass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from monai.losses import DiceCELoss, DiceFocalLoss, TverskyLoss, GeneralizedDiceFocalLoss
from accelerate import Accelerator
from accelerate.utils import gather_object
from accelerate.utils import DistributedDataParallelKwargs
from accelerate import InitProcessGroupKwargs, DataLoaderConfiguration
from datetime import timedelta
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from data.dataset import get_data_loaders
from model_custom.stunet import get_stunet_base

import warnings
import wandb
# fft_conv_pytorch에서 발생하는 특정 경고 메시지를 무시하도록 설정
warnings.filterwarnings("ignore", message="Using a non-tuple sequence for multidimensional indexing is deprecated")

#import pywt, ptwt  # [WAVELET]

if not os.environ.get("WANDB_API_KEY"):
    warnings.warn(
        "WANDB_API_KEY is not set. Authenticate with `wandb login` or export "
        "the key before training if experiment logging is required."
    )

def trainable_state_dict(model, include_buffers=True):
    # Keep the complete segmentation/conditioning model even during a
    # frozen-vision phase. Exclude only the frozen, reloadable LLM backbone.
    llm_prefixes = ("text_encoder.transformer.", "text_encoder.token_embedding.")
    trainable_param_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad or not name.startswith(llm_prefixes)
    }

    buffer_names = set()
    if include_buffers:
        module_prefixes = {n.rsplit('.', 1)[0] for n in trainable_param_names if '.' in n}
        module_prefixes |= {n for n in trainable_param_names if '.' not in n}
        for n, _ in model.named_buffers():
            if n.startswith(llm_prefixes):
                continue
            if any(n == pref or n.startswith(pref + '.') for pref in module_prefixes):
                buffer_names.add(n)

    keep_names = trainable_param_names | buffer_names
    full = model.state_dict()
    return {k: v for k, v in full.items() if k in keep_names}


def rater_type(x):
    if x == "all":
        return x
    try:
        val = int(x)
    except ValueError:
        raise argparse.ArgumentTypeError("Rater must be 1–4 or 'all'")
    if val not in [1, 2, 3, 4]:
        raise argparse.ArgumentTypeError("Rater must be 1–4 or 'all'")
    return val


def cycle(dl):
    while True:
        for data in dl:
            yield data


def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =========================
# [WAVELET] 3D Wavelet 클래스
# =========================
class WaveletTransform3D:
    """
    3D wavelet (level=1) wrapper.
    - 입력:  (B, C, D, H, W)
    - 출력:  (B, C*8, D/2, H/2, W/2)  (LL + 7 detail)
    - reconstruction 시:
        (B, 8, D', H', W') → (B, 1, D, H, W)  (binary mask 1채널 가정)
    """
    def __init__(self, wavelet: str = "haar", level: int = 1, mode: str = "zero"):
        self.wavelet = pywt.Wavelet(wavelet)
        self.level = level
        self.mode = mode
        self.subband_keys = ["aad", "ada", "add", "daa", "dad", "dda", "ddd"]

    def decomposition(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, D, H, W) → (B, C*8, D', H', W')
        """
        coeffs3 = ptwt.wavedec3(x, self.wavelet, level=self.level, mode=self.mode)
        LL = coeffs3[0]         # (B, C, D', H', W')
        detail_dict = coeffs3[1]

        bands = [LL]
        for k in self.subband_keys:
            bands.append(detail_dict[k])   # (B, C, D', H', W')
        out = torch.cat(bands, dim=1)      # (B, C*8, D', H', W')
        return out

    def reconstruction(self, x_wt: torch.Tensor) -> torch.Tensor:
        """
        x_wt : (B, 8, D', H', W') → (B, 1, D, H, W)
        """
        B, C, Dp, Hp, Wp = x_wt.shape
        assert C == 1 + len(self.subband_keys), \
            f"reconstruction 용 출력 채널은 8이어야 합니다. (지금 C={C})"

        LL = x_wt[:, 0:1]
        details = {
            k: x_wt[:, i+1:i+2]
            for i, k in enumerate(self.subband_keys)
        }
        coeffs3 = (LL, details)
        out = ptwt.waverec3(coeffs3, self.wavelet)    # (B,1,D,H,W)
        return out


def get_args_parser():
    parser = argparse.ArgumentParser('LLM-Seg-3D', add_help=False)

    # Training parameters
    parser.add_argument('--epochs', default=600, type=int)
    parser.add_argument('--n_iter_per_epoch', default=256, type=int)
    parser.add_argument('--n_iter_valid', default=50, type=int)
    parser.add_argument('--val_interval', default=1, type=int)
    parser.add_argument('--mixed_precision', default='no', type=str)
    parser.add_argument('--loss_fct', default='gdfl', type=str, choices=['dice', 'bce', 'dice_focal', 'tversky', 'gdfl'])
    parser.add_argument('--cls_loss_weight', default=0.0, type=float,
                        help='Auxiliary lesion-presence classification BCE loss weight. 0 disables it.')
    parser.add_argument('--cls_pos_weight', default=None, type=float,
                        help='Optional positive weight for auxiliary classification BCE.')
    parser.add_argument('--safe_context_training', default=True,
                        action=argparse.BooleanOptionalAction,
                        help='Return vision/raw/fused logits and apply uncertainty-safe text fusion.')
    parser.add_argument('--vision_seg_weight', default=0.5, type=float)
    parser.add_argument('--null_consistency_weight', default=0.1, type=float)
    parser.add_argument('--context_residual_weight', default=0.01, type=float)
    parser.add_argument('--compatibility_weight', default=0.1, type=float)
    parser.add_argument('--text_burden_weight', default=0.1, type=float)
    parser.add_argument('--text_trauma_weight', default=0.05, type=float)
    parser.add_argument('--compatibility_margin', default=0.2, type=float)
    parser.add_argument('--text_fusion_warmup_epochs', default=5, type=int,
                        help='Force text residual strength during the initial grounding phase.')
    parser.add_argument('--context_patch_mask_probability', default=0.15, type=float,
                        help='Coarse vision-patch masking probability for the text-conditioned pass.')
    parser.add_argument('--context_hard_bypass_threshold', default=0.5, type=float,
                        help='At evaluation, fall back to vision below this compatibility.')
    parser.add_argument('--warmup_epochs', default=10, type=int, help='Number of epochs for learning rate warmup')
    parser.add_argument('--context', default=True, action=argparse.BooleanOptionalAction, help='Use context in the model (use --no-context to disable)')
    parser.add_argument('--use_dicom', default=False, action=argparse.BooleanOptionalAction,
                        help='Condition vision features with the separate DICOM FiLM branch.')
    parser.add_argument('--freeze_vision', default=False, action=argparse.BooleanOptionalAction,
                        help='Train conditioning branches only; use a later run for joint fine-tuning.')
    parser.add_argument('--use_ema', action=argparse.BooleanOptionalAction, default=False, help='Use EMA for model weights (use --no-use_ema to disable)')
    parser.add_argument('--ema_decay', default=0.999, type=float, help='EMA decay factor')
    parser.add_argument('--grad_accum', default=8, type=int, help='Gradient accumulation steps for effective batch size')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='AdamW weight decay')
    
    # Dataset parameters
    parser.add_argument('--channels', default=3, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--positive_prob', default=0.8, type=float)
    parser.add_argument('--rater', default=1, type=rater_type)
    parser.add_argument('--patch_size', default=(16, 256, 256), type=int, nargs=3)
    parser.add_argument('--train_csv', default="/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/ICH_pair/train_split.csv", type=str)
    parser.add_argument('--valid_csv', default="/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/ICH_pair/valid_split.csv", type=str)


    # [WAVELET] 옵션
    parser.add_argument('--use_wavelet', default=False, type=str2bool,
                        help='Apply 3D wavelet transform (wavedec3) to image.')
    parser.add_argument('--wavelet_name', default='haar', type=str)
    parser.add_argument('--wavelet_level', default=1, type=int)
    parser.add_argument('--wavelet_mode', default='zero', type=str)

    # Model parameters
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--num_classes', default=1, type=int)

   # LLM backbone repo (HF)
    parser.add_argument('--llm_repo', default="Qwen/Qwen3-8B-Base", type=str, help='HF repo id for LLM backbone (e.g., Qwen/Qwen3-8B-Base, meta-llama/Llama-2-7b-hf, ...)')
    parser.add_argument('--use_lora', action=argparse.BooleanOptionalAction, default=False, help='Use PEFT LoRA for LLM backbone')



    # Continue Training (Pretrain weight init)
    parser.add_argument('--pretrained', default=None)

    # Resume Training (full state: model + optimizer + scheduler + epoch)
    parser.add_argument('--resume', default=None, help='Path to checkpoint to RESUME training from (restores model, optimizer, scheduler, epoch)')

    # Save setting
    parser.add_argument('--checkpoint_dir', default='', help='path where to save checkpoint or output')
    parser.add_argument('--checkpoint_interval', default=10, type=int,
                        help='Save a resumable checkpoint every N epochs.')
    parser.add_argument('--experiment_name', default='default')

    # prompt parameter
    parser.add_argument('--include_clinical', default=False, type=str2bool,
                        help='Deprecated and ignored: report body is prohibited.')
    parser.add_argument('--include_findings', default=False, type=str2bool,
                        help='Deprecated and ignored: report conclusion is prohibited.')
    parser.add_argument('--include_cc', default=True, type=str2bool,
                        help='Include extracted_cc in the safe clinical prompt.')
    parser.add_argument('--include_chief_complaint', default=True, type=str2bool,
                        help='Include chief_complaint in the safe clinical prompt.')
    parser.add_argument('--include_emr', default=False, type=str2bool,
                        help='Deprecated and ignored: refined/full EMR is prohibited.')
    parser.add_argument('--dicom_prompt_mode', default='none', type=str,
                        choices=['full', 'extended', 'limited', 'geometry', 'kernel_only',
                                 'spacing_only', 'scanner_only', 'protocol_only', 'none'],
                        help='DICOM prompt field subset for context ablations')
    parser.add_argument('--include_demographics', default=False, type=str2bool,
                        help='Deprecated and ignored by the safe clinical prompt.')
    parser.add_argument('--cfg_scale', default=1.0, type=float,
                        help='Deprecated compatibility option; grounded fusion always uses 1.0.')

    return parser


def make_context_tokens_batch(tokenizer, max_length, context_length, contexts, device):
    """
    Return padded token ids [B, L] for TextContextEncoder.
    IMPORTANT:
      - Padding must match tokenizer.pad_token_id (NOT hard-coded 0).
      - Add special tokens so text[0] can be BOS when the tokenizer provides it.
     - Keep room for learnable context prefix (context_length).
    """
    # reserve slots for learnable context tokens inside TextContextEncoder
    L = max(2, int(max_length) - int(context_length))  # ensure >=2 so [:,1:] exists

    # HF tokenizer handles truncation/padding consistently across LLaMA/Qwen
    enc = tokenizer(
        list(ctx or "" for ctx in contexts),
        add_special_tokens=True,
        truncation=True,
        padding="max_length",
        max_length=L,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    return input_ids, attention_mask


def main(args):
    seed_everything()

    # 8-bit 양자화(BitsAndBytes) 모델의 커스텀 Autograd 훅은 static_graph=True와 충돌하여 내부 C++ 단언(Assert) 에러를 냅니다.
    # 하지만 이제 쓰지 않는 파라미터들(seg_outputs)을 앞에서 미리 동결(requires_grad=False)해두었으므로,
    # find_unused_parameters=False를 설정해도 완벽히 동작하며 메모리 누수도 잡을 수 있습니다!
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    ipg_handler = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
    data_config = DataLoaderConfiguration(split_batches=False)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with='wandb',
        kwargs_handlers=[kwargs, ipg_handler],
        dataloader_config=data_config,
        gradient_accumulation_steps=args.grad_accum
    )

    # === WandB 실험 이름 및 Config 자동 생성 ===
    eff_batch = args.batch_size * args.grad_accum * accelerator.num_processes
    run_name = f"{'Context' if args.context else 'Vision'}_BS{eff_batch}_P{args.patch_size[0]}"
    if getattr(args, 'experiment_name', 'default') != 'default':
        run_name += f"_{args.experiment_name}"
    if args.use_ema:
        run_name += "_EMA"
    if not getattr(args, 'include_emr', True):
        run_name += "_DICOMonly"
    
    my_custom_config = {
        "learning_rate": getattr(args, 'lr', 'unknown'),
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "accumulated_batch_size": eff_batch, 
        "num_epochs": args.epochs,
        "dataset_version": str(args.train_csv).split('/')[-1] if hasattr(args, 'train_csv') else "unknown",
        "patch_size": list(args.patch_size) if hasattr(args, 'patch_size') else [16, 256, 256],
        "mixed_precision": args.mixed_precision,
        "use_llm": args.context,
        "use_dicom_film": args.use_dicom,
        "loss_function": args.loss_fct,
        "cls_loss_weight": args.cls_loss_weight,
        "cls_pos_weight": args.cls_pos_weight,
        "use_ema": args.use_ema,
        "experiment_name": args.experiment_name,
        "include_cc": getattr(args, 'include_cc', True),
        "include_chief_complaint": getattr(args, 'include_chief_complaint', True),
        "include_emr": getattr(args, 'include_emr', True),
        "include_demographics": getattr(args, 'include_demographics', True),
        "dicom_prompt_mode": getattr(args, 'dicom_prompt_mode', 'full'),
        "memo": "자동생성된 실험 설정 로그"
    }

    accelerator.init_trackers(
        project_name="LLM-SEG-Patch",
        config=my_custom_config,
        init_kwargs={"wandb": {
            "name": run_name,
            "settings": wandb.Settings(init_timeout=300)
        }}
    )
    accelerator.print(f"[ARGS] context={args.context} llm_repo={args.llm_repo}")

    # =========================
    # [WAVELET] 설정 & 입력 채널 계산
    # =========================
    wavelet = None
    in_channels = args.channels
    NUM_WAVELET_BANDS = 1  # 기본값 (wavelet 사용 안 할 때)

    if getattr(args, "use_wavelet", False):
        wavelet = WaveletTransform3D(
            wavelet=args.wavelet_name,
            level=args.wavelet_level,
            mode=args.wavelet_mode,
        )
        dummy = torch.zeros(
            1, args.channels,
            args.patch_size[0],
            args.patch_size[1],
            args.patch_size[2],
        )
        with torch.no_grad():
            wt_out = wavelet.decomposition(dummy)
        in_channels = wt_out.shape[1]
        NUM_WAVELET_BANDS = 1 + len(wavelet.subband_keys)  # = 8
        accelerator.print(f"[Wavelet] enabled: {args.channels}ch -> {in_channels}ch after 3D DWT, "
                          f"output wavelet bands = {NUM_WAVELET_BANDS}")
    else:
        accelerator.print("[Wavelet] disabled")

    # === DataLoaders ===
    train_dl, valid_dl, data_metadata = get_data_loaders(
        train_csv=args.train_csv,
        valid_csv=args.valid_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        positive_prob=args.positive_prob,
        rater=args.rater,
        patch_size=tuple(args.patch_size),
        include_clinical=args.include_clinical,
        include_findings=args.include_findings,
        include_cc=getattr(args, 'include_cc', True),
        include_chief_complaint=getattr(args, 'include_chief_complaint', True),
        include_emr=getattr(args, 'include_emr', True),
        dicom_prompt_mode=getattr(args, 'dicom_prompt_mode', 'full'),
        include_demographics=getattr(args, 'include_demographics', True),
        use_dicom=getattr(args, 'use_dicom', False),
        return_metadata=True,
    )

    # === Model / Optimizer / Scheduler ===
    if args.loss_fct == 'bce':
        # wavelet 사용 시: 모델은 wavelet coeff 8채널을 출력
        out_ch = NUM_WAVELET_BANDS if wavelet is not None else 1
    else:
        out_ch = args.num_classes + 1
    llm_repo = args.llm_repo if args.context else None
    use_cls_loss = float(getattr(args, "cls_loss_weight", 0.0) or 0.0) > 0.0
    dicom_schema = data_metadata.get("dicom_schema")
    dicom_category_sizes = (
        tuple(
            len(dicom_schema["categories"][column])
            for column in dicom_schema["categorical_columns"]
        )
        if dicom_schema is not None
        else (2, 2)
    )
    if accelerator.is_main_process and dicom_schema is not None:
        with open(
            os.path.join(args.checkpoint_dir, "dicom_schema.json"),
            "w",
            encoding="utf-8",
        ) as schema_file:
            json.dump(dicom_schema, schema_file, ensure_ascii=False, indent=2)

    model = get_stunet_base(
        num_input_channels=in_channels,
        num_classes=out_ch,
        enable_deep_supervision=False,
        context=args.context,
        llm_repo=llm_repo,
        use_lora=getattr(args, 'use_lora', False),
        use_cls_head=use_cls_loss,
        use_dicom=getattr(args, 'use_dicom', False),
        dicom_numeric_dim=10,
        dicom_category_sizes=dicom_category_sizes,
     )
    accelerator.print('parameters(trainable):', count_params(model))

    if args.pretrained is not None and os.path.exists(args.pretrained):
        accelerator.print(f'Loading pretrained weights from: {args.pretrained}')
        checkpoint = torch.load(args.pretrained, map_location='cpu', weights_only=False)

        if 'model' in checkpoint:
            sd = checkpoint['model']
        else:
            sd = checkpoint

        cleaned_sd = {}
        for k, v in sd.items():
            if k.startswith('module.'):
                cleaned_sd[k[7:]] = v
            else:
                cleaned_sd[k] = v

        # [Pretrain Loading] Vision (STUNet) pretrain weight만 가져오고, 
        # LLM 파트(text_encoder, txt2vis, attn_transformer 등)의 shape mismatch나 missing key는 무시하도록 strict=False 설정.
        missing_keys, unexpected_keys = model.load_state_dict(cleaned_sd, strict=False)
        accelerator.print(f'[Pretrain] Loaded vision keys. Missing keys ignored: {len(missing_keys)}')
        
    # Freeze the LLM before constructing AdamW. Adding the full 8B backbone
    # and freezing it afterwards made distributed optimizer preparation
    # process billions of parameters that never receive gradients.
    if args.context and hasattr(model, "text_encoder") and model.text_encoder is not None:
        if getattr(args, "use_lora", False):
            if accelerator.is_local_main_process:
                print("[INFO] LoRA active: keeping PEFT-marked LLM adapters trainable.")
        else:
            for p in model.text_encoder.parameters():
                p.requires_grad_(False)
        model.text_encoder.eval()

    if not model.decoder.deep_supervision:
        for output_head in list(model.seg_outputs)[:-1]:
            for parameter in output_head.parameters():
                parameter.requires_grad_(False)

    if args.freeze_vision:
        vision_modules = [
            model.conv_blocks_context,
            model.upsample_layers,
            model.conv_blocks_localization,
            model.seg_outputs,
            model.cls_head,
        ]
        for module in vision_modules:
            if module is None:
                continue
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        accelerator.print(
            "[TRAINING PHASE] vision backbone/decoder frozen; conditioning branches only."
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    accelerator.print(
        "parameters(trainable, optimizer):",
        sum(p.numel() for p in trainable_params),
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # Gradient accumulation 적용 시 실제 optimizer step 수 계산 (1에폭당 업데이트 횟수)
    # n_iter_per_epoch // grad_accum 만큼 scheduler.step()이 일어납니다.
    steps_per_epoch = max(1, args.n_iter_per_epoch // args.grad_accum)
    warmup_steps = steps_per_epoch * args.warmup_epochs
    main_steps = steps_per_epoch * (args.epochs - args.warmup_epochs)

    # 1) Warmup: lr이 0 근처에서 1.0(args.lr)까지 서서히 증가
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    #2) Main: Cosine Annealing
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, main_steps), eta_min=args.lr * 0.1
    )

    # main_scheduler = torch.optim.lr_scheduler.LinearLR(
    #     optimizer, start_factor=1.0, end_factor=0.1, total_iters=main_steps
    # )
    
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps]
    )

    # === [메모리 최적화 (팁 2): 안 쓰는 파라미터 강제 동결] ===
    # deep_supervision이 꺼져있을 때 버려지는 중간 seg_outputs 레이어들을 미리 얼려둡니다.
    if hasattr(model, "seg_outputs"):
        is_deep_sup = False
        if hasattr(model, "decoder") and hasattr(model.decoder, "deep_supervision"):
            is_deep_sup = model.decoder.deep_supervision
        elif hasattr(model, "deep_supervision"):
            is_deep_sup = model.deep_supervision
            
        if not is_deep_sup:
            if accelerator.is_local_main_process:
                print("[INFO] deep_supervision=False 구조 감지됨. 메모리 확보를 위해 중간 seg_outputs 파라미터를 동결합니다.")
            for u in range(len(model.seg_outputs) - 1):
                for p in model.seg_outputs[u].parameters():
                    p.requires_grad = False

    # [중요] text_encoder 및 STUNet 버려지는 어텐션 모듈 동결
    # [UNFREEZED] <SEG> 토큰을 이용한 채널 Gating 아키텍처 개편으로 인해,
    # TwoWayTransformer의 최종 queries가 손실되지 않고 loss 경로에 정상 참여합니다!
    # 따라서, DDP did_not_receive_grad 에러 없이 정상적으로 End-to-End 학습이 가능하므로 unfreeze 합니다.
    # if hasattr(model, "attn_transformer"):
    #     for attn_block in model.attn_transformer:
    #         if hasattr(attn_block, "final_attn_token_to_image"):
    #             for p in attn_block.final_attn_token_to_image.parameters():
    #                 p.requires_grad = False
    #         if hasattr(attn_block, "norm_final_attn"):
    #             for p in attn_block.norm_final_attn.parameters():
    #                 p.requires_grad = False

    if args.context and hasattr(model, "text_encoder") and model.text_encoder is not None:
        if getattr(args, 'use_lora', False):
            if accelerator.is_local_main_process:
                print("[INFO] LoRA가 활성화되어 텍스트 인코더 전체 강제 동결을 건너뜁니다.")
        else:
            for p in model.text_encoder.parameters():
                p.requires_grad_(False)
        model.text_encoder.eval()

    model, optimizer, scheduler, train_dl, valid_dl = accelerator.prepare(
        model, optimizer, scheduler, train_dl, valid_dl
    )

    # === [RESUME] Full state restoration (must be after accelerator.prepare) ===
    if args.resume is not None and os.path.exists(args.resume):
        accelerator.print(f'[RESUME] Loading full checkpoint from: {args.resume}')
        resume_ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)

        # --- Model weights ---
        if 'model' in resume_ckpt:
            raw_sd = resume_ckpt['model']
            cleaned_sd = {k[7:] if k.startswith('module.') else k: v for k, v in raw_sd.items()}
            missing, unexpected = accelerator.unwrap_model(model).load_state_dict(cleaned_sd, strict=False)
            accelerator.print(f'[RESUME] Model weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}')

        # --- Optimizer state ---
        optimizer_restored = False
        if 'optimizer' in resume_ckpt:
            try:
                optimizer.load_state_dict(resume_ckpt['optimizer'])
                optimizer_restored = True
                accelerator.print('[RESUME] Optimizer state restored.')
            except (ValueError, RuntimeError) as exc:
                accelerator.print(
                    '[RESUME] WARNING: optimizer state is incompatible with the '
                    f'current trainable-parameter layout; using a fresh optimizer. ({exc})'
                )

        # --- Scheduler state ---
        if 'scheduler' in resume_ckpt and optimizer_restored:
            try:
                scheduler.load_state_dict(resume_ckpt['scheduler'])
                accelerator.print('[RESUME] Scheduler state restored.')
            except (ValueError, RuntimeError, KeyError) as exc:
                accelerator.print(
                    f'[RESUME] WARNING: scheduler state was not restored. ({exc})'
                )

        # --- Epoch ---
        if 'epoch' in resume_ckpt:
            args.start_epoch = int(resume_ckpt['epoch']) + 1
            accelerator.print(f'[RESUME] Resuming from epoch {args.start_epoch} (checkpoint epoch={resume_ckpt["epoch"]})')
    elif args.resume is not None:
        accelerator.print(f'[RESUME] WARNING: --resume path not found: {args.resume}. Starting from scratch.')

    ema_model = None
    if args.use_ema:
        ema_avg_fn = get_ema_multi_avg_fn(args.ema_decay)
        ema_model = AveragedModel(model, multi_avg_fn=ema_avg_fn)
        accelerator.print(f"[CHECK] EMA Model initialized with decay: {args.ema_decay}")

    train_iter, valid_iter = cycle(train_dl), cycle(valid_dl)
    base = accelerator.unwrap_model(model)
    print("[CHECK] context flag:", getattr(base, "context", None))
    print("[CHECK] has tokenizer:", hasattr(base, "tokenizer"))
    print("[CHECK] has text_encoder:", hasattr(base, "text_encoder"))
    # (text_encoder 동결 로직은 DDP 래핑 이전으로 이동되었습니다.)

    # === Loss function ===
    if args.loss_fct == 'dice':
        # 배경 클래스 비율이 압도적으로 높을 때 CE Loss가 튀는 현상을 막기 위해 lambda_ce 매개변수를 낮춥니다.
        loss_fct = DiceCELoss(include_background=False, softmax=True, batch=True,
                              smooth_nr=1e-3, smooth_dr=1e-3, to_onehot_y=True, lambda_ce=0.2, lambda_dice=1.0)
    elif args.loss_fct == 'dice_focal':
        # 출혈 크기가 극도로 작은 클래스 불균형 문제를 해결하기 위해 Focal Loss 도입 (gamma=2.0)
        loss_fct = DiceFocalLoss(include_background=False, softmax=True, batch=True,
                                 smooth_nr=1e-3, smooth_dr=1e-3, to_onehot_y=True, lambda_focal=1.0, lambda_dice=1.0, gamma=2.0)
    elif args.loss_fct == 'tversky':
        # False Negative(병변을 놓침)에 엄청난 페널티(beta=0.7~0.8)를 부여하여 0.0 붕괴 방지
        loss_fct = TverskyLoss(include_background=False, softmax=True, batch=True,
                               to_onehot_y=True, alpha=0.3, beta=0.7)
    elif args.loss_fct == 'gdfl':
        # Generalized Dice Focal Loss (GDFL) - 가장 극단적인 클래스 불균형 해결 (부피 기준 가중치 + 난이도 기준 가중치)
        loss_fct = GeneralizedDiceFocalLoss(include_background=False, softmax=True, batch=True,
                                            to_onehot_y=True, lambda_focal=1.0, lambda_gdl=1.0, gamma=2.0)
    else:
        pw = None
        if getattr(args, "pos_weight", None) is not None:
            pw = torch.tensor([float(args.pos_weight)], dtype=torch.float,
                              device=accelerator.device)
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=pw, reduction='mean')

    cls_loss_weight = float(getattr(args, "cls_loss_weight", 0.0) or 0.0)
    cls_loss_fct = None
    if cls_loss_weight > 0.0:
        cls_pw = None
        if getattr(args, "cls_pos_weight", None) is not None:
            cls_pw = torch.tensor([float(args.cls_pos_weight)], dtype=torch.float,
                                  device=accelerator.device)
        cls_loss_fct = nn.BCEWithLogitsLoss(pos_weight=cls_pw, reduction='mean')
        accelerator.print(
            f"[INFO] Auxiliary classification loss enabled: weight={cls_loss_weight}, "
            f"pos_weight={getattr(args, 'cls_pos_weight', None)}"
        )

    # === Training loop ===
    for epoch in range(args.start_epoch, args.epochs + 1):
        is_warmup = epoch <= args.warmup_epochs
        phase_str = "[Warmup] " if is_warmup else ""
        
        accelerator.print(f"{phase_str}Epoch {epoch}")
        model.train()
        epoch_train_loss = []
        base_model = accelerator.unwrap_model(model)
        if args.context and hasattr(base_model, "text_encoder") and base_model.text_encoder is not None:
            base_model.text_encoder.eval()

        progress_bar = tqdm(
            range(args.n_iter_per_epoch),
            desc=f'{phase_str}Epoch {epoch}/{args.epochs}',
            dynamic_ncols=True,
            disable=not accelerator.is_local_main_process
        )

        accumulated_loss = 0.0
        accumulated_seg_loss = 0.0
        accumulated_cls_loss = 0.0
        acc_step_count = 0

        for n_iter in progress_bar:
            batch = next(train_iter)

            image = batch['data'].to(accelerator.device)   # (B,3,D,H,W)
            mask = batch['target'].to(accelerator.device).float()  # (B,1,D,H,W)

            # [WAVELET] 입력에 3D DWT 적용
            if wavelet is not None:
                image_in = wavelet.decomposition(image)    # (B,24,D',H',W')
            else:
                image_in = image
            dicom_numeric = (
                batch['dicom_numeric'].to(accelerator.device)
                if args.use_dicom else None
            )
            dicom_categorical = (
                batch['dicom_categorical'].to(accelerator.device)
                if args.use_dicom else None
            )

            with accelerator.accumulate(model):
                with accelerator.autocast():
                    return_cls = cls_loss_fct is not None
                    if args.context:
                        contexts = list(batch['context'])
                        if getattr(base_model, "tokenizer", None) is not None:
                            ctx_ids, attn_mask = make_context_tokens_batch(
                                base_model.tokenizer, base_model.max_length, base_model.context_length,
                                contexts, accelerator.device
                            )
                            if accelerator.is_local_main_process and epoch == args.start_epoch and n_iter == 0:
                                base_model = accelerator.unwrap_model(model)
                                print("[DBG] tokenizer.pad_token_id:", int(base_model.tokenizer.pad_token_id))
                                if hasattr(base_model, "text_encoder") and base_model.text_encoder is not None:
                                    print("[DBG] text_encoder.pad_id:", int(getattr(base_model.text_encoder, "pad_id", -1)))
                        else:
                            ctx_ids, attn_mask = None, None
                        model_out = model(
                            image_in,
                            report_in=ctx_ids,
                            attn_mask=attn_mask,
                            dicom_numeric=dicom_numeric,
                            dicom_categorical=dicom_categorical,
                            return_cls=return_cls,
                            return_context_variants=args.safe_context_training,
                            force_text_fusion=(
                                epoch <= args.text_fusion_warmup_epochs
                            ),
                            context_patch_mask_probability=args.context_patch_mask_probability,
                            context_hard_bypass_threshold=args.context_hard_bypass_threshold,
                        )
                    else:
                        model_out = model(
                            image_in,
                            dicom_numeric=dicom_numeric,
                            dicom_categorical=dicom_categorical,
                            return_cls=return_cls,
                        )

                    vision_logits_wt = None
                    raw_fused_logits_wt = None
                    if isinstance(model_out, dict):
                        logits_wt = model_out["fused_logits"]
                        vision_logits_wt = model_out["vision_logits"]
                        raw_fused_logits_wt = model_out["raw_fused_logits"]
                        cls_logits = model_out.get("cls_logits")
                    elif return_cls:
                        logits_wt, cls_logits = model_out
                    else:
                        logits_wt, cls_logits = model_out, None

                    # [WAVELET] 출력 wavelet coeff → full-res seg logits 복원
                    if wavelet is not None:
                        seg_logits = wavelet.reconstruction(logits_wt)
                        vision_seg_logits = (
                            wavelet.reconstruction(vision_logits_wt)
                            if vision_logits_wt is not None else None
                        )
                        raw_fused_seg_logits = (
                            wavelet.reconstruction(raw_fused_logits_wt)
                            if raw_fused_logits_wt is not None else None
                        )
                    else:
                        seg_logits = logits_wt
                        vision_seg_logits = vision_logits_wt
                        raw_fused_seg_logits = raw_fused_logits_wt

                    # === Loss 계산 ===
                    if args.loss_fct in ['dice', 'dice_focal', 'tversky', 'gdfl']:
                        if seg_logits.shape[2:] != mask.shape[2:]:
                            mask_resized = F.interpolate(mask.float(), size=seg_logits.shape[2:], mode="nearest")
                        else:
                            mask_resized = mask
                        mask_resized = (mask_resized > 0).long()
                        seg_loss = loss_fct(seg_logits.float(), mask_resized)
                    else:
                        target_bin = (mask > 0).float()
                        if target_bin.ndim == 4:
                            target_bin = target_bin.unsqueeze(1)
                        if seg_logits.shape[2:] != target_bin.shape[2:]:
                            target_bin = F.interpolate(target_bin, size=seg_logits.shape[2:], mode="nearest")
                        if accelerator.is_local_main_process:
                            print(f"Current Mask Sum: {mask.sum().item()}")
                        seg_loss = loss_fct(seg_logits.float(), target_bin)

                    loss = seg_loss
                    auxiliary_losses = {}
                    if vision_seg_logits is not None:
                        if args.loss_fct in ['dice', 'dice_focal', 'tversky', 'gdfl']:
                            vision_seg_loss = loss_fct(
                                vision_seg_logits.float(), mask_resized
                            )
                        else:
                            vision_seg_loss = loss_fct(
                                vision_seg_logits.float(), target_bin
                            )
                        loss = loss + args.vision_seg_weight * vision_seg_loss
                        auxiliary_losses["vision_seg"] = vision_seg_loss

                        if vision_seg_logits.shape[1] == 1:
                            vision_probability = torch.sigmoid(vision_seg_logits.float())
                            fused_probability = torch.sigmoid(seg_logits.float())
                        else:
                            vision_probability = torch.softmax(
                                vision_seg_logits.float(), dim=1
                            )[:, 1:2]
                            fused_probability = torch.softmax(
                                seg_logits.float(), dim=1
                            )[:, 1:2]
                        vision_uncertainty = 4.0 * vision_probability * (
                            1.0 - vision_probability
                        )
                        null_consistency = (
                            (1.0 - vision_uncertainty)
                            * (fused_probability - vision_probability).square()
                        ).mean()
                        residual_penalty = (
                            raw_fused_seg_logits.float() - vision_seg_logits.float()
                        ).abs().mean()
                        loss = (
                            loss
                            + args.null_consistency_weight * null_consistency
                            + args.context_residual_weight * residual_penalty
                        )
                        auxiliary_losses["null_consistency"] = null_consistency
                        auxiliary_losses["context_residual"] = residual_penalty

                        context_aux = getattr(
                            accelerator.unwrap_model(model), "context_aux", {}
                        )
                        burden_logits = context_aux.get("text_burden_logits")
                        trauma_logits = context_aux.get("text_trauma_logits")
                        if burden_logits is not None:
                            burden_target = batch["hemorrhage_burden"].to(
                                accelerator.device
                            ).float()
                            burden_loss = F.binary_cross_entropy_with_logits(
                                burden_logits.float(), burden_target
                            )
                            loss = loss + args.text_burden_weight * burden_loss
                            auxiliary_losses["text_burden"] = burden_loss
                        if trauma_logits is not None:
                            trauma_target = batch["trauma_target"].to(
                                accelerator.device
                            ).float()
                            trauma_loss = F.binary_cross_entropy_with_logits(
                                trauma_logits.float(), trauma_target
                            )
                            loss = loss + args.text_trauma_weight * trauma_loss
                            auxiliary_losses["text_trauma"] = trauma_loss

                        compatibility_terms = []
                        for name, stats in context_aux.items():
                            if not name.startswith("fusion_") or not isinstance(stats, dict):
                                continue
                            correct = stats.get("confidence_vector")
                            shuffled = stats.get("shuffled_confidence_vector")
                            if correct is not None and shuffled is not None:
                                compatibility_terms.append(
                                    F.relu(
                                        args.compatibility_margin
                                        - correct.float()
                                        + shuffled.float()
                                    ).mean()
                                )
                        if compatibility_terms:
                            compatibility_loss = torch.stack(
                                compatibility_terms
                            ).mean()
                            loss = loss + args.compatibility_weight * compatibility_loss
                            auxiliary_losses["compatibility"] = compatibility_loss

                    cls_loss = None
                    if cls_loss_fct is not None:
                        cls_target = (mask.flatten(1).sum(dim=1) > 0).float()
                        cls_loss = cls_loss_fct(cls_logits.float().view(-1), cls_target)
                        loss = loss + cls_loss_weight * cls_loss

                # NaN/Inf 체크
                is_bad_batch = torch.isnan(loss) or torch.isinf(loss)
                is_bad_tensor = torch.tensor([1.0 if is_bad_batch else 0.0], device=accelerator.device)
                accelerator.wait_for_everyone()
                global_is_bad = accelerator.reduce(is_bad_tensor, reduction="sum").item() > 0

                if global_is_bad:
                    accelerator.print(f"Skipping batch {n_iter} due to NaN/inf loss.")
                    loss = loss * 0.0  # Zero out to maintain accumulation flow

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # accelerate가 SequentialLR 등 복합 스케줄러의 step()을 자동으로 
                # 건너뛰지(skip) 못하는 버그(Mismatched state)를 방지하기 위해 강제로 묶음
                if accelerator.sync_gradients:
                    scheduler.step()

            # Loss 누적 (NaN이 아닌 경우만)
            if not global_is_bad:
                accumulated_loss += loss.item()
                accumulated_seg_loss += seg_loss.item()
                if cls_loss is not None:
                    accumulated_cls_loss += cls_loss.item()
                acc_step_count += 1

            # accelerator.sync_gradients = TRUE 일 때가 실제 1개 축적 사이클이 완료된 시점입니다.
            if accelerator.sync_gradients:
                avg_acc_loss = accumulated_loss / max(acc_step_count, 1)
                avg_seg_loss = accumulated_seg_loss / max(acc_step_count, 1)
                avg_cls_loss = accumulated_cls_loss / max(acc_step_count, 1)
                epoch_train_loss.append(avg_acc_loss)

                if ema_model is not None:
                    ema_model.update_parameters(model)

                # 실제 Weight Update가 일어날 때만 WandB에 seg_loss와 lr을 로깅합니다.
                # (진동 없는 깨끗한 그래프를 보실 수 있습니다)
                # X축(step) 기준을 과거 그래프들과 동일하게 맞추기 위해 실제 iteration 기준으로 로깅합니다.
                global_step = n_iter + (epoch - 1) * args.n_iter_per_epoch
                
                log_payload = {
                    'seg_loss': avg_seg_loss,
                    'train/total_loss': avg_acc_loss,
                    'train/seg_loss': avg_seg_loss,
                    "lr": scheduler.get_last_lr()[-1],
                    'epoch': epoch
                }
                for loss_name, loss_value in auxiliary_losses.items():
                    log_payload[f"train/{loss_name}_loss"] = float(
                        loss_value.detach().item()
                    )
                context_aux = getattr(accelerator.unwrap_model(model), "context_aux", {})
                for fusion_name, fusion_stats in context_aux.items():
                    if not fusion_name.startswith("fusion_") or not isinstance(fusion_stats, dict):
                        continue
                    for stat_name, stat_value in fusion_stats.items():
                        if torch.is_tensor(stat_value) and stat_value.numel() == 1:
                            log_payload[
                                f"train/context_{fusion_name}_{stat_name}"
                            ] = float(stat_value.detach().item())
                dicom_aux = getattr(accelerator.unwrap_model(model), "dicom_aux", {})
                for fusion_name, fusion_stats in dicom_aux.items():
                    if not isinstance(fusion_stats, dict):
                        continue
                    for stat_name, stat_value in fusion_stats.items():
                        if torch.is_tensor(stat_value) and stat_value.numel() == 1:
                            log_payload[
                                f"train/dicom_{fusion_name}_{stat_name}"
                            ] = float(stat_value.detach().item())
                if cls_loss_fct is not None:
                    log_payload['train/cls_loss'] = avg_cls_loss
                accelerator.log(log_payload, step=global_step)
                
                progress_bar.set_postfix({'loss': f'{avg_acc_loss:.4f}'})

                # Reset accumulator for next cycle
                accumulated_loss = 0.0
                accumulated_seg_loss = 0.0
                accumulated_cls_loss = 0.0
                acc_step_count = 0

            accelerator.wait_for_everyone()

        loss_lists = gather_object(epoch_train_loss)
        flat_losses = []
        for ls in loss_lists:
            if isinstance(ls, (list, tuple, np.ndarray)):
                flat_losses.extend([float(x) for x in ls])
            else:
                flat_losses.append(float(ls))
        mean_train_loss = float(np.mean(flat_losses)) if flat_losses else float('nan')
        accelerator.print(f'Epoch {epoch} train loss: {mean_train_loss:.6f}')
        accelerator.log({"epoch_train_loss": mean_train_loss, "epoch": epoch},
                        step=epoch * args.n_iter_per_epoch)

        # === Validation ===
        val_dice_for_this_epoch = 0.0
        if epoch % args.val_interval == 0:
            eval_model = ema_model.module if ema_model is not None else model
            eval_model.eval()
            epoch_valid_tp, epoch_valid_fp, epoch_valid_fn = 0, 0, 0
            epoch_valid_loss = []
            epoch_valid_cls_loss = []
            epoch_valid_cls_correct = 0
            epoch_valid_cls_count = 0
            wandb_seg_img = None
            progress_bar = tqdm(
                range(args.n_iter_valid),
                desc=f'Validation {epoch}/{args.epochs}',
                dynamic_ncols=True,
                disable=not accelerator.is_local_main_process
            )

            
            with torch.no_grad():
                for _ in progress_bar:
                    batch = next(valid_iter)
                    image = batch['data'].to(accelerator.device)
                    mask = batch['target'].to(accelerator.device).float()

                    # OOM 방지: Validation 연산 시 float32 대신 Mixed Precision (fp16/bf16) 강제 적용
                    with accelerator.autocast():
                        if wavelet is not None:
                            image_in = wavelet.decomposition(image)
                        else:
                            image_in = image
                        dicom_numeric = (
                            batch['dicom_numeric'].to(accelerator.device)
                            if args.use_dicom else None
                        )
                        dicom_categorical = (
                            batch['dicom_categorical'].to(accelerator.device)
                            if args.use_dicom else None
                        )

                        return_cls = cls_loss_fct is not None
                        if args.context:
                            contexts = list(batch['context'])
                            base_eval_model = accelerator.unwrap_model(eval_model)
                            if getattr(base_eval_model, "tokenizer", None) is not None:
                                ctx_ids, attn_mask = make_context_tokens_batch(
                                    base_eval_model.tokenizer, base_eval_model.max_length, base_eval_model.context_length,
                                    contexts, accelerator.device
                                )
                            else:
                                ctx_ids, attn_mask = None, None
                            model_out = eval_model(
                                image_in,
                                report_in=ctx_ids,
                                attn_mask=attn_mask,
                                dicom_numeric=dicom_numeric,
                                dicom_categorical=dicom_categorical,
                                cfg_scale=getattr(args, 'cfg_scale', 1.0),
                                return_cls=return_cls,
                                return_context_variants=args.safe_context_training,
                                context_hard_bypass_threshold=args.context_hard_bypass_threshold,
                            )
                        else:
                            model_out = eval_model(
                                image_in,
                                dicom_numeric=dicom_numeric,
                                dicom_categorical=dicom_categorical,
                                return_cls=return_cls,
                            )

                        if isinstance(model_out, dict):
                            logits_wt = model_out["fused_logits"]
                            cls_logits = model_out.get("cls_logits")
                        elif return_cls:
                            logits_wt, cls_logits = model_out
                        else:
                            logits_wt, cls_logits = model_out, None

                        # wavelet 출력 → seg logits 복원
                        if wavelet is not None:
                            seg_logits = wavelet.reconstruction(logits_wt)
                        else:
                            seg_logits = logits_wt

                    # Validation loss
                    if args.loss_fct in ['dice', 'dice_focal', 'tversky', 'gdfl']:
                        if seg_logits.shape[2:] != mask.shape[2:]:
                            mask_resized = F.interpolate(mask.float(), size=seg_logits.shape[2:], mode="nearest")
                        else:
                            mask_resized = mask
                        mask_resized = (mask_resized > 0).long()
                        # [FP16 Safety] Validation loss 계산 시에도 동일하게 float32 변환
                        seg_loss = loss_fct(seg_logits.float(), mask_resized)
                    else:
                        target_bin = (mask > 0).float()
                        if target_bin.ndim == 4:
                            target_bin = target_bin.unsqueeze(1)

                        if seg_logits.shape[2:] != target_bin.shape[2:]:
                            target_bin = F.interpolate(target_bin, size=seg_logits.shape[2:], mode="nearest")

                        seg_loss = loss_fct(seg_logits.float(), target_bin)

                    loss = seg_loss
                    if cls_loss_fct is not None:
                        cls_target = (mask.flatten(1).sum(dim=1) > 0).float()
                        cls_loss = cls_loss_fct(cls_logits.float().view(-1), cls_target)
                        loss = seg_loss + cls_loss_weight * cls_loss
                        epoch_valid_cls_loss.append(cls_loss.item())

                        cls_pred = torch.sigmoid(cls_logits.float().view(-1)) >= 0.5
                        cls_target_bool = cls_target >= 0.5
                        epoch_valid_cls_correct += int((cls_pred == cls_target_bool).sum().item())
                        epoch_valid_cls_count += int(cls_target.numel())
                    epoch_valid_loss.append(loss.item())

                    # Dice metric용 pred/gt
                    if args.loss_fct != 'bce':
                        # 모델 출력이 2채널(0:배경, 1:출혈)이므로 Softmax를 씌우고,
                        prob = torch.softmax(seg_logits, dim=1)
                        # 출혈을 나타내는 '인덱스 1' 채널만 쏙 뽑아냅니다 -> (B, 1, D, H, W)
                        pred = prob[:, 1:2, ...] 
                    else:
                        # BCE 방식 (1채널)
                        prob = torch.sigmoid(seg_logits)
                        pred = prob
                    gt   = (mask > 0).float()

                    if pred.ndim == 4:
                        pred = pred.unsqueeze(1)
                    if gt.ndim == 4:
                        gt = gt.unsqueeze(1)
                    if pred.shape[2:] != gt.shape[2:]:
                        pred = F.interpolate(pred, size=gt.shape[2:], mode="bilinear", align_corners=False)
                    pred_bin = (pred > 0.5)
                    gt_bin   = (gt   > 0.5)
                    
                    # --- WANDB IMAGE LOGGING (Capture 1 sample per validation epoch) ---
                    if wandb_seg_img is None and accelerator.is_main_process:
                        try:
                            # (B, C, D, H, W) -> 중앙 Slice 추출
                            d_idx = pred_bin.shape[2] // 2
                            img_slice = image[0, 0, d_idx, :, :].detach().cpu().numpy()
                            # 픽셀값 0~255 변환 (이미지 시각화를 위함)
                            img_slice = ((img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8) * 255.0).astype(np.uint8)
                            
                            p_slice = pred_bin[0, 0, d_idx, :, :].detach().cpu().numpy().astype(np.uint8)
                            g_slice = gt_bin[0, 0, d_idx, :, :].detach().cpu().numpy().astype(np.uint8)
                            
                            wandb_seg_img = wandb.Image(
                                img_slice,
                                masks={
                                    "predictions": {"mask_data": p_slice, "class_labels": {0: "BG", 1: "Hemo"}},
                                    "ground_truth": {"mask_data": g_slice, "class_labels": {0: "BG", 1: "Hemo"}}
                                }
                            )
                        except Exception as e:
                            pass
                            
                    pred_bin = pred_bin.view(-1)
                    gt_bin   = gt_bin.view(-1)

                    tp = (pred_bin & gt_bin).sum().item()
                    fp = (pred_bin & (~gt_bin)).sum().item()
                    fn = ((~pred_bin) & gt_bin).sum().item()

                    epoch_valid_tp += int(tp)
                    epoch_valid_fp += int(fp)
                    epoch_valid_fn += int(fn)

            loss_lists_val = gather_object(epoch_valid_loss)
            flat_losses_val = []
            for ls in loss_lists_val:
                if isinstance(ls, (list, tuple, np.ndarray)):
                    flat_losses_val.extend([float(x) for x in ls])
                else:
                    flat_losses_val.append(float(ls))
            mean_valid_loss = float(np.mean(flat_losses_val)) if flat_losses_val else float('nan')

            mean_valid_cls_loss = float('nan')
            if cls_loss_fct is not None:
                cls_loss_lists_val = gather_object(epoch_valid_cls_loss)
                flat_cls_losses_val = []
                for ls in cls_loss_lists_val:
                    if isinstance(ls, (list, tuple, np.ndarray)):
                        flat_cls_losses_val.extend([float(x) for x in ls])
                    else:
                        flat_cls_losses_val.append(float(ls))
                mean_valid_cls_loss = float(np.mean(flat_cls_losses_val)) if flat_cls_losses_val else float('nan')

            tp_sum = accelerator.reduce(
                torch.tensor(epoch_valid_tp, device=accelerator.device, dtype=torch.long),
                reduction="sum"
            ).item()
            fp_sum = accelerator.reduce(
                torch.tensor(epoch_valid_fp, device=accelerator.device, dtype=torch.long),
                reduction="sum"
            ).item()
            fn_sum = accelerator.reduce(
                torch.tensor(epoch_valid_fn, device=accelerator.device, dtype=torch.long),
                reduction="sum"
            ).item()
            cls_correct_sum = 0
            cls_count_sum = 0
            if cls_loss_fct is not None:
                cls_correct_sum = accelerator.reduce(
                    torch.tensor(epoch_valid_cls_correct, device=accelerator.device, dtype=torch.long),
                    reduction="sum"
                ).item()
                cls_count_sum = accelerator.reduce(
                    torch.tensor(epoch_valid_cls_count, device=accelerator.device, dtype=torch.long),
                    reduction="sum"
                ).item()

            eps = 1e-6
            sens = tp_sum / (tp_sum + fn_sum + eps)
            dsc = (2 * tp_sum) / (2 * tp_sum + fp_sum + fn_sum + eps)
            val_dice_for_this_epoch = dsc

            accelerator.print(f'Epoch {epoch} valid loss       : {mean_valid_loss:.6f}')
            accelerator.print(f'Epoch {epoch} valid sensitivity: {sens:.4f}')
            accelerator.print(f'Epoch {epoch} valid dice       : {dsc:.4f}')
            if cls_loss_fct is not None:
                cls_acc = cls_correct_sum / max(cls_count_sum, 1)
                accelerator.print(f'Epoch {epoch} valid cls loss   : {mean_valid_cls_loss:.6f}')
                accelerator.print(f'Epoch {epoch} valid cls acc    : {cls_acc:.4f}')
            
            log_dict = {
                "epoch_valid_loss": mean_valid_loss, 
                "epoch_valid_sensitivity": sens,
                "epoch_valid_dice": dsc, 
                "epoch": epoch
            }
            if cls_loss_fct is not None:
                log_dict["epoch_valid_cls_loss"] = mean_valid_cls_loss
                log_dict["epoch_valid_cls_acc"] = cls_correct_sum / max(cls_count_sum, 1)
            if wandb_seg_img is not None and accelerator.is_main_process:
                log_dict["val/seg_sample"] = wandb_seg_img
                
            accelerator.log(log_dict, step=epoch * args.n_iter_per_epoch)

        if args.checkpoint_interval > 0 and epoch % args.checkpoint_interval == 0:
            unwrapped = accelerator.unwrap_model(ema_model.module if ema_model is not None else model)
            checkpoint = {
                'model': trainable_state_dict(unwrapped, include_buffers=True),  # ✅ trainable만
                'optimizer': optimizer.state_dict(),                              # ✅ optimizer 상태
                'scheduler': scheduler.state_dict(),                              # ✅ scheduler 상태
                'val_dice': val_dice_for_this_epoch,
                'epoch': epoch,
                'dicom_schema': dicom_schema,
            }
            accelerator.save(checkpoint, os.path.join(args.checkpoint_dir, f'model_epoch_{epoch}.pth'))
            accelerator.print(f"Saved checkpoint for epoch {epoch} (val_dice: {val_dice_for_this_epoch:.4f}) [with optimizer & scheduler state]")

    # === Save final model ===
    unwrapped = accelerator.unwrap_model(ema_model.module if ema_model is not None else model)
    sd = trainable_state_dict(unwrapped, include_buffers=True)
    accelerator.save(sd, os.path.join(args.checkpoint_dir, 'final_model.pth'))
    accelerator.end_training()


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    with open(os.path.join(args.checkpoint_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    main(args)
