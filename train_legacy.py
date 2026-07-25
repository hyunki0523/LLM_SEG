import os, json, argparse
os.environ["WANDB_MODE"] = "offline"
os.environ["ACCELERATE_MIXED_PRECISION"] = "fp16"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from monai.losses import DiceCELoss
from accelerate import Accelerator
from accelerate.utils import gather_object
from accelerate.utils import DistributedDataParallelKwargs
from accelerate import InitProcessGroupKwargs, DataLoaderConfiguration
from datetime import timedelta

from data.dataset import get_data_loaders
from model_custom.stunet_legacy import get_stunet_base
#import pywt, ptwt

class WaveletTransform2D:
    def __init__(self, wavelet: str = 'haar', level: int = 1, mode: str = 'zero'):
        self.wavelet = pywt.Wavelet(wavelet)
        self.level = level
        self.mode = mode

    def decomposition(self, image: torch.Tensor) -> torch.Tensor:
        """
        image: (B, C, H, W)
        return: (B, C * 4, H', W')  # 1레벨 기준
        """
        coeffs2 = ptwt.wavedec2(image, self.wavelet, level=self.level, mode=self.mode)
        # coeffs2[0]: LL (B, C, H', W')
        # coeffs2[1]: (LH, HL, HH) 각각 (B, C, H', W')
        return torch.cat((coeffs2[0], *coeffs2[1]), dim=1)

    def reconstruction(self, image: torch.Tensor) -> torch.Tensor:
        C = image.shape[1]
        assert C % 4 == 0, "The number of channels must be a multiple of 4 for 2D wavelet reconstruction"
        n = C // 4
        LL = image[:, :n, :, :]
        LH, HL, HH = image[:, n:2 * n, :, :], image[:, 2 * n:3 * n, :, :], image[:, 3 * n:, :, :]
        coeffs = [LL, (LH, HL, HH)]
        return ptwt.waverec2(coeffs, self.wavelet)


class WaveletTransform3D:
    def __init__(self, wavelet: str = 'haar', level: int = 1, mode: str = 'zero'):
        self.wavelet = pywt.Wavelet(wavelet)
        self.level = level
        self.mode = mode

    def decomposition(self, image: torch.Tensor) -> torch.Tensor:
        """
        image: (B, C, D, H, W)
        return: (B, C * N_subbands, D', H', W')  # 1레벨 기준 N_subbands=8
        """
        coeffs3 = ptwt.wavedec3(image, self.wavelet, level=self.level, mode=self.mode)
        # coeffs3[0] : LL(L L L)
        # coeffs3[1] : dict {'aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd'}
        return torch.cat((coeffs3[0], *coeffs3[1].values()), dim=1)

    def reconstruction(self, image: torch.Tensor) -> torch.Tensor:
        """
        image: (B, C, D, H, W)  # C = 1 + 7 = 8 * 원래채널 (1레벨 기준)
        """
        # 여기서는 원래 채널 수를 알고 있다고 가정하고, 가장 단순한 형태로 작성
        # 키 순서는 wavedec3가 반환하는 순서와 동일해야 함
        keys = ['aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd']
        LL = image[:, 0:1, :, :, :]
        detail_dict = {}
        for i, k in enumerate(keys):
            detail_dict[k] = image[:, i + 1:i + 2, :, :, :]
        coeffs3 = (LL, detail_dict)
        return ptwt.waverec3(coeffs3, self.wavelet)

def trainable_state_dict(model, include_buffers=True):
    """
    model의 state_dict 중 requires_grad=True인 파라미터(그리고 선택적으로 해당 모듈의 버퍼)만 반환.
    """
    # 학습 중인 파라미터 이름들
    trainable_param_names = {n for n, p in model.named_parameters() if p.requires_grad}

    # 버퍼 포함 옵션: (BatchNorm running_mean 등) 학습 중인 모듈의 버퍼만 포함
    buffer_names = set()
    if include_buffers:
        # 파라미터가 속한 모듈 prefix들(e.g. "layer1.block2")
        module_prefixes = {n.rsplit('.', 1)[0] for n in trainable_param_names if '.' in n}
        # 파라미터 이름이 모듈 루트(최상위)인 경우도 대비
        module_prefixes |= {n for n in trainable_param_names if '.' not in n}
        for n, _ in model.named_buffers():
            # 버퍼가 학습 중인 모듈에 속하면 포함
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
    
# function of counting trainable parameters
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_args_parser():
    parser = argparse.ArgumentParser('LLM-Seg-3D', add_help=False)

    parser.add_argument('--config', type=str, default='', help='Path to json config file')

    # Training parameters
    parser.add_argument('--epochs',  default=600, type=int)
    parser.add_argument('--n_iter_per_epoch',  default=250, type=int)
    parser.add_argument('--n_iter_valid',  default=50, type=int)
    parser.add_argument('--val_interval',  default=1, type=int)
    parser.add_argument('--mixed_precision',  default='no', type=str)
    parser.add_argument('--loss_fct',  default='bce', type=str, choices=['dice', 'bce'])
    parser.add_argument('--context', default=True, type=str2bool, help='Use context in the model')

    # Dataset parameters  
    parser.add_argument('--channels',  default=3, type=int)
    parser.add_argument('--batch_size',  default=1, type=int)
    parser.add_argument('--num_workers',  default=6, type=int)
    parser.add_argument('--seed',  default=42, type=int)
    parser.add_argument('--positive_prob',  default=0.8, type=float, help='Probability of sampling positive examples')
    parser.add_argument('--rater', default=1, type=rater_type, help='Rater ID for the dataset')
    parser.add_argument('--use_full_volume', default=False, type=str2bool, help='If True, ignore patch_size and use full 3D volume')
    parser.add_argument('--patch_size', default=(16, 224, 224), type=int, nargs=3,help='3D patch size (D H W)')
    
    # Model parameters 
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--num_classes', default=1, type=int)

    parser.add_argument('--start_epoch', default=1, type=int, help='Epoch to start training from')

    # Continue Training (Resume)
    parser.add_argument('--pretrained', default=None,  help='pretrained checkpoint')

    # Save setting
    parser.add_argument('--checkpoint_dir', default='', help='path where to save checkpoint or output')
    parser.add_argument('--experiment_name', default='default', help='name of the experience')
    
    # Dataset parsing paths
    parser.add_argument('--train_csv', default="/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/ICH_pair/train_split.csv", type=str)
    parser.add_argument('--valid_csv', default="/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/ICH_pair/valid_split.csv", type=str)

    # prompt parameter 
    parser.add_argument('--include_clinical', default=True, type=str2bool, help='Include Clinical Information in prompt')
    parser.add_argument('--include_findings', default=True, type=str2bool, help='Include Findings in prompt')
    parser.add_argument('--include_emr', default=True, type=str2bool, help='Include EMR/CC text in prompt (False = DICOM header only)')
    parser.add_argument('--llm_repo', default='/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_seg_jhk/model_custom/llama2/Llama-2-7b-chat-hf', type=str, help='HF repo id for LLM backbone')


    #wavelet
    parser.add_argument('--use_wavelet',default=False, type=str2bool, help='Apply 3D wavelet transform to input before STUNet')

    return parser
def make_context_tokens_batch(tokenizer, max_length, context_length, contexts, device):
    ids_list = []
    max_len = max_length - context_length - 1
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        try:
            eos_id = tokenizer.convert_tokens_to_ids("</s>")
        except Exception:
            eos_id = 0

    for ctx in contexts:
        tok = tokenizer.encode(ctx or "")
        tok = tok[:max_len] + [int(eos_id)]
        ids_list.append(torch.tensor(tok, dtype=torch.long))

    L = min(max(x.numel() for x in ids_list), max_length - context_length)
    padded = [F.pad(t[:L], (0, L - t.numel()), value=0) for t in ids_list]
    return torch.stack(padded, dim=0).to(device)   # (B, L)



def main(args):
    #torch.autograd.set_detect_anomaly(True)
    seed_everything()
    
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    ipg_handler = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
    data_config = DataLoaderConfiguration(split_batches=False)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with='wandb',
        kwargs_handlers=[kwargs, ipg_handler],
        dataloader_config=data_config
    )
    accelerator.init_trackers("LLM-SEG")
    patch_size = tuple(args.patch_size)
    
    # === DataLoaders ===
    train_dl, valid_dl = get_data_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        positive_prob=args.positive_prob,
        rater=args.rater,
        patch_size=patch_size,
        include_clinical=args.include_clinical,
        include_findings=args.include_findings,
        include_emr=args.include_emr,
        train_csv=args.train_csv,
        valid_csv=args.valid_csv
    )
        # === Wavelet 설정 ===
    wavelet = None
    model_in_ch = args.channels 
    
    if args.use_wavelet:
        wavelet = WaveletTransform3D(wavelet='haar', level=1, mode='zero')

        # dummy 1개로 wavelet 이후 채널 수 계산 (공간 크기는 아무거나 상관 없음)
        dummy = torch.zeros(
            1,
            args.channels,         # 3ch 입력
            patch_size[0],
            patch_size[1],
            patch_size[2],
            device=accelerator.device,
        )
        with torch.no_grad():
            wt_out = wavelet.decomposition(dummy)

        model_in_ch = wt_out.shape[1]
        accelerator.print(
            f"[Wavelet] enabled: {args.channels}ch -> {model_in_ch}ch after 3D DWT"
        )
        # === Model / Optimizer / Scheduler ===
    out_ch = 1 if args.loss_fct == 'bce' else (args.num_classes + 1)
    model = get_stunet_base(
        num_input_channels=model_in_ch,   # wavelet 이후 채널 수
        num_classes=out_ch,
        enable_deep_supervision=False,
        context=args.context,
        llama_rep=args.llm_repo
    )
    accelerator.print('parameters:', count_params(model))

    # 기본값: 인자로 들어온 start_epoch (기본 1)
    start_epoch = args.start_epoch

    if args.pretrained is not None and os.path.exists(args.pretrained):
        accelerator.print(f'Loading pretrained weights from: {args.pretrained}')
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        
        # Extract state_dict from the new checkpoint format
        if 'model' in checkpoint:
            sd = checkpoint['model']
        else:
            # Fallback for old format where the object is the state_dict itself
            sd = checkpoint

        # Clean keys for DDP/etc. by removing 'module.' prefix
        cleaned_sd = {}
        for k, v in sd.items():
            if k.startswith('module.'):
                cleaned_sd[k[7:]] = v
            else:
                cleaned_sd[k] = v
                
        model.load_state_dict(cleaned_sd, strict=True)
        accelerator.print('Loaded pretrained model')

        # 🔥 체크포인트에 epoch 정보가 있으면, 자동으로 다음 epoch부터 시작
        if 'epoch' in checkpoint and args.start_epoch == 1:
            # 예: checkpoint['epoch'] == 103 이면 → 104부터 시작
            start_epoch = int(checkpoint['epoch']) + 1
            accelerator.print(
                f"Checkpoint epoch={checkpoint['epoch']} detected. "
                f"Training will resume from epoch {start_epoch}."
            )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.01,
        total_iters=args.n_iter_per_epoch * args.epochs   # world_size 곱하지 않음
    )

    # === Prepare everything for multi-GPU ===
    model, optimizer, scheduler, train_dl, valid_dl = accelerator.prepare(
        model, optimizer, scheduler, train_dl, valid_dl
    )
    train_iter, valid_iter = cycle(train_dl), cycle(valid_dl)
    base = accelerator.unwrap_model(model)
    if args.context and hasattr(base, "text_encoder"):
        for p in base.text_encoder.parameters():
            p.requires_grad = False
        base.text_encoder.eval()

        # V100은 bf16 미지원 → fp16 사용 중이면 half로 줄이기 (권장)
        if str(args.mixed_precision).lower() == "fp16":
            base.text_encoder.half()

    # === Loss function ===
    if args.loss_fct == 'dice':
        loss_fct = DiceCELoss(include_background=False, softmax=True, batch=True,
                              smooth_nr=1e-3, smooth_dr=1e-3, to_onehot_y=True)
    else:
        pw = None
        if getattr(args, "pos_weight", None) is not None:
            pw = torch.tensor([float(args.pos_weight)], dtype=torch.float,
                              device=accelerator.device)
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=pw, reduction='mean')

    # === Training loop ===
    global_valid_dice, global_valid_dice_ema = 0, 0
    for epoch in range(start_epoch, args.epochs + 1):
        accelerator.print(f"Epoch {epoch}")
        model.train()
        epoch_train_loss = []
        base_model = accelerator.unwrap_model(model)
        
        progress_bar = tqdm(
            range(args.n_iter_per_epoch),
            desc=f'Epoch {epoch}/{args.epochs}',
            dynamic_ncols=True,
            disable=not accelerator.is_local_main_process
        )

        for n_iter in progress_bar:
            batch = next(train_iter)
            optimizer.zero_grad(set_to_none=True)

            image = batch['data'].to(accelerator.device)     # (B,3,D,H,W)
            # wavelet 적용 (채널 3 → 24 등)
            if args.use_wavelet and wavelet is not None:
                image = wavelet.decomposition(image)         # (B, model_in_ch, D',H',W')

            mask  = batch['target'].to(accelerator.device).float()

            with accelerator.autocast():
                if args.context:
                    contexts = list(batch['context'])
                    ctx_ids  = make_context_tokens_batch(
                        base_model.tokenizer, base_model.max_length, base_model.context_length,
                        contexts, accelerator.device
                    )
                    logits   = model(image, ctx_ids)
                else:
                    logits   = model(image)

                if args.loss_fct == 'dice':
                    loss = loss_fct(logits, mask.long())
                else:
                    target_bin = (mask > 0).float()
                    if target_bin.ndim == 4:
                        target_bin = target_bin.unsqueeze(1)
                    # STUNet 출력 해상도와 맞추기
                    if target_bin.shape[2:] != logits.shape[2:]:
                        target_bin = F.interpolate(
                            target_bin,
                            size=logits.shape[2:],   # (D,H,W)
                            mode="nearest" )


                    if accelerator.is_local_main_process:
                        print(f"Current Mask Sum: {mask.sum().item()}")
                    loss = loss_fct(logits, target_bin)

            # Check for NaN/inf loss on each process (local state)
            is_bad_batch = torch.isnan(loss) or torch.isinf(loss)
            # Create a tensor for consensus
            is_bad_tensor = torch.tensor([1.0 if is_bad_batch else 0.0], device=accelerator.device)
            
            # Reach global consensus
            accelerator.wait_for_everyone()
            global_is_bad_tensor = accelerator.reduce(is_bad_tensor, reduction="sum")

            # If any process had a bad batch, all processes skip it
            if global_is_bad_tensor.item() > 0:
                accelerator.print(f"Skipping batch {n_iter} due to NaN/inf loss on at least one process.")
                continue

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 12)
            optimizer.step()
            scheduler.step()
            accelerator.wait_for_everyone()

            epoch_train_loss.append(loss.item())
            progress_bar.set_postfix({'loss': loss.item()})

            if n_iter % 20 == 0:
                step = n_iter + (epoch - 1) * args.n_iter_per_epoch
                accelerator.log({'seg_loss': loss.item(), "lr": scheduler.get_last_lr()[-1], 'epoch': epoch}, step=step)

        # === Epoch loss logging ===
        loss_lists = gather_object(epoch_train_loss)
        flat_losses = []
        for ls in loss_lists:
            if isinstance(ls, (list, tuple, np.ndarray)):
                flat_losses.extend([float(x) for x in ls])
            else:
                flat_losses.append(float(ls))
        mean_train_loss = float(np.mean(flat_losses)) if flat_losses else float('nan')
        accelerator.print(f'Epoch {epoch} train loss: {mean_train_loss:.6f}')
        accelerator.log({"epoch_train_loss": mean_train_loss, "epoch": epoch}, step=epoch * args.n_iter_per_epoch)

        # === Validation ===
        val_dice_for_this_epoch = 0.0
        if epoch % args.val_interval == 0:
            model.eval()
            epoch_valid_tp, epoch_valid_fp, epoch_valid_fn = 0, 0, 0
            epoch_valid_loss = []
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

                    if args.use_wavelet and wavelet is not None:
                        image = wavelet.decomposition(image)

                    mask = batch['target'].to(accelerator.device).float()
                    if args.context:
                        contexts = list(batch['context'])
                        ctx_ids  = make_context_tokens_batch(
                            base.tokenizer, base.max_length, base.context_length,
                            contexts, accelerator.device
                        )
                        logits   = model(image, ctx_ids)
                    else:
                        logits   = model(image)

                    # Calculate validation loss
                    if args.loss_fct == 'dice':
                        loss = loss_fct(logits, mask.long())
                    else:
                        target_bin = (mask > 0).float()
                        if target_bin.ndim == 4:
                            target_bin = target_bin.unsqueeze(1)
                        if target_bin.shape[2:] != logits.shape[2:]:
                            target_bin = F.interpolate(
                                target_bin,
                                size=logits.shape[2:],
                                mode="nearest"
                            )
                        loss = loss_fct(logits, target_bin)
                    epoch_valid_loss.append(loss.item())

                    prob = torch.sigmoid(logits)
                    pred = (prob > 0.5)
                    gt   = (mask > 0)
                    
                    if gt.ndim == 4:
                        gt = gt.unsqueeze(1)  # (B,1,D,H,W)
                    if gt.shape[2:] != pred.shape[2:]:
                        gt = F.interpolate(
                            gt.float(),
                            size=pred.shape[2:],   # (D,H,W)
                            mode="nearest"
                        ).bool()

                    pred = pred.squeeze(1) if pred.ndim == 5 else pred
                    gt   = gt.squeeze(1)   if gt.ndim   == 5 else gt

                    tp = (pred & gt).sum().item()
                    fp = (pred & (~gt)).sum().item()
                    fn = ((~pred) & gt).sum().item()

                    epoch_valid_tp += int(tp)
                    epoch_valid_fp += int(fp)
                    epoch_valid_fn += int(fn)

            # Gather validation loss from all processes
            loss_lists_val = gather_object(epoch_valid_loss)
            flat_losses_val = []
            for ls in loss_lists_val:
                if isinstance(ls, (list, tuple, np.ndarray)):
                    flat_losses_val.extend([float(x) for x in ls])
                else:
                    flat_losses_val.append(float(ls))
            mean_valid_loss = float(np.mean(flat_losses_val)) if flat_losses_val else float('nan')

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

            eps = 1e-6
            sens = tp_sum / (tp_sum + fn_sum + eps)
            dsc  = (2 * tp_sum) / (2 * tp_sum + fp_sum + fn_sum + eps)
            val_dice_for_this_epoch = dsc

            accelerator.print(f'Epoch {epoch} valid loss       : {mean_valid_loss:.6f}')
            accelerator.print(f'Epoch {epoch} valid sensitivity: {sens:.4f}')
            accelerator.print(f'Epoch {epoch} valid dice       : {dsc:.4f}')
            accelerator.log({"epoch_valid_loss": mean_valid_loss, "epoch_valid_sensitivity": sens, "epoch_valid_dice": dsc, "epoch": epoch}, step=epoch * args.n_iter_per_epoch)


        # === 100 에포크마다 모델 저장 ===
        if epoch % 100 == 0:
            unwrapped = accelerator.unwrap_model(model)
            checkpoint = {
                'model': unwrapped.state_dict(),
                'val_dice': val_dice_for_this_epoch,
                'epoch': epoch,
            }
            accelerator.save(checkpoint, os.path.join(args.checkpoint_dir, f'model_epoch_{epoch}.pth'))
            accelerator.print(f"Saved checkpoint for epoch {epoch} (val_dice: {val_dice_for_this_epoch:.4f})")

        # # === 매 에포크가 끝날 때마다 모델 저장 ===
        # unwrapped = accelerator.unwrap_model(model)
        # checkpoint = {
        #     'model': unwrapped.state_dict(),
        #     'val_dice': val_dice_for_this_epoch,
        #     'epoch': epoch,
        # }
        # accelerator.save(checkpoint, os.path.join(args.checkpoint_dir, f'model_epoch_{epoch}.pth'))
        # accelerator.print(f"Saved checkpoint for epoch {epoch} (val_dice: {val_dice_for_this_epoch:.4f})")

    # === Save final model ===
    unwrapped = accelerator.unwrap_model(model)
    sd = trainable_state_dict(unwrapped, include_buffers=True)
    accelerator.save(sd, os.path.join(args.checkpoint_dir, 'final_model.pth'))
    accelerator.end_training()

    
if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    
    # JSON config 파일이 주어졌다면, 파일의 설정값으로 설정 덮어씌우기
    if hasattr(args, 'config') and args.config:
        with open(args.config, 'r') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)
            
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # save args
    with open(os.path.join(args.checkpoint_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    main(args)
