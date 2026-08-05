import os, sys, argparse, traceback, site, json
import contextlib
import io

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

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
import nibabel as nib
sitk.ProcessObject.SetGlobalDefaultDirectionTolerance(1e-3)
from tqdm.auto import tqdm

from model_custom.stunet import get_stunet_base
from model_custom.text_feature_cache import TextFeatureCache
from data.dataset import (
    build_safe_clinical_prompt,
    encode_dicom_row,
    windowing_3ch,
)
from utils.inference import predict_sliding_window_return_logits
from utils.mask_paths import find_case_mask_path
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import gather_object
from datetime import timedelta
import math
from monai.transforms import AsDiscrete
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# ==========================================
# MONAI Metrics Setup
# ==========================================
# We use MONAI's native metric classes. 
# include_background=False ensures class 0 (Normal) is excluded from the Dice/HD95 calculation.
dice_metric = DiceMetric(include_background=False, reduction="mean")
hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95.0, reduction="mean")
post_pred = AsDiscrete(argmax=True, to_onehot=2)
post_label = AsDiscrete(to_onehot=2)


# ==========================================
# Data Loading & Text Context
# ==========================================
def load_image_3ch(img_path: Path):
    itk = sitk.ReadImage(str(img_path))
    arr = sitk.GetArrayFromImage(itk)   # (D,H,W)
    img = windowing_3ch(arr)            # (3,D,H,W) in [0,1]
    img = (img - 0.5) / 0.5             # [-1,1] normalization
    return torch.from_numpy(img).float(), itk


def find_case_image_path(case_id: str, fallback_path: Path = None) -> Path:
    test_image_dir = Path('/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/FUtest_data/image')
    for candidate in (
        test_image_dir / f"{case_id}.nii.gz",
        test_image_dir / f"{case_id}.nii",
    ):
        if candidate.exists():
            return candidate

    if fallback_path is not None and fallback_path.exists():
        return fallback_path

    return Path('/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/pair_data_nifti') / f"{case_id}.nii.gz"


def same_image_geometry(a: sitk.Image, b: sitk.Image, atol: float = 1e-5) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), rtol=0.0, atol=atol)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), rtol=0.0, atol=atol)
        and np.allclose(a.GetDirection(), b.GetDirection(), rtol=0.0, atol=atol)
    )


def align_mask_to_reference(mask_itk: sitk.Image, ref_itk: sitk.Image) -> sitk.Image:
    if same_image_geometry(mask_itk, ref_itk):
        return mask_itk

    if mask_itk.GetSize() == ref_itk.GetSize():
        aligned = sitk.Image(mask_itk)
        aligned.CopyInformation(ref_itk)
        return aligned

    return sitk.Resample(
        mask_itk, ref_itk,
        sitk.Transform(), sitk.sitkNearestNeighbor, 0, mask_itk.GetPixelID()
    )


def make_context_tokens(model, context_text: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    from train import make_context_tokens_batch
    # Re-use the logic from train.py to ensure 100% compatibility
    contexts = [context_text]
    ctx_ids, attn_mask = make_context_tokens_batch(
        model.tokenizer, model.max_length, model.context_length, contexts, device
    )
    return ctx_ids, attn_mask


def remove_small_connected_components(
    mask: np.ndarray,
    min_voxels: int = 0,
    min_volume_ml: float = 0.0,
    spacing_xyz: Optional[Tuple[float, float, float]] = None,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    nnU-Net style post-processing: remove tiny foreground islands.
    mask is binary (D, H, W). spacing_xyz follows SimpleITK order (x, y, z).
    """
    min_voxels = max(0, int(min_voxels or 0))

    if min_volume_ml and min_volume_ml > 0:
        if spacing_xyz is None:
            spacing_xyz = (1.0, 1.0, 1.0)
        voxel_volume_mm3 = float(np.prod(spacing_xyz))
        if voxel_volume_mm3 > 0:
            volume_min_voxels = int(np.ceil(float(min_volume_ml) * 1000.0 / voxel_volume_mm3))
            min_voxels = max(min_voxels, volume_min_voxels)

    mask = (mask > 0).astype(np.uint8)
    before_voxels = int(mask.sum())
    info = {
        "min_voxels": min_voxels,
        "before_voxels": before_voxels,
        "after_voxels": before_voxels,
        "removed_voxels": 0,
        "removed_components": 0,
        "kept_components": None,
    }

    if min_voxels <= 1 or before_voxels == 0:
        return mask, info

    from scipy.ndimage import label

    labeled, num_components = label(mask.astype(bool), structure=np.ones((3, 3, 3), dtype=bool))
    if num_components == 0:
        info["after_voxels"] = 0
        info["removed_voxels"] = before_voxels
        return mask, info

    sizes = np.bincount(labeled.ravel())
    keep_labels = np.flatnonzero(sizes >= min_voxels)
    keep_labels = keep_labels[keep_labels != 0]
    filtered = np.isin(labeled, keep_labels).astype(np.uint8)

    after_voxels = int(filtered.sum())
    info.update({
        "after_voxels": after_voxels,
        "removed_voxels": before_voxels - after_voxels,
        "removed_components": int(num_components - len(keep_labels)),
        "kept_components": int(len(keep_labels)),
    })
    return filtered, info


def _row_text(row, columns, default=""):
    for col in columns:
        if col in row and pd.notna(row[col]):
            text = str(row[col]).strip()
            if text:
                return text
    return default


def _axis_region(delta: float, tolerance: float, negative_label: str, positive_label: str, middle_label: str) -> str:
    if abs(delta) <= tolerance:
        return middle_label
    return positive_label if delta > 0 else negative_label


def summarize_prediction_annotation(
    case_id: str,
    row,
    pred_bin_np: np.ndarray,
    ref_img: sitk.Image,
    pred_path: Optional[Path] = None,
    max_prob: Optional[float] = None,
    prob_threshold: Optional[float] = None,
    post_info: Optional[Dict[str, object]] = None,
    case_metrics: Optional[Dict[str, object]] = None,
    prediction_source: str = "new_inference",
) -> Dict[str, object]:
    pred_bin = (pred_bin_np > 0).astype(np.uint8)
    pred_voxels = int(pred_bin.sum())
    spacing_xyz = tuple(float(x) for x in ref_img.GetSpacing())
    voxel_volume_mm3 = float(np.prod(spacing_xyz))
    volume_cc = pred_voxels * voxel_volume_mm3 / 1000.0

    connected_components = 0
    largest_component_voxels = 0
    largest_component_volume_cc = 0.0
    centroid_index_zyx = ""
    centroid_physical_xyz = ""
    main_location = "none"

    if pred_voxels > 0:
        from scipy.ndimage import label

        labeled, connected_components = label(
            pred_bin.astype(bool),
            structure=np.ones((3, 3, 3), dtype=bool),
        )
        if connected_components > 0:
            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0
            largest_label = int(np.argmax(sizes))
            largest_component_voxels = int(sizes[largest_label])
            largest_component_volume_cc = largest_component_voxels * voxel_volume_mm3 / 1000.0

            coords_zyx = np.argwhere(labeled == largest_label)
            center_zyx = coords_zyx.mean(axis=0)
            centroid_index_zyx = ",".join(f"{x:.2f}" for x in center_zyx)

            center_xyz = [float(center_zyx[2]), float(center_zyx[1]), float(center_zyx[0])]
            image_center_xyz = [(s - 1) / 2.0 for s in ref_img.GetSize()]
            try:
                center_phys = ref_img.TransformContinuousIndexToPhysicalPoint(center_xyz)
                image_center_phys = ref_img.TransformContinuousIndexToPhysicalPoint(image_center_xyz)
                centroid_physical_xyz = ",".join(f"{x:.2f}" for x in center_phys)

                extent_xyz = [
                    max(float(ref_img.GetSize()[i]) * spacing_xyz[i], spacing_xyz[i])
                    for i in range(3)
                ]
                tol_xyz = [max(extent * 0.05, spacing_xyz[i]) for i, extent in enumerate(extent_xyz)]

                # SimpleITK uses patient physical axes; for standard LPS, +x=left, +y=posterior, +z=superior.
                lr = _axis_region(center_phys[0] - image_center_phys[0], tol_xyz[0], "right", "left", "midline")
                ap = _axis_region(center_phys[1] - image_center_phys[1], tol_xyz[1], "anterior", "posterior", "middle")
                si = _axis_region(center_phys[2] - image_center_phys[2], tol_xyz[2], "inferior", "superior", "middle")
                main_location = f"{lr}-{ap}-{si}"
            except Exception:
                z, y, x = center_zyx
                shape = np.array(pred_bin.shape, dtype=float)
                lr = _axis_region(x - (shape[2] - 1) / 2.0, max(shape[2] * 0.05, 1.0), "right", "left", "midline")
                ap = _axis_region(y - (shape[1] - 1) / 2.0, max(shape[1] * 0.05, 1.0), "anterior", "posterior", "middle")
                si = _axis_region(z - (shape[0] - 1) / 2.0, max(shape[0] * 0.05, 1.0), "inferior", "superior", "middle")
                main_location = f"{lr}-{ap}-{si}"

    post_info = post_info or {}
    patient_id = _row_text(row, ["expected_pid", "Patient_ID", "patient_id", "Patient ID"], default=case_id)
    class_label = _row_text(row, ["class", "Class"], default="")
    subclass_label = _row_text(row, ["subclass", "Subclass"], default="")
    cc_text = _row_text(row, ["extracted_cc", "chief_complaint", "CC", "cc"], default="")

    record = {
        "Patient_ID": patient_id,
        "case_id": case_id,
        "class": class_label,
        "subclass": subclass_label,
        "hemorrhage_volume_ml": round(volume_cc, 4),
        "hemorrhage_volume_cc": round(volume_cc, 4),
        "voxel": pred_voxels,
        "cc": cc_text,
        "connected_components": int(connected_components),
        "largest_component_voxels": int(largest_component_voxels),
        "largest_component_volume_cc": round(largest_component_volume_cc, 4),
        "main_hemorrhage_location": main_location,
        "main_hemorrhage_centroid_index_zyx": centroid_index_zyx,
        "main_hemorrhage_centroid_physical_xyz": centroid_physical_xyz,
        "max_hemo_prob": round(float(max_prob), 6) if max_prob is not None and np.isfinite(max_prob) else "",
        "prob_threshold": float(prob_threshold) if prob_threshold is not None else "",
        "post_before_voxels": post_info.get("before_voxels", ""),
        "post_after_voxels": post_info.get("after_voxels", ""),
        "post_removed_voxels": post_info.get("removed_voxels", ""),
        "post_removed_components": post_info.get("removed_components", ""),
        "pred_path": str(pred_path) if pred_path else "",
        "prediction_source": prediction_source,
    }
    if case_metrics:
        record.update(case_metrics)
    return record


# ==========================================
# Main Script
# ==========================================
def main(args):
    try:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        
        # Init Accelerator for multi-gpu inference
        ipg_handler = InitProcessGroupKwargs(timeout=timedelta(seconds=54000))
        accelerator = Accelerator(mixed_precision=None, kwargs_handlers=[ipg_handler]) # We cast manually
        device = accelerator.device
        
        # Check and handle cuDNN initialization issues
        if getattr(args, 'no_cudnn', False):
            import torch.backends.cudnn as cudnn
            cudnn.enabled = False
            accelerator.print("[INFO] 🚫 cuDNN manually disabled via --no_cudnn. Using native PyTorch CUDA kernels.")
        else:
            try:
                if torch.cuda.is_available():
                    # Test cuDNN initialization with a tiny dummy 3D convolution
                    x_dummy = torch.zeros(1, 1, 3, 3, 3, device=device)
                    w_dummy = torch.zeros(1, 1, 3, 3, 3, device=device)
                    torch.nn.functional.conv3d(x_dummy, w_dummy)
            except RuntimeError as e:
                if "cuDNN" in str(e) or "CUDNN" in str(e):
                    import torch.backends.cudnn as cudnn
                    cudnn.enabled = False
                    accelerator.print("[WARN] ⚠️ cuDNN failed to initialize. Automatically disabling cuDNN to use native PyTorch CUDA kernels instead.")
            except Exception:
                pass

        accelerator.print(f"[INFO] Using Device: {device} (Process {accelerator.process_index}/{accelerator.num_processes})")
        if getattr(args, 'context', False):
            accelerator.print(f"[INFO] CFG scale: {getattr(args, 'cfg_scale', 1.0)}")
        
        CSV_PATH = Path(args.csv_path)
        save_root = Path(args.save_root)
        if accelerator.is_main_process:
            save_root.mkdir(parents=True, exist_ok=True)
            if getattr(args, 'save_annotation_csv', True) and getattr(args, 'overwrite_annotation_csv', True):
                for stale_path in save_root.glob("annotation_rank*.csv"):
                    stale_path.unlink(missing_ok=True)
                (save_root / getattr(args, 'annotation_csv_name', 'annotation.csv')).unlink(missing_ok=True)
        accelerator.wait_for_everyone()
            
        # Load CSV or Excel
        if CSV_PATH.suffix == '.xlsx':
            df = pd.read_excel(CSV_PATH, engine='openpyxl')
        else:
            df = pd.read_csv(CSV_PATH, sep=None, engine='python', encoding='utf-8-sig')
        
        total_len = len(df)
        accelerator.print(f"[INFO] Total cases in CSV/Excel: {total_len}")
        
        # Split DataFrame for DDP
        chunk_size = math.ceil(total_len / accelerator.num_processes)
        start_idx = accelerator.process_index * chunk_size
        end_idx = min(start_idx + chunk_size, total_len)
        df = df.iloc[start_idx:end_idx]
        
        print(f"👉 [Process {accelerator.process_index}] Handling {len(df)} cases: index {start_idx} to {end_idx-1}")
        
        # Safe text prompts never contain DICOM, reports, refined EMR or labels.
        prompt_map = {}
        if args.context:
            text_columns = []
            if args.include_cc:
                text_columns.append("extracted_cc")
            if args.include_chief_complaint:
                text_columns.append("chief_complaint")
            prompt_map = build_safe_clinical_prompt(df, text_columns)
            accelerator.print(f"[INFO] Safe text columns: {text_columns}")
        else:
            accelerator.print("[INFO] Vision-only mode active. Skipping text prompt generation.")
 
        # Check existing predictions before loading model weights to inform user of resume status
        existing_count = 0
        for _, row in df.iterrows():
            case_id = str(row['영상일련번호ID']).strip()
            pred_exist_path = save_root / case_id / "pred_hemo_bin.nii.gz"
            if getattr(args, 'save_pred', True) and not getattr(args, 'overwrite_pred', False) and pred_exist_path.exists():
                existing_count += 1
        
        if existing_count > 0:
            accelerator.print(f"🔄 [RESUME] Found {existing_count} existing predictions. These will be skipped automatically.")

        # Load Weights (Tolerant/Strict=False for safe loading)
        print(f"[INFO] Loading Checkpoint from: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
        sd = checkpoint['model'] if 'model' in checkpoint else checkpoint
        dicom_schema = checkpoint.get("dicom_schema") if isinstance(checkpoint, dict) else None
        if args.use_dicom and dicom_schema is None:
            schema_path = Path(args.model_path).with_name("dicom_schema.json")
            if not schema_path.exists():
                raise FileNotFoundError(
                    f"DICOM FiLM requires its train-fitted schema: {schema_path}"
                )
            with open(schema_path, encoding="utf-8") as schema_file:
                dicom_schema = json.load(schema_file)
        
        cleaned_sd = {}
        for k, v in sd.items():
            if k.startswith('module.'):
                cleaned_sd[k[7:]] = v
            else:
                cleaned_sd[k] = v

        # [AUTO DETECT] 이전 구버전 checkpoint(num_classes=1) 추론 지원용 동적 클래스 탐지
        detected_num_classes = 2 # 기본값
        if 'seg_outputs.0.weight' in cleaned_sd:
            detected_num_classes = cleaned_sd['seg_outputs.0.weight'].shape[0]
            accelerator.print(f"[INFO] 가중치 파일에서 동적으로 num_classes={detected_num_classes} 임을 감지했습니다.")

        # Initialize Model
        llm_repo = args.llm_repo if args.context else None
        soft_prompt_mode = args.soft_prompt_mode
        if soft_prompt_mode is None:
            soft_prompt_mode = (
                "learned" if "contexts" in cleaned_sd else "disabled"
            )
            accelerator.print(
                "[INFO] Auto-detected soft_prompt_mode="
                f"{soft_prompt_mode} from checkpoint keys."
            )
        text_feature_cache = None
        cached_text_dim = 4096
        if args.text_feature_cache:
            if not args.context:
                raise ValueError("--text_feature_cache requires --context.")
            if soft_prompt_mode != "disabled":
                raise ValueError(
                    "--text_feature_cache requires a no-soft-prompt checkpoint."
                )
            text_feature_cache = TextFeatureCache(
                args.text_feature_cache, read_only=True
            )
            cached_text_dim = int(
                text_feature_cache.metadata.get("hidden_dim", 4096)
            )
            accelerator.print(
                f"[TEXT CACHE] entries={len(text_feature_cache)} "
                f"hidden_dim={cached_text_dim} path={args.text_feature_cache}"
            )
        model = get_stunet_base(
            num_input_channels=3,
            num_classes=detected_num_classes,   
            enable_deep_supervision=False,
            context=args.context,
            llm_repo=llm_repo,
            use_lora=getattr(args, 'use_lora', False),
            soft_prompt_mode=soft_prompt_mode,
            use_cached_text_features=text_feature_cache is not None,
            cached_text_dim=cached_text_dim,
            use_dicom=args.use_dicom,
            dicom_numeric_dim=10,
            dicom_category_sizes=(
                tuple(
                    len(dicom_schema["categories"][column])
                    for column in dicom_schema["categorical_columns"]
                )
                if dicom_schema is not None else (2, 2)
            ),
        )

        # [AUTO DETECT] 이전 구버전 checkpoint(Soft Prompt 차원 다름) 형태 탐지 및 덮어쓰기
        if 'contexts' in cleaned_sd and hasattr(model, 'contexts') and model.contexts is not None:
            # 만약 과거 코드 에러로 인해 contexts 파라미터가 배치 차원(예: 4)을 포함하여 저장되었을 경우 1로 깎아냅니다.
            if cleaned_sd['contexts'].shape[0] > 1:
                cleaned_sd['contexts'] = cleaned_sd['contexts'][0:1, :, :]
                accelerator.print(f"[INFO] 구버전의 contexts 배치 차원이 감지되어 {cleaned_sd['contexts'].shape} 으로 슬라이싱했습니다.")
            
            ckpt_contexts_shape = cleaned_sd['contexts'].shape
            if ckpt_contexts_shape != model.contexts.shape:
                accelerator.print(f"[INFO] 텍스트 프롬프트(contexts) 형태를 구버전 {ckpt_contexts_shape} 에 맞게 재정의합니다.")
                model.contexts = nn.Parameter(torch.zeros(ckpt_contexts_shape, dtype=model.contexts.dtype, device=model.contexts.device))

        if args.use_ema:
            ema_avg_fn = get_ema_multi_avg_fn(args.ema_decay)
            ema_model = AveragedModel(model, multi_avg_fn=ema_avg_fn)
            missing, unexpected = ema_model.load_state_dict(cleaned_sd, strict=False)
            model = ema_model.module
            print(f"[INFO] Loaded EMA Weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        else:
            missing, unexpected = model.load_state_dict(cleaned_sd, strict=False)
            print(f"[INFO] Loaded Weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        
        model.to(device).eval()
        
        # --- Soft Prompt 번역 (전역으로 한 번만 수행) ---
        global_nearest_words = "N/A"
        if args.context and accelerator.is_main_process:
            try:
                base_m = getattr(model, "module", model)
                if hasattr(base_m, "contexts") and base_m.contexts is not None:
                    soft_prompt = base_m.contexts # [1, 16, Dim]
                    token_embeddings = base_m.text_encoder.token_embedding.weight
                    # L2 거리 계산 후 가장 가까운 단어 인덱스 추출
                    distances = torch.cdist(soft_prompt[0], token_embeddings)
                    nearest_token_ids = distances.argmin(dim=-1)
                    global_nearest_words = base_m.tokenizer.convert_ids_to_tokens(nearest_token_ids)
                    
                    print(f"\n💡 [Soft Prompt Info]")
                    print(f" - Shape: {soft_prompt.shape}")
                    print(f" - Nearest Tokens: {global_nearest_words}\n")
            except Exception as e:
                print(f"Soft prompt translation failed: {e}")
        
        # Metrics setup
        dice_metric.reset()
        hd95_metric.reset()
        sens_list_positive = []   # 출혈 케이스 sensitivity
        sens_list_normal   = []   # 정상 케이스 (GT empty) → specificity용
        n_positive = 0            # GT 출혈 케이스 수
        n_normal   = 0            # GT 정상 케이스 수
        n_unlabeled = 0           # GT mask를 찾지 못한 케이스 수
        processed = 0
        annotation_records = []

        # Progress bar
        if accelerator.is_main_process:
            pbar = tqdm(df.iterrows(), total=len(df), desc=f"Infer & Eval (Process {accelerator.process_index})")
        else:
            pbar = df.iterrows()
            
        for _, row in pbar:
            case_id = str(row['영상일련번호ID']).strip()
            
            # Skip if prediction already exists
            pred_exist_path = save_root / case_id / "pred_hemo_bin.nii.gz"
            if (
                getattr(args, 'save_pred', True)
                and pred_exist_path.exists()
                and not getattr(args, 'overwrite_pred', False)
            ):
                if getattr(args, 'save_annotation_csv', True):
                    try:
                        pred_itk = sitk.ReadImage(str(pred_exist_path))
                        pred_arr = sitk.GetArrayFromImage(pred_itk)
                        annotation_records.append(
                            summarize_prediction_annotation(
                                case_id,
                                row,
                                pred_arr,
                                pred_itk,
                                pred_path=pred_exist_path,
                                prediction_source="existing_pred",
                            )
                        )
                    except Exception as ann_e:
                        if accelerator.is_main_process:
                            tqdm.write(f"[WARN] Failed to annotate existing prediction for {case_id}: {ann_e}")
                processed += 1
                continue
            
            # --- Image Path ---
            fallback_img_path = None
            if 'image_path' in row and pd.notna(row['image_path']):
                fallback_img_path = Path(str(row['image_path']).strip())
            img_path = find_case_image_path(case_id, fallback_img_path)

            if not img_path.exists():
                print(f"[SKIP] Missing image file on disk for {case_id}: {img_path}")
                continue

            # --- Mask Path ---
            explicit_mask_path = None
            if 'mask_path' in row and pd.notna(row['mask_path']):
                explicit_mask_path = Path(str(row['mask_path']).strip())

            mask_base = Path('/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/FUdata')
            test_mask_dir = Path('/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/FUtest_data/mask')
            mask_search_dirs = [
                test_mask_dir,
                mask_base / 'hemo_masks' / 'thick_th0.56',
                mask_base / 'hemo_masks' / 'thin_th0.56',
                mask_base / 'normal_masks',
                mask_base / 'mask',
            ]
            mask_path = find_case_mask_path(case_id, explicit_mask_path, mask_search_dirs)

            has_gt = mask_path is not None and mask_path.exists()
            if not has_gt and accelerator.is_main_process:
                tqdm.write(f"[WARN] Missing mask file for {case_id}. Prediction will be saved without online GT metrics.")

            img_t, itk_ref = load_image_3ch(img_path)
            
            ctx_ids, attn_mask = None, None
            cached_text_features, cached_text_mask = None, None
            if args.context:
                ctx_txt = prompt_map.get(case_id, "<SEG>")
                
                # 프롬프트 출력 (최초 3개까지만 출력하거나 전체 출력. 여기서는 전체 출력)
                if accelerator.is_main_process and getattr(args, 'print_discrete_prompts', False):
                    tqdm.write(f"📝 [{case_id}] Prompt: {ctx_txt}")
                    
                    # CSV에 기록
                    csv_log_path = Path("./prompts.csv")
                    # 파일이 없으면 헤더 작성
                    if not csv_log_path.exists():
                        with open(csv_log_path, "w", encoding="utf-8-sig") as f:
                            f.write("case_id,discrete_prompt,soft_prompt_nearest_tokens\n")
                    
                    # 데이터 추가
                    with open(csv_log_path, "a", encoding="utf-8-sig") as f:
                        # 쉼표가 포함될 수 있으므로 쌍따옴표 처리
                        clean_ctx = str(ctx_txt).replace('"', '""')
                        clean_soft = str(global_nearest_words).replace('"', '""')
                        f.write(f'"{case_id}","{clean_ctx}","{clean_soft}"\n')

                if text_feature_cache is not None:
                    cached_text_features, cached_text_mask = (
                        text_feature_cache.get_batch([ctx_txt], device)
                    )
                else:
                    ctx_ids, attn_mask = make_context_tokens(
                        model, ctx_txt, device
                    )

            dicom_numeric, dicom_categorical = None, None
            if args.use_dicom:
                numeric_np, categorical_np = encode_dicom_row(row, dicom_schema)
                dicom_numeric = torch.from_numpy(numeric_np).unsqueeze(0).to(device)
                dicom_categorical = (
                    torch.from_numpy(categorical_np).unsqueeze(0).to(device)
                )

            # Inference
            amp_enabled = torch.cuda.is_available()
            amp_dtype = torch.bfloat16 if str(args.mixed_precision).lower() == 'bf16' else torch.float16
            
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                original_forward = model.forward
                if args.context or args.use_dicom:
                    model.forward = lambda x, **kwargs: original_forward(
                        x,
                        report_in=(
                            ctx_ids.expand(x.shape[0], -1)
                            if ctx_ids is not None else None
                        ),
                        attn_mask=(
                            attn_mask.expand(x.shape[0], -1)
                            if attn_mask is not None else None
                        ),
                        precomputed_text_features=(
                            cached_text_features.expand(x.shape[0], -1, -1)
                            if cached_text_features is not None else None
                        ),
                        precomputed_text_mask=(
                            cached_text_mask.expand(x.shape[0], -1)
                            if cached_text_mask is not None else None
                        ),
                        dicom_numeric=(
                            dicom_numeric.expand(x.shape[0], -1)
                            if dicom_numeric is not None else None
                        ),
                        dicom_categorical=(
                            dicom_categorical.expand(x.shape[0], -1)
                            if dicom_categorical is not None else None
                        ),
                        cfg_scale=1.0,
                        return_context_variants=args.context,
                        context_hard_bypass_threshold=args.context_hard_bypass_threshold,
                    )
                
                logits = predict_sliding_window_return_logits(
                    model, img_t, tuple(args.patch_size), device,
                    step_size=0.5, mirror_axes=None,
                    sw_batch_size=getattr(args, 'sw_batch_size', 1),
                ) # Returns (2, D, H, W) without batch dimension!
                
                # Restore original forward
                model.forward = original_forward
                
            # Logits to Pred
            prob_threshold = float(getattr(args, 'prob_threshold', 0.5))
            if detected_num_classes == 1:
                # 1 Class (Sigmoid 기반 BCE) 로직
                hemo_prob = torch.sigmoid(logits)[0].float().cpu().numpy()
                pred_bin_np = (hemo_prob >= prob_threshold).astype(np.uint8)
            else:
                # 2 Classes (Softmax 기반 CE) 로직
                hemo_prob = torch.softmax(logits, dim=0)[1].float().cpu().numpy()
                pred_bin_np = (hemo_prob >= prob_threshold).astype(np.uint8)

            pred_bin_np, post_info = remove_small_connected_components(
                pred_bin_np,
                min_voxels=getattr(args, 'min_component_voxels', 0),
                min_volume_ml=getattr(args, 'min_component_volume_ml', 0.0),
                spacing_xyz=itk_ref.GetSpacing(),
            )

            pred_onehot = torch.zeros((1, 2, *pred_bin_np.shape), dtype=torch.float32, device=device)
            pred_onehot[0, 1] = torch.from_numpy(pred_bin_np).to(device).float()
            pred_onehot[0, 0] = 1.0 - pred_onehot[0, 1]
             
            gt_is_positive = None
            case_metrics = {}
            if has_gt:
                # Ground Truth Loading
                try:
                    gt_itk = sitk.ReadImage(str(mask_path))
                except RuntimeError:
                    nib_img = nib.load(str(mask_path))
                    arr_tmp = np.asarray(nib_img.dataobj)
                    gt_itk = sitk.GetImageFromArray(arr_tmp)
                    gt_itk.SetOrigin(itk_ref.GetOrigin())
                    gt_itk.SetSpacing(itk_ref.GetSpacing())
                    gt_itk.SetDirection(itk_ref.GetDirection())

                gt_rs = align_mask_to_reference(gt_itk, itk_ref)
                gt_mask = sitk.GetArrayFromImage(gt_rs)
                gt_bin = (gt_mask > 0).astype(np.uint8) # (D, H, W)
                
                gt_tensor = torch.from_numpy(gt_bin).unsqueeze(0).unsqueeze(0).to(device) # (1, 1, D, H, W)
                gt_onehot = post_label(gt_tensor[0]).unsqueeze(0) # (1, 2, D, H, W)
                
                # ── GT 출혈 여부 판단 ──
                gt_bin_np = gt_onehot[0, 1].cpu().numpy()
                gt_is_positive = gt_bin_np.sum() > 0
                pred_bool = pred_bin_np.astype(bool)
                gt_bool = gt_bin_np.astype(bool)
                pred_voxels = int(pred_bool.sum())
                gt_voxels = int(gt_bool.sum())
                tp = int(np.logical_and(pred_bool, gt_bool).sum())
                fp = int(np.logical_and(pred_bool, np.logical_not(gt_bool)).sum())
                fn = int(np.logical_and(np.logical_not(pred_bool), gt_bool).sum())
                dice_denominator = pred_voxels + gt_voxels
                case_metrics = {
                    "gt_is_positive": bool(gt_is_positive),
                    "gt_voxels": gt_voxels,
                    "pred_voxels": pred_voxels,
                    "tp_voxels": tp,
                    "fp_voxels": fp,
                    "fn_voxels": fn,
                    "case_dice": (
                        2.0 * tp / dice_denominator
                        if dice_denominator > 0 else 1.0
                    ),
                    "case_sensitivity": (
                        tp / (tp + fn + 1e-6) if gt_voxels > 0 else ""
                    ),
                    "normal_no_fp": (
                        "" if gt_is_positive else float(pred_voxels == 0)
                    ),
                }

                # ── MONAI Metrics: 출혈 케이스만 집계 (정상군 Dice=NaN 오염 방지) ──
                if gt_is_positive:
                    dice_metric(y_pred=pred_onehot, y=gt_onehot)
                    hd95_metric(y_pred=pred_onehot, y=gt_onehot)
                    n_positive += 1

                    sens = tp / (tp + fn + 1e-6)
                    sens_list_positive.append(sens)
                else:
                    # 정상군: 예측에 출혈이 없으면 TN (specificity)
                    n_normal += 1
                    pred_has_lesion = pred_bin_np.sum() > 0
                    sens_list_normal.append(0.0 if pred_has_lesion else 1.0)
            else:
                n_unlabeled += 1

            # Print Max Prob for context
            if np.max(hemo_prob) > 0.001:
                if gt_is_positive is None:
                    tag = "[NO_GT]"
                else:
                    tag = "[ICH]" if gt_is_positive else "[NRM]"
                msg = (
                    f"{tag}[{case_id}] Max Prob: {np.max(hemo_prob):.6f}, "
                    f"Voxels >= {prob_threshold:g}: "
                    f"{post_info['before_voxels']} -> {post_info['after_voxels']}"
                )
                if post_info["removed_components"]:
                    msg += (
                        f" | removed CC: {post_info['removed_components']} "
                        f"({post_info['removed_voxels']} voxels, min={post_info['min_voxels']})"
                    )
                if accelerator.is_main_process:
                    tqdm.write(msg)
                else:
                    print(msg)

            if accelerator.is_main_process:
                disp_sens = sens_list_positive[-1] if gt_is_positive is True else float('nan')
                pbar.set_postfix({'Sens(ICH)': f"{disp_sens:.4f}" if not np.isnan(disp_sens) else 'N/A'})
            processed += 1
            
            # Save NIfTI 
            annotation_pred_path = None
            if getattr(args, 'save_pred', True):
                out_case_dir = save_root / case_id
                out_case_dir.mkdir(parents=True, exist_ok=True)
                
                # Restore to original numpy shape and cast to SITK
                pred_img = sitk.GetImageFromArray(pred_bin_np.astype(np.uint8))
                pred_img.CopyInformation(itk_ref)
                
                # Atomic save to prevent corrupted files on interruption
                final_path = out_case_dir / "pred_hemo_bin.nii.gz"
                temp_path = out_case_dir / "pred_hemo_bin_temp.nii.gz"
                
                sitk.WriteImage(pred_img, str(temp_path))
                temp_path.rename(final_path)
                annotation_pred_path = final_path

            if getattr(args, 'save_annotation_csv', True):
                annotation_records.append(
                    summarize_prediction_annotation(
                        case_id,
                        row,
                        pred_bin_np,
                        itk_ref,
                        pred_path=annotation_pred_path,
                        max_prob=float(np.max(hemo_prob)),
                        prob_threshold=prob_threshold,
                        post_info=post_info,
                        case_metrics=case_metrics,
                        prediction_source="new_inference",
                    )
                )
            
        if getattr(args, 'save_annotation_csv', True):
            rank_annotation_path = save_root / f"annotation_rank{accelerator.process_index}.csv"
            pd.DataFrame(annotation_records).to_csv(rank_annotation_path, index=False, encoding='utf-8-sig')

        accelerator.wait_for_everyone()

        if accelerator.is_main_process and getattr(args, 'save_annotation_csv', True):
            annotation_parts = []
            for rank_idx in range(accelerator.num_processes):
                part_path = save_root / f"annotation_rank{rank_idx}.csv"
                if part_path.exists():
                    try:
                        part_df = pd.read_csv(part_path, encoding='utf-8-sig')
                        if not part_df.empty:
                            annotation_parts.append(part_df)
                    except Exception as read_e:
                        tqdm.write(f"[WARN] Failed to read annotation part {part_path}: {read_e}")

            if annotation_parts:
                annotation_df = pd.concat(annotation_parts, ignore_index=True)
                if 'case_id' in annotation_df.columns:
                    annotation_df = annotation_df.drop_duplicates(subset=['case_id'], keep='last')
                    annotation_df = annotation_df.sort_values(by='case_id').reset_index(drop=True)
                annotation_path = save_root / getattr(args, 'annotation_csv_name', 'annotation.csv')
                annotation_df.to_csv(annotation_path, index=False, encoding='utf-8-sig')
                accelerator.print(f"[INFO] Saved annotation CSV: {annotation_path} ({len(annotation_df)} cases)")

        def metric_buffer_values(metric):
            buffer = metric.get_buffer()
            if buffer is None:
                return []
            return buffer.detach().float().cpu().reshape(-1).tolist()

        local_summary = {
            "processed": processed,
            "n_positive": n_positive,
            "n_normal": n_normal,
            "n_unlabeled": n_unlabeled,
            "dice_positive": metric_buffer_values(dice_metric),
            "hd95_positive_mm": metric_buffer_values(hd95_metric),
            "sensitivity_positive": [float(v) for v in sens_list_positive],
            "normal_no_fp": [float(v) for v in sens_list_normal],
        }
        gathered_summaries = gather_object([local_summary])

        if text_feature_cache is not None:
            text_feature_cache.close()

        if accelerator.is_main_process:
            summaries = [item for item in gathered_summaries if isinstance(item, dict)]

            def merged_values(name):
                values = []
                for summary_part in summaries:
                    values.extend(summary_part.get(name, []))
                return values

            def finite_mean(values):
                values = np.asarray(values, dtype=np.float64)
                values = values[np.isfinite(values)]
                return float(values.mean()) if values.size else None

            dice_values = merged_values("dice_positive")
            hd95_values = merged_values("hd95_positive_mm")
            sens_values = merged_values("sensitivity_positive")
            normal_values = merged_values("normal_no_fp")
            summary = {
                "model_path": str(args.model_path),
                "csv_path": str(args.csv_path),
                "world_size": accelerator.num_processes,
                "processed": sum(int(s.get("processed", 0)) for s in summaries),
                "n_positive": sum(int(s.get("n_positive", 0)) for s in summaries),
                "n_normal": sum(int(s.get("n_normal", 0)) for s in summaries),
                "n_unlabeled": sum(int(s.get("n_unlabeled", 0)) for s in summaries),
                "mean_dice_positive": finite_mean(dice_values),
                "mean_sensitivity_positive": finite_mean(sens_values),
                "mean_hd95_positive_mm": finite_mean(hd95_values),
                "hd95_finite_cases": int(np.isfinite(np.asarray(hd95_values)).sum()),
                "normal_no_fp_rate": finite_mean(normal_values),
            }
            summary_path = save_root / "evaluation_summary.json"
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            def show(value, digits=4):
                return "N/A" if value is None else f"{value:.{digits}f}"

            print("\n" + "="*60)
            print("INFERENCE EVALUATION SUMMARY [all processes]")
            print("="*60)
            print(
                f"Total Processed : {summary['processed']}  "
                f"(ICH: {summary['n_positive']}, Normal: {summary['n_normal']}, "
                f"No GT: {summary['n_unlabeled']})"
            )
            print("─"*60)
            print(f"  - Mean Dice (ICH only):         {show(summary['mean_dice_positive'])}")
            print(f"  - Mean Sensitivity (ICH voxel): {show(summary['mean_sensitivity_positive'])}")
            print(f"  - Mean HD95 (finite cases):     {show(summary['mean_hd95_positive_mm'], 2)} mm")
            print(f"  - Normal no-FP rate:            {show(summary['normal_no_fp_rate'])}")
            print(f"  - Summary JSON:                 {summary_path}")
            print("="*60)
        
    except Exception as e:
        print("\n[EXCEPTION] Script crashed:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to best/final pth")
    parser.add_argument("--save_root", type=str, required=True, help="Path to save predictions")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to eval csv or xlsx (must have 'image_path' and 'mask_path')")
    parser.add_argument(
        "--llm_repo",
        type=str,
        default="/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_SEG_hk/model_custom/llama2/Llama-2-7b-chat-hf/",
    )
    parser.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=False, help="Use PEFT LoRA for LLM backbone")
    parser.add_argument(
        "--soft_prompt_mode",
        choices=["learned", "disabled"],
        default=None,
        help="Override soft-prompt mode; default auto-detects from checkpoint.",
    )
    parser.add_argument(
        "--text_feature_cache",
        type=str,
        default=None,
        help="Read-only SQLite cache for no-soft-prompt context inference.",
    )
    parser.add_argument("--patch_size", type=int, nargs=3, default=[32, 224, 224])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--context", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_dicom", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include_cc", action=argparse.BooleanOptionalAction, default=True,
                        help="Include extracted_cc in the safe prompt.")
    parser.add_argument("--include_chief_complaint", action=argparse.BooleanOptionalAction,
                        default=True, help="Include chief_complaint in the safe prompt.")
    parser.add_argument("--context_hard_bypass_threshold", type=float, default=0.5)
    parser.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=False, help="Load EMA AveragedModel weights")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--mixed_precision", type=str, default='bf16', help="fp16 or bf16")
    parser.add_argument("--save_pred", action=argparse.BooleanOptionalAction, default=True, help="Save resulting NIfTI files")
    parser.add_argument("--sw_batch_size", type=int, default=1, help="Number of sliding window patches to batch together (increase for more VRAM utilization)")
    parser.add_argument("--cfg_scale", type=float, default=1.0,
                        help="Deprecated compatibility option; CFG is disabled.")
    parser.add_argument("--prob_threshold", type=float, default=0.5, help="Foreground probability threshold before post-processing")
    parser.add_argument("--min_component_voxels", type=int, default=0, help="Remove connected components smaller than this many voxels (0 disables)")
    parser.add_argument("--min_component_volume_ml", type=float, default=0.0, help="Remove connected components smaller than this volume in mL (0 disables)")
    parser.add_argument("--overwrite_pred", action=argparse.BooleanOptionalAction, default=False, help="Recompute even if pred_hemo_bin.nii.gz already exists")
    parser.add_argument("--no_cudnn", action="store_true", help="Disable cuDNN and use native PyTorch CUDA kernels instead")
    parser.add_argument("--print_discrete_prompts", action="store_true", help="Print and log generated text prompts in addition to soft prompt info")
    parser.add_argument("--save_annotation_csv", action=argparse.BooleanOptionalAction, default=True, help="Save per-case annotation CSV under save_root")
    parser.add_argument("--annotation_csv_name", type=str, default="annotation.csv", help="Merged annotation CSV filename under save_root")
    parser.add_argument("--overwrite_annotation_csv", action=argparse.BooleanOptionalAction, default=True, help="Remove stale annotation CSV shards before this run")
    args = parser.parse_args()
    
    main(args)
