#!/usr/bin/env python3
"""
compute_metrics.py
──────────────────
이미 inference_eval.py로 저장된 예측 마스크(NIfTI)와 Ground Truth를 비교하여
개별 환자별 Dice / Sensitivity / HD95 / Lesion-wise Metric을 계산하고 CSV 파일로 저장합니다.

⚡ 수정 사항:
   1. HD95 양방향 백분위수 독립 계산 후 Max 취하도록 공식 교정 (학술 가이드라인 준수)
   2. MONAI 의존성 완전 제거 -> Pure NumPy 계산으로 멀티프로세싱 안정성 및 속도 극대화
   3. Lesion-wise (Connected Component 기반 병변 단위 검출률) 지표 새롭게 추가
   4. Patient-level 판단 시 노이즈 방지를 위한 Voxel Threshold(>5) 도입
"""

import argparse
import os
import sys
from pathlib import Path

# [중요] 멀티프로세싱 시 백엔드 라이브러리가 OS 스레드 한계에 도달하는 것을 방지
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt, binary_erosion, label
from tqdm import tqdm
from utils.mask_paths import find_case_mask_path

# 일꾼 1명이 스레드를 무한정 늘리는 것을 방지 (1-Thread per Process)
sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)


DEFAULT_CSV_PATH = Path(
    "/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/FUdata/260601/final_test_set_test2.xlsx"
)
DEFAULT_TEST_MASK_DIR = Path(
    "/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/FUtest_data/mask"
)
DEFAULT_TEST_IMAGE_DIR = Path(
    "/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/FUtest_data/image"
)
NAS206_LINUX_PREFIX = "/mnt/nas206/forGPU/lhyunki"
NAS206_UNC_PREFIX = Path(r"\\192.168.45.206\forGPU\lhyunki")
NAS206_DRIVE_PREFIX = Path(r"D:\\")


def nas206_path_candidates(path_like):
    """Return likely local/UNC equivalents for /mnt/nas206 paths."""
    if path_like is None:
        return []

    text = str(path_like).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []

    candidates = [Path(text)]
    normalized = text.replace("\\", "/")
    if normalized.startswith(NAS206_LINUX_PREFIX + "/"):
        rel = normalized[len(NAS206_LINUX_PREFIX) + 1:]
        candidates.append(NAS206_UNC_PREFIX / Path(rel))
        candidates.append(NAS206_DRIVE_PREFIX / Path(rel))

    deduped = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def resolve_existing_path(path_like):
    """Prefer an existing path, while accepting Linux NAS paths on Windows."""
    candidates = nas206_path_candidates(path_like)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def resolve_required_input_path(path_like, description):
    path = resolve_existing_path(path_like)
    if path is None or not path.exists():
        raise FileNotFoundError(f"{description} not found: {path_like}")
    return path


def resolve_case_mask_path(case_id, explicit_path=None, search_dirs=None):
    resolved_explicit = resolve_existing_path(explicit_path)
    if resolved_explicit is not None and resolved_explicit.exists():
        return resolved_explicit
    return find_case_mask_path(case_id, resolved_explicit, search_dirs)


def detect_case_id_column(df, requested_col=None):
    if requested_col:
        if requested_col not in df.columns:
            raise ValueError(
                f"Requested id column '{requested_col}' was not found. "
                f"Available columns: {list(df.columns)}"
            )
        return requested_col

    preferred = ("영상일련번호ID", "case_id", "pid", "PID", "PatientID")
    for col in preferred:
        if col in df.columns:
            return col

    for col in df.columns:
        text = str(col).lower()
        if "id" in text and ("영상" in str(col) or "case" in text or "pid" in text):
            return col

    return df.columns[0]


def read_eval_table(csv_path):
    if csv_path.suffix.lower() == ".xlsx":
        return pd.read_excel(csv_path, engine="openpyxl")
    return pd.read_csv(csv_path, sep=None, engine="python", encoding="utf-8-sig")

# ===================================================================
# Metric Functions (Pure NumPy & SciPy)
# ===================================================================

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


def binarize_mask_array(arr: np.ndarray) -> np.ndarray:
    """Treat both 0/1 and 0/255 masks as binary foreground/background."""
    arr = np.nan_to_num(np.asarray(arr), nan=0.0)
    return (arr > 0).astype(np.uint8)


def is_hemo_label(value) -> bool:
    text = str(value).strip().lower()
    if not text or text in {"nan", "none"}:
        return False
    return any(key in text for key in ("hemorrhage", "hemo", "trauma", "ich"))


def is_normal_label(value) -> bool:
    text = str(value).strip().lower()
    if not text or text in {"nan", "none"}:
        return False
    return text == "normal"


def compute_hd95_corrected(pred: np.ndarray, gt: np.ndarray, spacing: tuple = (1.0, 1.0, 1.0)) -> float:
    """
    양방향의 95% 백분위수를 각각 구한 후 최댓값(Max)을 취하는 표준 HD95 계산 함수.
    Pred 또는 GT가 빈(Empty) 마스크인 경우 np.nan을 반환합니다.
    """
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)

    if pred_bool.sum() == 0 or gt_bool.sum() == 0:
        return np.nan

    # Surface voxels (border) 추출
    pred_border = np.logical_xor(pred_bool, binary_erosion(pred_bool))
    gt_border = np.logical_xor(gt_bool, binary_erosion(gt_bool))

    # 단일 픽셀인 경우 erosion 후 빈 배열이 되므로 원본 사용
    if pred_border.sum() == 0:
        pred_border = pred_bool
    if gt_border.sum() == 0:
        gt_border = gt_bool

    # Distance transforms
    dt_gt = distance_transform_edt(~gt_border, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_border, sampling=spacing)

    # Directed distances 계산
    d_pred_to_gt = dt_gt[pred_border]
    d_gt_to_pred = dt_pred[gt_border]

    # [교정] 각각 95% 백분위수를 구한 후 Max를 취함
    hd95_pred_to_gt = np.percentile(d_pred_to_gt, 95)
    hd95_gt_to_pred = np.percentile(d_gt_to_pred, 95)
    
    return float(max(hd95_pred_to_gt, hd95_gt_to_pred))


def compute_lesion_metrics(pred: np.ndarray, gt: np.ndarray):
    """
    Connected Component Labelling 기반 병변 단위(Instance-level) 검출률 계산.
    Returns:
        lesion_sens: 검출률 (hit_lesions / total_gt_lesions)
        num_gt: GT 내 총 독립 병변 개수
        hit_lesions: 모델이 1픽셀이라도 겹치게 맞춘 병변 개수
    """
    gt_labeled, num_gt = label(gt)
    if num_gt == 0:
        return np.nan, 0, 0  # GT에 병변이 없으면 계산 제외

    hit_lesions = 0
    for i in range(1, num_gt + 1):
        gt_lesion_mask = (gt_labeled == i)
        # 개별 병변 영역에 예측(pred) 결과가 1픽셀이라도 오버랩되는지 확인
        if np.logical_and(gt_lesion_mask, pred).any():
            hit_lesions += 1

    lesion_sens = hit_lesions / num_gt
    return float(lesion_sens), int(num_gt), int(hit_lesions)


# ===================================================================
def process_case(case_tuple):
    (
        case_id,
        mask_path,
        pred_root,
        class_label,
        subclass_label,
        class_hemo,
        hemo_dice_class_filter,
        empty_gt_for_missing_mask,
    ) = case_tuple

    pred_path = pred_root / case_id / "pred_hemo_bin.nii.gz"
    if not pred_path.exists():
        return None

    mask_path = Path(mask_path) if mask_path else None
    if mask_path is None and not empty_gt_for_missing_mask:
        return None
    if mask_path is not None and not mask_path.exists():
        return None

    try:
        pred_itk = sitk.ReadImage(str(pred_path))
        pred_arr = binarize_mask_array(sitk.GetArrayFromImage(pred_itk))

        if mask_path is None:
            gt_arr = np.zeros_like(pred_arr, dtype=np.uint8)
            gt_source = "synthetic_empty_missing_normal"
        else:
            try:
                gt_itk = sitk.ReadImage(str(mask_path))
            except RuntimeError:
                import nibabel as nib
                nib_img = nib.load(str(mask_path))
                arr_tmp = np.asarray(nib_img.dataobj)
                gt_itk = sitk.GetImageFromArray(arr_tmp)
                gt_itk.SetOrigin(pred_itk.GetOrigin())
                gt_itk.SetSpacing(pred_itk.GetSpacing())
                gt_itk.SetDirection(pred_itk.GetDirection())

            gt_itk = align_mask_to_reference(gt_itk, pred_itk)
            gt_arr = binarize_mask_array(sitk.GetArrayFromImage(gt_itk))
            gt_source = "mask_file"

        # Get spacing for HD95 (mm) -> ITK(W,H,D)에서 numpy(D,H,W) 순서로 맞춤
        spacing = pred_itk.GetSpacing()[::-1]

        # 1. 픽셀 수 카운팅 및 픽셀 기반 메트릭 계산
        gt_voxels = int(gt_arr.sum())
        pred_voxels = int(pred_arr.sum())
        
        tp_voxels = int(np.logical_and(pred_arr, gt_arr).sum())
        fn_voxels = int(np.logical_and(np.logical_not(pred_arr), gt_arr).sum())
        fp_voxels = int(np.logical_and(pred_arr, np.logical_not(gt_arr)).sum())

        # Pure NumPy Dice & Pixel Sensitivity
        total_voxels = pred_voxels + gt_voxels
        dice = (2.0 * tp_voxels) / total_voxels if total_voxels > 0 else np.nan
        pixel_sens = tp_voxels / (tp_voxels + fn_voxels) if (tp_voxels + fn_voxels) > 0 else 1.0

        # 2. HD95 교정본 계산
        hd95 = compute_hd95_corrected(pred_arr, gt_arr, spacing=spacing)

        # 3. 병변 단위(Lesion-wise) 메트릭 계산
        lesion_sens, gt_lesion_count, hit_lesion_count = compute_lesion_metrics(pred_arr, gt_arr)

        # 4. 환자 단위(Patient-level) 분류 플래그 설정
        # 모델의 미세한 노이즈(1~5픽셀 수준)로 인해 정상 환자가 FP로 오염되는 것을 방지하기 위해 5픽셀 임계값 적용
        VOXEL_THRESHOLD = 5
        gt_has_lesion = gt_voxels > 0
        pred_has_lesion = pred_voxels > VOXEL_THRESHOLD
        metric_include_hemo_dice = bool(
            gt_has_lesion and ((not hemo_dice_class_filter) or class_hemo)
        )
        excluded_reason = ""
        if gt_has_lesion and hemo_dice_class_filter and not class_hemo:
            excluded_reason = "class_nonhemo_mask_positive"

        return {
            'case_id': case_id,
            'mask_path': str(mask_path) if mask_path is not None else '',
            'gt_source': gt_source,
            'class_label': class_label,
            'subclass_label': subclass_label,
            'class_hemo': class_hemo,
            'metric_include_hemo_dice': metric_include_hemo_dice,
            'excluded_from_hemo_dice_reason': excluded_reason,
            'dice': round(dice, 4) if not np.isnan(dice) else 'N/A',
            'pixel_sensitivity': round(pixel_sens, 4),
            'hd95_mm': round(hd95, 2) if not np.isnan(hd95) else 'N/A',
            'lesion_sensitivity': round(lesion_sens, 4) if not np.isnan(lesion_sens) else 'N/A',
            'gt_lesion_count': gt_lesion_count,
            'hit_lesion_count': hit_lesion_count,
            'gt_has_lesion': gt_has_lesion,
            'pred_has_lesion': pred_has_lesion,
            'gt_voxels': gt_voxels,
            'pred_voxels': pred_voxels,
            'tp_voxels': tp_voxels,
            'fn_voxels': fn_voxels,
            'fp_voxels': fp_voxels,
        }
    except Exception as e:
        print(f"\n[ERROR] Failed on case {case_id}: {str(e)}")
        return None

# ===================================================================
# Main
# ===================================================================
def main(args):
    pred_root = Path(args.pred_root)
    output_csv = Path(args.output_csv) if args.output_csv else pred_root / "metrics_report.csv"

    # Load CSV/Excel for GT paths
    csv_path = resolve_required_input_path(args.csv_path, "CSV/Excel label file")
    df = read_eval_table(csv_path)
    case_id_col = detect_case_id_column(df, args.id_col)

    test_mask_dir = resolve_existing_path(args.test_mask_dir) or Path(args.test_mask_dir)
    test_image_dir = resolve_existing_path(args.test_image_dir) or Path(args.test_image_dir)
    mask_search_dirs = [test_mask_dir]
    if args.include_legacy_mask_dirs:
        mask_base = Path('/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/FUdata')
        mask_search_dirs.extend([
            resolve_existing_path(mask_base / 'hemo_masks' / 'thick_th0.56') or mask_base / 'hemo_masks' / 'thick_th0.56',
            resolve_existing_path(mask_base / 'hemo_masks' / 'thin_th0.56') or mask_base / 'hemo_masks' / 'thin_th0.56',
            resolve_existing_path(mask_base / 'normal_masks') or mask_base / 'normal_masks',
            resolve_existing_path(mask_base / 'mask') or mask_base / 'mask',
        ])

    print(f"[INFO] Total cases in CSV: {len(df)}")
    print(f"[INFO] CSV path: {csv_path}")
    print(f"[INFO] Case ID column: {case_id_col}")
    print(f"[INFO] Pred root: {pred_root}")
    print(f"[INFO] Test mask dir: {test_mask_dir}")
    if args.require_image_mask_pair:
        print(f"[INFO] Requiring image/mask pairs with image dir: {test_image_dir}")
    if args.hemo_dice_class_filter:
        if 'class' not in df.columns:
            print("[WARN] --hemo_dice_class_filter was set, but CSV has no 'class' column. No cases will pass the class-hemo filter.")
        else:
            print("[INFO] Hemo Dice filter: GT mask-positive AND class label is hemo/trauma.")
    if args.exclude_class_hemo_without_mask_volume:
        print("[INFO] Excluding class-hemo cases whose GT mask is missing or zero-volume.")
    if args.treat_missing_normal_as_empty:
            print("[INFO] Missing Normal GT masks will be treated as empty masks.")

    tasks = []
    missing_image_cases = []
    missing_mask_cases = []
    synthetic_empty_cases = []
    excluded_missing_class_hemo_cases = []
    for _, row in df.iterrows():
        case_id = str(row[case_id_col]).strip()

        class_label = str(row['class']).strip() if 'class' in row and pd.notna(row['class']) else ""
        subclass_label = str(row['subclass']).strip() if 'subclass' in row and pd.notna(row['subclass']) else ""
        class_hemo = is_hemo_label(class_label)
        class_normal = is_normal_label(class_label)

        if args.require_image_mask_pair:
            explicit_image_path = None
            if 'image_path' in row and pd.notna(row['image_path']):
                explicit_image_path = str(row['image_path']).strip()
            image_path = resolve_existing_path(explicit_image_path)
            if image_path is None or not image_path.exists():
                image_path = resolve_existing_path(test_image_dir / f"{case_id}.nii.gz")
            if image_path is None or not image_path.exists():
                missing_image_cases.append(case_id)
                continue

        explicit_mask_path = None
        if 'mask_path' in row and pd.notna(row['mask_path']):
            explicit_mask_path = str(row['mask_path']).strip()

        mask_path = resolve_case_mask_path(case_id, explicit_mask_path, mask_search_dirs)

        if mask_path is None or not mask_path.exists():
            if args.exclude_class_hemo_without_mask_volume and class_hemo:
                excluded_missing_class_hemo_cases.append({
                    'case_id': case_id,
                    'mask_path': '',
                    'gt_source': 'missing_mask',
                    'class_label': class_label,
                    'subclass_label': subclass_label,
                    'class_hemo': class_hemo,
                    'gt_has_lesion': False,
                    'gt_voxels': 0,
                    'excluded_reason': 'class_hemo_missing_mask',
                })
                continue
            if args.treat_missing_normal_as_empty and class_normal:
                synthetic_empty_cases.append(case_id)
                tasks.append((
                    case_id,
                    None,
                    pred_root,
                    class_label,
                    subclass_label,
                    class_hemo,
                    args.hemo_dice_class_filter,
                    True,
                ))
                continue
            missing_mask_cases.append(case_id)
            continue

        tasks.append((
            case_id,
            mask_path,
            pred_root,
            class_label,
            subclass_label,
            class_hemo,
            args.hemo_dice_class_filter,
            False,
        ))

    if missing_mask_cases:
        print(
            f"[WARN] Missing GT masks: {len(missing_mask_cases)} cases. "
            f"These cases will be excluded from metric calculation."
        )
        print(f"[WARN] Missing mask samples: {missing_mask_cases[:10]}")
    if synthetic_empty_cases:
        print(
            f"[INFO] Missing Normal GT masks treated as empty: "
            f"{len(synthetic_empty_cases)} cases."
        )
        print(f"[INFO] Synthetic empty samples: {synthetic_empty_cases[:10]}")
    if missing_image_cases:
        print(
            f"[WARN] Missing source images: {len(missing_image_cases)} cases. "
            f"These cases will be excluded because --require_image_mask_pair is enabled."
        )
        print(f"[WARN] Missing image samples: {missing_image_cases[:10]}")
    if excluded_missing_class_hemo_cases:
        print(
            f"[INFO] Class-hemo missing GT masks excluded before metric: "
            f"{len(excluded_missing_class_hemo_cases)} cases."
        )
        print(
            "[INFO] Excluded missing hemo samples: "
            f"{[item['case_id'] for item in excluded_missing_class_hemo_cases[:10]]}"
        )

    results = []
    skipped = 0

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    try:
        available_cores = len(os.sched_getaffinity(0))
    except AttributeError:
        available_cores = multiprocessing.cpu_count()
        
    num_workers = max(1, available_cores - 2)
    num_workers = min(num_workers, 24)  # NAS I/O 병목 및 행(Hang) 예방용 제한
    print(f"[INFO] 🚀 CPU detected. Using {num_workers} CPU cores to balance Compute and NAS I/O...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_case, task): task[0] for task in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Computing Metrics"):
            try:
                res = future.result()
                if res is None:
                    skipped += 1
                else:
                    results.append(res)
            except Exception as exc:
                case_id = futures[future]
                print(f"\n[FATAL ERROR] Process crashed on case {case_id}: {exc}")
                skipped += 1

    if not results:
        print("[WARNING] No metrics were calculated. Everything was skipped.")
        return

    # Build DataFrame
    results_df = pd.DataFrame(results)
    if not results_df.empty and 'case_id' in results_df.columns:
        results_df = results_df.sort_values(by='case_id').reset_index(drop=True)

    excluded_metric_df = pd.DataFrame(excluded_missing_class_hemo_cases)
    if args.exclude_class_hemo_without_mask_volume and not results_df.empty:
        zero_volume_hemo_mask = (
            (results_df['class_hemo'] == True)
            & (results_df['gt_has_lesion'] == False)
        )
        zero_volume_hemo_df = results_df[zero_volume_hemo_mask].copy()
        if not zero_volume_hemo_df.empty:
            zero_volume_hemo_df['excluded_reason'] = 'class_hemo_zero_volume_mask'
            excluded_metric_df = pd.concat(
                [excluded_metric_df, zero_volume_hemo_df],
                ignore_index=True,
                sort=False,
            )
            results_df = results_df[~zero_volume_hemo_mask].reset_index(drop=True)
            results = results_df.to_dict('records')

    if not excluded_metric_df.empty:
        excluded_csv = output_csv.with_name(output_csv.stem + "_excluded_cases.csv")
        excluded_metric_df = excluded_metric_df.sort_values(by='case_id').reset_index(drop=True)
        excluded_metric_df.to_csv(excluded_csv, index=False, encoding='utf-8-sig')
        print(f"[INFO] Excluded metric cases saved to: {excluded_csv}")

    # ===== Summary Statistics =====
    print("\n" + "=" * 60)
    print("📊 METRICS REPORT")
    print("=" * 60)
    print(f"Total Processed: {len(results)}")
    print(f"Skipped (missing files): {skipped}")
    if not excluded_metric_df.empty:
        print(f"Excluded class-hemo missing/zero GT masks: {len(excluded_metric_df)}")

    if len(results) > 0:
        lesion_df = results_df[results_df['gt_has_lesion'] == True]
        hemo_metric_df = results_df[results_df['metric_include_hemo_dice'] == True]
        excluded_hemo_df = results_df[results_df['excluded_from_hemo_dice_reason'] != ""]
        normal_df = results_df[results_df['gt_has_lesion'] == False]

        # =============================================
        # 1. Dice & HD95 & Lesion-wise (Hemorrhage Cases ONLY)
        # =============================================
        print(f"\n{'─'*60}")
        print(f"🩸 Lesion-level Metrics — Hemo Dice Set ({len(hemo_metric_df)} cases)")
        print(f"{'─'*60}")
        if args.hemo_dice_class_filter:
            print(f"  Mask-positive GT cases: {len(lesion_df)}")
            print(f"  Excluded by non-hemo class label: {len(excluded_hemo_df)}")
        if len(hemo_metric_df) > 0:
            valid_dice = hemo_metric_df[hemo_metric_df['dice'] != 'N/A']['dice'].astype(float)
            print(f"  Mean Dice:          {valid_dice.mean():.4f} ± {valid_dice.std():.4f}")
            
            hd95_valid = hemo_metric_df[hemo_metric_df['hd95_mm'] != 'N/A']['hd95_mm'].astype(float)
            if len(hd95_valid) > 0:
                print(f"  Mean HD95:          {hd95_valid.mean():.2f} ± {hd95_valid.std():.2f} mm  (valid: {len(hd95_valid)} cases)")
            else:
                print(f"  Mean HD95:          N/A (no valid pairs)")

            # [추가] 병변 단위 검출률 레포트
            valid_lesion_sens = hemo_metric_df[hemo_metric_df['lesion_sensitivity'] != 'N/A']['lesion_sensitivity'].astype(float)
            total_gt_lesions = hemo_metric_df['gt_lesion_count'].sum()
            total_hit_lesions = hemo_metric_df['hit_lesion_count'].sum()
            print(f"  Lesion-wise Recall: {valid_lesion_sens.mean():.4f} ± {valid_lesion_sens.std():.4f}")
            print(f"  Total Hemorrhages:  {total_gt_lesions} (Detected: {total_hit_lesions})")
        else:
            print(f"  No hemorrhage cases found.")

        # =============================================
        # 2. Sensitivity — Pixel-wise (ALL Cases)
        # =============================================
        total_tp = results_df['tp_voxels'].sum()
        total_fn = results_df['fn_voxels'].sum()
        total_fp = results_df['fp_voxels'].sum()
        pixel_sens = total_tp / (total_tp + total_fn + 1e-8)
        pixel_prec = total_tp / (total_tp + total_fp + 1e-8)

        print(f"\n{'─'*60}")
        print(f"🔬 Pixel-wise Sensitivity — All Cases ({len(results)} cases)")
        print(f"{'─'*60}")
        print(f"  Total TP voxels:  {total_tp:,}")
        print(f"  Total FN voxels:  {total_fn:,}")
        print(f"  Total FP voxels:  {total_fp:,}")
        print(f"  Pixel Sensitivity (Recall): {pixel_sens:.4f}")
        print(f"  Pixel Precision:            {pixel_prec:.4f}")

        # =============================================
        # 3. Sensitivity — Patient-wise (ALL Cases)
        # =============================================
        tp_patient = len(results_df[(results_df['gt_has_lesion'] == True) & (results_df['pred_has_lesion'] == True)])
        fn_patient = len(results_df[(results_df['gt_has_lesion'] == True) & (results_df['pred_has_lesion'] == False)])
        fp_patient = len(results_df[(results_df['gt_has_lesion'] == False) & (results_df['pred_has_lesion'] == True)])
        tn_patient = len(results_df[(results_df['gt_has_lesion'] == False) & (results_df['pred_has_lesion'] == False)])

        patient_sens = tp_patient / (tp_patient + fn_patient + 1e-8)
        patient_spec = tn_patient / (tn_patient + fp_patient + 1e-8)
        patient_prec = tp_patient / (tp_patient + fp_patient + 1e-8)
        patient_acc = (tp_patient + tn_patient) / len(results)
        patient_fn_ratio = fn_patient / len(lesion_df) if len(lesion_df) > 0 else 0

        print(f"\n{'─'*60}")
        print(f"🧑‍⚕️ Patient-wise (Case-wise) — All Cases ({len(results)} cases)")
        print(f"{'─'*60}")
        print(f"  Confusion Matrix:")
        print(f"    TP (출혈 정탐): {tp_patient:>5}   FN (출혈 놓침): {fn_patient:>5}")
        print(f"    FP (정상 오탐): {fp_patient:>5}   TN (정상 맞춤): {tn_patient:>5}")
        print(f"  Patient Sensitivity (Recall):  {patient_sens:.4f}")
        print(f"  Patient FN Ratio (놓침 비율):  {patient_fn_ratio:.4f}")
        print(f"  Patient Specificity:           {patient_spec:.4f}")
        print(f"  Patient Precision (PPV):       {patient_prec:.4f}")
        print(f"  Patient Accuracy:              {patient_acc:.4f}")

    print("=" * 60)

    # Save CSV
    results_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n💾 Saved per-case metrics to: {output_csv}")


if __name__ == "__main__":
    import multiprocessing
    # Linux fork() 간 자식 프로세스 교착상태(Deadlock) 방지용 spawn 세팅
    multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description="Compute Dice/Sensitivity/HD95 from saved NIfTI predictions")
    parser.add_argument("--pred_root", type=str, required=True,
                        help="Root directory containing inference results (each subfolder = case_id)")
    parser.add_argument("--csv_path", type=str, default=str(DEFAULT_CSV_PATH),
                        help="Path to test CSV/Excel with 'mask_path' and '영상일련번호ID' columns")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Output CSV path (default: pred_root/metrics_report.csv)")
    parser.add_argument("--id_col", type=str, default=None,
                        help="Case ID column name (default: auto-detect, usually '영상일련번호ID')")
    parser.add_argument("--test_mask_dir", type=str, default=str(DEFAULT_TEST_MASK_DIR),
                        help="Fallback FUtest_data mask directory")
    parser.add_argument("--test_image_dir", type=str, default=str(DEFAULT_TEST_IMAGE_DIR),
                        help="Fallback FUtest_data image directory used for pair validation")
    parser.set_defaults(require_image_mask_pair=True)
    parser.add_argument("--require_image_mask_pair", dest="require_image_mask_pair", action="store_true",
                        help="Require both source image and GT mask to exist before metric calculation")
    parser.add_argument("--no-require_image_mask_pair", dest="require_image_mask_pair", action="store_false",
                        help="Do not require source image existence before metric calculation")
    parser.add_argument("--include_legacy_mask_dirs", action="store_true",
                        help="Also search legacy FUdata mask folders if mask_path/FUtest_data mask is missing")
    parser.add_argument("--hemo_dice_class_filter", action="store_true",
                        help="Compute hemo Dice on GT mask-positive cases whose class label is hemo/trauma; exclude mask-positive non-hemo labels from hemo Dice.")
    parser.add_argument("--exclude_class_hemo_without_mask_volume", action="store_true",
                        help="Exclude class-hemo cases when the GT mask is missing or has zero foreground voxels.")
    parser.add_argument("--treat_missing_normal_as_empty", action="store_true",
                        help="Treat missing GT masks for class Normal cases as empty masks without writing mask files.")
    args = parser.parse_args()
    main(args)
