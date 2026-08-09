import os, random, re
from pathlib import Path
import pandas as pd
import numpy as np
import ast
from types import SimpleNamespace
from torch.utils.data import Dataset
from batchgenerators.dataloading.data_loader import DataLoader
try:
    from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
        NonDetMultiThreadedAugmenter,
    )
except ImportError:
    from batchgenerators.dataloading.multi_threaded_augmenter import (
        MultiThreadedAugmenter as NonDetMultiThreadedAugmenter,
    )
import SimpleITK as sitk
import nibabel as nib
from scipy.ndimage import label as connected_component_label
# NIfTI 헤더의 비직교 direction cosine 허용 (일부 마스크 파일 호환)
sitk.ProcessObject.SetGlobalDefaultDirectionTolerance(1e-3)
import torch
from threadpoolctl import threadpool_limits
from data.transform_nnunet import get_training_transforms, get_validation_transforms
from utils.mask_paths import find_case_mask_path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union


DEFAULT_IMAGE_PATH_REWRITE_FROM = "/mnt/nas100/Brain_ER/data/BrainCT_NIfTIv2"
DEFAULT_IMAGE_PATH_REWRITE_TO = "/mnt/nas100/Brain_ER/IDs/kevin/BrainCT_NIfTIv2"

CASE_ID_COLUMN = "영상일련번호ID"
SAFE_TEXT_COLUMNS = ("extracted_cc", "chief_complaint")
DICOM_NUMERIC_COLUMNS = (
    "KVP",
    "PixelSpacingX",
    "PixelSpacingY",
    "SliceThickness",
    "XRayTubeCurrent",
)
DICOM_CATEGORICAL_COLUMNS = ("Manufacturer", "ConvolutionKernel")
TRAUMA_PATTERN = re.compile(
    r"\b(trauma|fall|fell|mva|mvc|accident|collision|assault|hit)\b|"
    r"(외상|낙상|넘어|교통사고|충돌|부딪|맞아서)",
    flags=re.IGNORECASE,
)


def _case_id_column(df: pd.DataFrame) -> str:
    if CASE_ID_COLUMN in df.columns:
        return CASE_ID_COLUMN
    for candidate in ("case_id", "CaseID", "id"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Missing case-id column '{CASE_ID_COLUMN}'. Available columns: {list(df.columns)}"
    )


def read_dataset_table(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    last_error = None
    for encoding in ("utf-8-sig", "cp949", "euc-kr", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"Could not decode dataset table: {path}") from last_error


def _strict_bool_series(series: pd.Series, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }
    unknown = ~normalized.isin(mapping)
    if unknown.any():
        examples = sorted(normalized[unknown].unique().tolist())[:5]
        raise ValueError(
            f"Vision manifest column '{column}' has invalid booleans: {examples}"
        )
    return normalized.map(mapping).astype(bool)


def _clean_text(value) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip()
    return "" if value.lower() in {"", "nan", "none", "null"} else value


def build_safe_clinical_prompt(
    df: pd.DataFrame,
    text_columns: Sequence[str] = SAFE_TEXT_COLUMNS,
    dicom_prompt_mode: str = "none",
) -> Dict[str, str]:
    """Build prompts from explicitly allowed clinical and DICOM columns.

    Demographics, reports, labels and refined EMR are always excluded. DICOM
    serialization is opt-in so text-DICOM and FiLM-DICOM remain separable.
    """
    forbidden = {
        "검사결과본문",
        "검사결과결론",
        "History\n(판독문)",
        "초진기록지(EMR)",
        "refined_emr_v3",
        "class",
        "subclass",
    }
    requested = tuple(text_columns)
    invalid = set(requested) - set(SAFE_TEXT_COLUMNS)
    if invalid or set(requested) & forbidden:
        raise ValueError(
            f"Unsafe text columns requested: {sorted(invalid | (set(requested) & forbidden))}. "
            f"Allowed columns are: {SAFE_TEXT_COLUMNS}"
        )

    dicom_fields = _dicom_fields_for_mode(dicom_prompt_mode)
    id_col = _case_id_column(df)
    prompts: Dict[str, str] = {}
    for _, row in df.iterrows():
        parts = []
        for column in requested:
            if column not in df.columns:
                continue
            value = _clean_text(row[column])
            if value:
                label = "Extracted CC" if column == "extracted_cc" else "Chief complaint"
                parts.append(f"{label}: {value}")
        dicom_parts = _safe_dicom_prompt_parts(row, dicom_fields)
        if dicom_parts:
            parts.append("DICOM: " + ", ".join(dicom_parts))
        case_id = str(row[id_col]).strip()
        prompts[case_id] = ("; ".join(parts) + " <SEG>").strip() if parts else "<SEG>"
    return prompts


def _parse_float(value) -> float:
    try:
        parsed = float(value)
        return parsed if np.isfinite(parsed) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _parse_pixel_spacing(value) -> Tuple[float, float]:
    values = _as_list(value)
    parsed = [_parse_float(v) for v in values]
    parsed = [v for v in parsed if np.isfinite(v)]
    if not parsed:
        return np.nan, np.nan
    if len(parsed) == 1:
        return parsed[0], parsed[0]
    return parsed[0], parsed[1]


def _dicom_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    spacing = df.get("PixelSpacing", pd.Series(np.nan, index=df.index)).apply(
        _parse_pixel_spacing
    )
    return pd.DataFrame(
        {
            "KVP": df.get("KVP", pd.Series(np.nan, index=df.index)).map(_parse_float),
            "PixelSpacingX": spacing.map(lambda x: x[0]),
            "PixelSpacingY": spacing.map(lambda x: x[1]),
            "SliceThickness": df.get(
                "SliceThickness", pd.Series(np.nan, index=df.index)
            ).map(_parse_float),
            "XRayTubeCurrent": df.get(
                "XRayTubeCurrent", pd.Series(np.nan, index=df.index)
            ).map(_parse_float),
        },
        index=df.index,
    )


def fit_dicom_schema(df: pd.DataFrame) -> Dict[str, object]:
    """Fit normalization/category mappings on the training split only."""
    numeric = _dicom_numeric_frame(df)
    means = numeric.mean(skipna=True).fillna(0.0)
    stds = numeric.std(skipna=True).replace(0.0, np.nan).fillna(1.0)
    categories = {}
    for column in DICOM_CATEGORICAL_COLUMNS:
        values = sorted(
            {
                _clean_text(value)
                for value in df.get(column, pd.Series(dtype=object)).tolist()
                if _clean_text(value)
            }
        )
        categories[column] = {"<UNK>": 0, "<MISSING>": 1, **{
            value: index + 2 for index, value in enumerate(values)
        }}
    return {
        "numeric_columns": list(DICOM_NUMERIC_COLUMNS),
        "numeric_mean": means.to_dict(),
        "numeric_std": stds.to_dict(),
        "categorical_columns": list(DICOM_CATEGORICAL_COLUMNS),
        "categories": categories,
    }


def encode_dicom_row(row: pd.Series, schema: Dict[str, object]):
    numeric_frame = _dicom_numeric_frame(pd.DataFrame([row]))
    numeric_values = numeric_frame.iloc[0]
    missing = numeric_values.isna().to_numpy(dtype=np.float32)
    means = pd.Series(schema["numeric_mean"])
    stds = pd.Series(schema["numeric_std"])
    normalized = (
        (numeric_values.fillna(means) - means) / stds
    ).to_numpy(dtype=np.float32)
    numeric_with_missing = np.concatenate([normalized, missing]).astype(np.float32)

    category_ids = []
    for column in schema["categorical_columns"]:
        value = _clean_text(row[column]) if column in row else ""
        mapping = schema["categories"][column]
        category_ids.append(mapping.get(value, 1 if not value else 0))
    return numeric_with_missing, np.asarray(category_ids, dtype=np.int64)


def encode_dicom_frame(df: pd.DataFrame, schema: Dict[str, object]):
    numeric = _dicom_numeric_frame(df)
    missing = numeric.isna().to_numpy(dtype=np.float32)
    means = pd.Series(schema["numeric_mean"])
    stds = pd.Series(schema["numeric_std"])
    normalized = ((numeric.fillna(means) - means) / stds).to_numpy(
        dtype=np.float32
    )
    numeric_with_missing = np.concatenate([normalized, missing], axis=1)

    categorical_columns = []
    for column in schema["categorical_columns"]:
        mapping = schema["categories"][column]
        values = df.get(column, pd.Series("", index=df.index)).map(_clean_text)
        categorical_columns.append(
            values.map(lambda value: mapping.get(value, 1 if not value else 0))
            .to_numpy(dtype=np.int64)
        )
    categorical = np.stack(categorical_columns, axis=1)
    return numeric_with_missing, categorical


DICOM_PROMPT_FIELD_MODES = {
    "full": {
        "manufacturer",
        "kernel",
        "slice_thickness",
        "kvp",
        "tube_current",
        "pixel_spacing",
    },
    "extended": {
        "manufacturer",
        "kernel",
        "slice_thickness",
        "kvp",
        "tube_current",
        "pixel_spacing",
        "spacing_between_slices",
        "series_description",
        "contrast",
    },
    "limited": {"kernel", "slice_thickness", "pixel_spacing"},
    "geometry": {"slice_thickness", "pixel_spacing", "spacing_between_slices"},
    "kernel_only": {"kernel"},
    "spacing_only": {"slice_thickness", "pixel_spacing", "spacing_between_slices"},
    "scanner_only": {"manufacturer", "kvp", "tube_current"},
    "protocol_only": {"kernel", "series_description", "contrast", "kvp"},
    "none": set(),
}


def _clean_row_value(row, columns):
    if columns is None:
        return None
    if isinstance(columns, str):
        columns = [columns]
    for col in columns:
        if not col or col not in row or pd.isna(row[col]):
            continue
        value = str(row[col]).strip()
        if value and value.lower() not in {"nan", "none", "null"}:
            return value
    return None


def _dicom_fields_for_mode(mode):
    mode = str(mode or "full").strip().lower()
    if mode not in DICOM_PROMPT_FIELD_MODES:
        valid = ", ".join(sorted(DICOM_PROMPT_FIELD_MODES))
        raise ValueError(f"Unknown dicom_prompt_mode='{mode}'. Valid modes: {valid}")
    return DICOM_PROMPT_FIELD_MODES[mode]


def _safe_dicom_prompt_parts(row, dicom_fields):
    """Serialize only the allow-listed acquisition fields for LLM input."""
    field_specs = (
        ("manufacturer", "Manufacturer", "Manufacturer"),
        ("kernel", ("ConvolutionKernel", "ReconstructionKernel"), "kernel"),
        ("slice_thickness", "SliceThickness", "thickness_mm"),
        ("kvp", "KVP", "KVP"),
        ("tube_current", "XRayTubeCurrent", "XRayTubeCurrent"),
        ("pixel_spacing", "PixelSpacing", "PixelSpacing"),
        ("spacing_between_slices", "SpacingBetweenSlices", "SpacingBetweenSlices"),
        ("series_description", "SeriesDescription", "SeriesDescription"),
        ("contrast", ("Contrast", "ContrastBolusAgent"), "Contrast"),
    )
    parts = []
    for field, columns, label in field_specs:
        if field not in dicom_fields:
            continue
        value = _clean_row_value(row, columns)
        if value:
            parts.append(f"{label}={value}")
    return parts


def read_dicom_volume(dicom_dir: Path, series_uid: Optional[str] = None) -> Tuple[np.ndarray, sitk.Image]:
    """
    dicom_dir 안의 DICOM series를 읽어서 (D,H,W) numpy + reference sitk.Image 반환.
    - series_uid가 주어지면 그 UID로 고정해서 읽음 (CSV의 SeriesInstanceUID 활용 가능)
    - series_uid가 없으면: 가장 파일 수가 많은 series를 선택
    """
    dicom_dir = Path(dicom_dir)
    reader = sitk.ImageSeriesReader()

    uids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if not uids:
        raise FileNotFoundError(f"No DICOM series found in: {dicom_dir}")

    if series_uid is not None and str(series_uid).strip() in uids:
        use_uid = str(series_uid).strip()
    else:
        # 파일 수 가장 많은 series 선택 (가장 흔한 heuristic)
        counts = []
        for uid in uids:
            files = reader.GetGDCMSeriesFileNames(str(dicom_dir), uid)
            counts.append((len(files), uid))
        counts.sort(reverse=True)
        use_uid = counts[0][1]

    files = reader.GetGDCMSeriesFileNames(str(dicom_dir), use_uid)
    reader.SetFileNames(files)
    img_itk = reader.Execute()                 # reference geometry 포함
    arr = sitk.GetArrayFromImage(img_itk)      # (D,H,W)
    return arr, img_itk


def read_nifti_mask_resample_to_ref(mask_path: Path, ref_img: sitk.Image) -> np.ndarray:
    """
    mask NIfTI를 읽고 ref_img의 grid/spacing/origin/direction에 맞춰 nearest로 Resample.
    => 결과 shape이 image와 동일해짐.
    """
    mask_itk = sitk.ReadImage(str(mask_path))
    # ref_img로 resample (label이므로 NN)
    mask_rs = sitk.Resample(
        mask_itk,
        ref_img,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        mask_itk.GetPixelID(),
    )
    mask_arr = sitk.GetArrayFromImage(mask_rs)  # (D,H,W)
    return mask_arr


def windowing(img, window_center, window_width):
    lower_bound = window_center - window_width / 2
    upper_bound = window_center + window_width / 2
    img = np.clip(img, lower_bound, upper_bound)
    img = (img - lower_bound) / (upper_bound - lower_bound)
    return img

def windowing_3ch(img):    
    return np.stack([
        windowing(img, 40, 80),
        windowing(img, 50, 150),
        windowing(img, 600, 2000)],
        axis=0
        )

def get_valid_center_with_jitter(base_center, patch_size, image_shape, jitter_range=0.25):
    """
    bleeding 위치 주변에서 약간의 random offset을 준 center 계산 - 배치를 뽑아올때 무작위성 추가
    """
    center = []
    for i in range(3):
        # jitter 범위 계산 (patch_size의 일정 비율)
        jitter = int(patch_size[i] * jitter_range)
        
        # 최소/최대 허용 범위 계산
        min_valid = patch_size[i] // 2
        max_valid = image_shape[i] - patch_size[i] // 2
        
        # base_center에 random offset 추가
        offset = random.randint(-jitter, jitter)
        pos = base_center[i] + offset
        
        # 범위 체크 및 조정
        if pos < min_valid:
            center.append(min_valid)
        elif pos > max_valid:
            center.append(max_valid)
        else:
            center.append(pos)
            
    return center

def extract_patches(image, mask, 
                    sample_positive=True, 
                    patch_size=(32, 256, 256)):
    
    # Ensure image and mask are the same shape
    assert image.shape == mask.shape, "Image and mask shapes do not match before padding."

    # Calculate how much padding is needed
    need_to_pad = [max(0, patch_size[i] - image.shape[i]) for i in range(3)]
    pad_width = [(0, need_to_pad[i]) for i in range(3)]
    
    # Apply the same padding to both image and mask
    image = np.pad(image, pad_width, mode='constant', constant_values=-1024)
    mask = np.pad(mask, pad_width, mode='constant', constant_values=0)

    # Sample connected components uniformly so a large hematoma does not make
    # punctate hemorrhages almost impossible to select.
    if not sample_positive:
        bleeding_indices = []
    else:
        component_map, component_count = connected_component_label(mask > 0)
        if component_count == 0:
            bleeding_indices = []
        else:
            component_id = random.randint(1, int(component_count))
            bleeding_indices = np.argwhere(component_map == component_id)

    if sample_positive and len(bleeding_indices) > 0:
        base_center = random.choice(bleeding_indices)
        center = get_valid_center_with_jitter(base_center, patch_size, image.shape)

        # Calculate the start and end of the patch
        start = [center[i] - patch_size[i] // 2 for i in range(3)]
        end = [start[i] + patch_size[i] for i in range(3)]
    else:
        # For non-positive patches, ensure the center stays within bounds
        center = [np.random.randint(patch_size[i] // 2, (image.shape[i] - patch_size[i] // 2 )+1) for i in range(3)]

        # Calculate the start and end of the patch
        start = [center[i] - patch_size[i] // 2 for i in range(3)]
        end = [start[i] + patch_size[i] for i in range(3)]
            
    # Extract the patch from the image and mask
    patch_image = image[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    patch_mask = mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

    return patch_image, patch_mask


def _pad_to_shape(arr: np.ndarray, target_shape, pad_value=0):
    """
    arr: (D,H,W)
    target_shape: (D,H,W)
    center padding (symmetric) to match target_shape.
    """
    assert arr.ndim == 3
    pads = []
    for cur, tgt in zip(arr.shape, target_shape):
        if cur >= tgt:
            pads.append((0, 0))
        else:
            diff = tgt - cur
            p0 = diff // 2
            p1 = diff - p0
            pads.append((p0, p1))
    return np.pad(arr, pads, mode="constant", constant_values=pad_value)

def match_img_mask_shape(img: np.ndarray, mask: np.ndarray,
                         pad_value_img=-1024, pad_value_mask=0):
    """
    img/mask shape이 다르면, 두 배열을 (각 축별 max) shape으로 center-padding하여 맞춘다.
    -> mask가 더 크면 img를 pad해서 mask 크기에 맞춰짐.
    """
    assert img.ndim == 3 and mask.ndim == 3
    if img.shape == mask.shape:
        return img, mask

    target = tuple(max(i, m) for i, m in zip(img.shape, mask.shape))
    img2  = _pad_to_shape(img,  target, pad_value=pad_value_img)
    mask2 = _pad_to_shape(mask, target, pad_value=pad_value_mask)
    return img2, mask2


def _as_list(x):
    """
    '[0.41, 0.41]' 같은 문자열/리스트/스칼라를 float 리스트로 변환.
    실패하면 원본을 단일 원소 리스트로 반환.
    """
    try:
        if isinstance(x, (list, tuple, np.ndarray)):
            return [float(v) for v in x]
        if isinstance(x, str):
            s = x.strip()
            if s.startswith('[') and s.endswith(']'):
                vals = ast.literal_eval(s)
                return [float(v) for v in vals]
            if ',' in s:
                return [float(v) for v in s.replace('[','').replace(']','').split(',')]
            return [float(s)]
        return [float(x)]
    except Exception:
        return [x]
    


def _build_prompt_legacy(df, args):
    id_col = getattr(args, "csv_id_col", "영상일련번호ID")

    col_manuf  = getattr(args, "csv_manufacturer_col", "Manufacturer")
    col_kernel = getattr(args, "csv_kernel_col", "ConvolutionKernel")
    col_kvp    = getattr(args, "csv_kvp_col", "KVP")
    # col_slope  = getattr(args, "csv_slope_col", "RescaleSlope")
    # col_inter  = getattr(args, "csv_intercept_col", "RescaleIntercept")
    # col_space  = getattr(args, "csv_spacing_col", "PixelSpacingRow")
    col_thick  = getattr(args, "csv_thickness_col", "SliceThickness")
    col_tube   = getattr(args, "csv_tubecurrent_col", "XRayTubeCurrent")
    col_pxsp   = getattr(args, "csv_pixelspacing_col", "PixelSpacing")

    col_age    = getattr(args, "csv_ptnage_col", "검사나이")
    col_sex    = getattr(args, "csv_ptnsex_col", "성별코드")
    col_cc     = getattr(args, "csv_cc_col", "extracted_cc")
    col_emr    = getattr(args, "csv_emr_col", "refined_emr_v3")

    report = {}
    for _, row in df.iterrows():
        case_id = str(row[id_col]).strip()

        manuf  = str(row[col_manuf]).strip()  if (col_manuf in row)  and pd.notna(row[col_manuf])  else None
        kernel = str(row[col_kernel]).strip() if (col_kernel in row) and pd.notna(row[col_kernel]) else None
        kvp    = str(row[col_kvp]).strip()    if (col_kvp in row)    and pd.notna(row[col_kvp])    else None
        # slope  = str(row[col_slope]).strip()  if (col_slope in row)  and pd.notna(row[col_slope])  else None
        # inter  = str(row[col_inter]).strip()  if (col_inter in row)  and pd.notna(row[col_inter])  else None
        tube   = str(row[col_tube]).strip()   if (col_tube in row)   and pd.notna(row[col_tube])   else None
        pxsp   = str(row[col_pxsp]).strip()   if (col_pxsp in row)   and pd.notna(row[col_pxsp])   else None

        age = str(row[col_age]).strip() if col_age and (col_age in row) and pd.notna(row[col_age]) else None
        sex = str(row[col_sex]).strip() if col_sex and (col_sex in row) and pd.notna(row[col_sex]) else None

        cc  = str(row[col_cc]).strip()  if col_cc and (col_cc in row) and pd.notna(row[col_cc]) else None
        emr = str(row[col_emr]).strip() if col_emr and (col_emr in row) and pd.notna(row[col_emr]) else None

        thickness = row[col_thick] if (col_thick in row) and pd.notna(row[col_thick]) else None
        thick_str = f"thickness={thickness}mm" if thickness is not None else None

        parts = []
        if sex:         parts.append(f"sex={sex}")
        if age:         parts.append(f"AGE={age}")
        if manuf:       parts.append(f"Manufacturer={manuf}")
        if kernel:      parts.append(f"kernel={kernel}")
        if thick_str:   parts.append(thick_str)
        if kvp:         parts.append(f"KVP={kvp}")
        # if slope:       parts.append(f"RescaleSlope={slope}")
        # if inter:       parts.append(f"RescaleIntercept={inter}")
        if tube:        parts.append(f"XRayTubeCurrent={tube}")
        if pxsp:        parts.append(f"PixelSpacing={pxsp}")
        # if spacing_str: parts.append(spacing_str)

        # CC and Clinical Information
        if cc:          parts.append(f"CC : {cc}")
        if emr:         parts.append(f"Clinical Information : {emr}")

        prompt_text = ", ".join(parts).strip()
        report[case_id] = (prompt_text + " <SEG>").strip() if prompt_text else "<SEG>"

    print("\n" + "="*50)
    print("✨ [DEBUG] Generated Text Prompts Sample (first 5) ✨")
    for k in list(report)[:5]:
        print(f"[{k}] -> {report[k]}")
    print("="*50 + "\n")

    return report


def build_prompt(df, args):
    """Build text prompts with configurable DICOM-field ablations."""
    id_col = getattr(args, "csv_id_col", None)
    if not id_col or id_col not in df.columns:
        id_col = "case_id" if "case_id" in df.columns else df.columns[0]

    custom_fields = getattr(args, "dicom_prompt_fields", None)
    if custom_fields:
        dicom_fields = {field.strip().lower() for field in str(custom_fields).split(",") if field.strip()}
    else:
        dicom_fields = _dicom_fields_for_mode(getattr(args, "dicom_prompt_mode", "full"))
    include_demographics = bool(getattr(args, "include_demographics", True))

    col_manuf = getattr(args, "csv_manufacturer_col", "Manufacturer")
    col_kernel = [
        getattr(args, "csv_kernel_col", "ConvolutionKernel"),
        getattr(args, "csv_reconstruction_kernel_col", "ReconstructionKernel"),
    ]
    col_kvp = getattr(args, "csv_kvp_col", "KVP")
    col_thick = getattr(args, "csv_thickness_col", "SliceThickness")
    col_tube = getattr(args, "csv_tubecurrent_col", "XRayTubeCurrent")
    col_pxsp = getattr(args, "csv_pixelspacing_col", "PixelSpacing")
    col_spacing_between = getattr(args, "csv_spacing_between_slices_col", "SpacingBetweenSlices")
    col_series = getattr(args, "csv_series_description_col", "SeriesDescription")
    col_contrast = [
        getattr(args, "csv_contrast_col", "Contrast"),
        getattr(args, "csv_contrast_bolus_col", "ContrastBolusAgent"),
    ]

    col_age = getattr(args, "csv_ptnage_col", None)
    col_sex = getattr(args, "csv_ptnsex_col", None)
    col_cc = getattr(args, "csv_cc_col", "extracted_cc")
    col_emr = getattr(args, "csv_emr_col", "refined_emr_v3")

    report = {}
    for _, row in df.iterrows():
        case_id = str(row[id_col]).strip()

        manuf = _clean_row_value(row, col_manuf)
        kernel = _clean_row_value(row, col_kernel)
        kvp = _clean_row_value(row, col_kvp)
        thickness = _clean_row_value(row, col_thick)
        tube = _clean_row_value(row, col_tube)
        pxsp = _clean_row_value(row, col_pxsp)
        spacing_between = _clean_row_value(row, col_spacing_between)
        series = _clean_row_value(row, col_series)
        contrast = _clean_row_value(row, col_contrast)

        age = _clean_row_value(row, col_age) if include_demographics else None
        sex = _clean_row_value(row, col_sex) if include_demographics else None
        cc = _clean_row_value(row, col_cc)
        emr = _clean_row_value(row, col_emr)

        parts = []
        if sex:
            parts.append(f"sex={sex}")
        if age:
            parts.append(f"AGE={age}")
        if "manufacturer" in dicom_fields and manuf:
            parts.append(f"Manufacturer={manuf}")
        if "kernel" in dicom_fields and kernel:
            parts.append(f"kernel={kernel}")
        if "slice_thickness" in dicom_fields and thickness:
            parts.append(f"thickness={thickness}mm")
        if "kvp" in dicom_fields and kvp:
            parts.append(f"KVP={kvp}")
        if "tube_current" in dicom_fields and tube:
            parts.append(f"XRayTubeCurrent={tube}")
        if "pixel_spacing" in dicom_fields and pxsp:
            parts.append(f"PixelSpacing={pxsp}")
        if "spacing_between_slices" in dicom_fields and spacing_between:
            parts.append(f"SpacingBetweenSlices={spacing_between}")
        if "series_description" in dicom_fields and series:
            parts.append(f"SeriesDescription={series}")
        if "contrast" in dicom_fields and contrast:
            parts.append(f"Contrast={contrast}")
        if cc:
            parts.append(f"CC : {cc}")
        if emr:
            parts.append(f"Clinical Information : {emr}")

        prompt_text = ", ".join(parts).strip()
        report[case_id] = (prompt_text + " <SEG>").strip() if prompt_text else "<SEG>"

    print("\n" + "=" * 50)
    print(
        "[DEBUG] Generated Text Prompts Sample "
        f"(mode={getattr(args, 'dicom_prompt_mode', 'full')}, "
        f"fields={','.join(sorted(dicom_fields)) or 'none'}, "
        f"include_demographics={include_demographics})"
    )
    for key in list(report)[:5]:
        print(f"[{key}] -> {report[key]}")
    print("=" * 50 + "\n")

    return report


class HemoDataset(Dataset):
    def __init__(
        self, 
        mode: str = 'train',
        deep_supervision: bool = False,
        patch_size: tuple = (16, 160, 160),
        rater: int = 1,
        include_clinical: bool = True,   
        include_findings: bool = True,   
        include_cc: bool = True,
        include_chief_complaint: bool = True,
        include_emr: bool = False,
        dicom_prompt_mode: str = "none",
        include_demographics: bool = False,
        use_dicom: bool = False,
        dicom_schema: Optional[Dict[str, object]] = None,
        exclude_case_ids: Optional[Iterable[str]] = None,
        deduplicate_case_ids: bool = True,
        vision_manifest_path: Optional[Union[str, Path]] = None,
        filter_invalid_supervision: bool = False,
        labeled_validation_only: bool = False,
        prompt_args: Optional[SimpleNamespace] = None,
        csv_path: Optional[Union[str, Path]] = None, 
    ):
        print(patch_size)
        # 1) CSV 로드 (항상 self.df를 먼저 만든 다음 print)
        if csv_path is not None:
            self.csv_path = Path(csv_path)
        else:
            if mode == "train":
                self.csv_path = Path('/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/ICH_pair/train_split.csv')
            else:
                self.csv_path = Path('/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/CSV/ICH_pair/valid_split.csv')

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Data file not found (Check extension!): {self.csv_path}")

        try:
            if self.csv_path.suffix.lower() in ['.xlsx', '.xls']:
                self.df = pd.read_excel(self.csv_path)
            else:
                for enc in ['utf-8-sig', 'cp949', 'euc-kr', 'latin-1']:
                    try:
                        self.df = pd.read_csv(self.csv_path, encoding=enc)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise RuntimeError(f"CSV 인코딩 감지 실패: {self.csv_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to read Data: {self.csv_path} ({e})") from e

        self.case_id_col = _case_id_column(self.df)
        if deduplicate_case_ids:
            duplicate_count = int(self.df.duplicated(self.case_id_col).sum())
            if duplicate_count:
                self.df = self.df.drop_duplicates(
                    self.case_id_col, keep="first"
                ).reset_index(drop=True)
                print(f"[DATA CONTRACT] {mode}: removed {duplicate_count} duplicate case rows.")
        if exclude_case_ids:
            excluded = {str(value).strip() for value in exclude_case_ids}
            keep = ~self.df[self.case_id_col].astype(str).str.strip().isin(excluded)
            overlap_count = int((~keep).sum())
            if overlap_count:
                self.df = self.df.loc[keep].reset_index(drop=True)
                print(
                    f"[DATA CONTRACT] {mode}: removed {overlap_count} cases overlapping "
                    "the validation split."
                )

        self.vision_manifest = None
        self.positive_indices: List[int] = []
        self.negative_indices: List[int] = []
        if vision_manifest_path:
            manifest_path = Path(vision_manifest_path)
            if not manifest_path.exists():
                raise FileNotFoundError(f"Vision sampling manifest not found: {manifest_path}")
            manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
            required_manifest_columns = {
                "manifest_schema_version",
                "case_id",
                "actual_foreground_voxels",
                "foreground_space",
                "supervision_status",
                "train_eligible",
                "validation_eligible",
            }
            missing_manifest_columns = required_manifest_columns - set(manifest.columns)
            if missing_manifest_columns:
                raise ValueError(
                    f"Vision manifest lacks columns: {sorted(missing_manifest_columns)}"
                )
            schema_versions = set(
                pd.to_numeric(manifest["manifest_schema_version"], errors="coerce")
                .dropna()
                .astype(int)
                .tolist()
            )
            foreground_spaces = set(
                manifest["foreground_space"].astype(str).str.strip().tolist()
            )
            if schema_versions != {2} or foreground_spaces != {"image_aligned"}:
                raise ValueError(
                    "Vision manifest must use schema v2 image-aligned foreground "
                    f"statistics; versions={schema_versions}, spaces={foreground_spaces}."
                )
            manifest["case_id"] = manifest["case_id"].astype(str).str.strip()
            if manifest["case_id"].duplicated().any():
                raise ValueError("Vision manifest contains duplicate case_id rows.")
            self.vision_manifest = manifest.set_index("case_id", drop=False)
            case_ids = self.df[self.case_id_col].astype(str).str.strip()
            missing_manifest = ~case_ids.isin(self.vision_manifest.index)
            if missing_manifest.any():
                examples = case_ids[missing_manifest].head(5).tolist()
                raise ValueError(
                    f"Vision manifest misses {int(missing_manifest.sum())} dataset cases; "
                    f"examples={examples}"
                )
            if filter_invalid_supervision:
                eligibility_column = (
                    "validation_eligible" if mode == "valid" else "train_eligible"
                )
                eligible_by_case = _strict_bool_series(
                    self.vision_manifest[eligibility_column], eligibility_column
                )
                keep = case_ids.map(eligible_by_case).fillna(False).to_numpy(dtype=bool)
                removed = int((~keep).sum())
                self.df = self.df.loc[keep].reset_index(drop=True)
                print(
                    f"[VISION MANIFEST] {mode}: removed {removed} invalid-supervision rows; "
                    f"{len(self.df)} remain."
                )
            if labeled_validation_only and mode == "valid":
                case_ids = self.df[self.case_id_col].astype(str).str.strip()
                eligible_by_case = _strict_bool_series(
                    self.vision_manifest["validation_eligible"],
                    "validation_eligible",
                )
                keep = case_ids.map(eligible_by_case).fillna(False).to_numpy(dtype=bool)
                removed = int((~keep).sum())
                self.df = self.df.loc[keep].reset_index(drop=True)
                print(
                    f"[VISION MANIFEST] valid: excluded {removed} unlabeled/inconsistent rows; "
                    f"{len(self.df)} labeled cases remain."
                )

            case_ids = self.df[self.case_id_col].astype(str).str.strip()
            foreground_by_case = self.vision_manifest["actual_foreground_voxels"].astype(int)
            foreground_voxels = case_ids.map(foreground_by_case).fillna(0).to_numpy()
            self.positive_indices = np.flatnonzero(foreground_voxels > 0).astype(int).tolist()
            self.negative_indices = np.flatnonzero(foreground_voxels == 0).astype(int).tolist()
            print(
                f"[VISION MANIFEST] {mode}: positive_pool={len(self.positive_indices)} "
                f"negative_pool={len(self.negative_indices)}"
            )

        print(f'Number of {mode} samples: {len(self.df)}')
        self.image_path_rewrite_from = os.environ.get(
            "LLMSEG_IMAGE_PATH_REWRITE_FROM", DEFAULT_IMAGE_PATH_REWRITE_FROM
        ).strip().rstrip("/\\")
        self.image_path_rewrite_to = os.environ.get(
            "LLMSEG_IMAGE_PATH_REWRITE_TO", DEFAULT_IMAGE_PATH_REWRITE_TO
        ).strip().rstrip("/\\")
        if bool(self.image_path_rewrite_from) ^ bool(self.image_path_rewrite_to):
            raise ValueError(
                "Set both LLMSEG_IMAGE_PATH_REWRITE_FROM and LLMSEG_IMAGE_PATH_REWRITE_TO, or neither."
            )
        if self.image_path_rewrite_from:
            print(
                "[INFO] image path rewrite: "
                f"{self.image_path_rewrite_from} -> {self.image_path_rewrite_to}"
            )
        if os.environ.get("LLMSEG_SKIP_MISSING_IMAGE_PATHS", "0") == "1":
            if "image_path" not in self.df.columns:
                print("[WARN] Cannot skip missing images: CSV has no image_path column.")
            else:
                keep = self.df["image_path"].apply(
                    lambda value: pd.notna(value) and self._resolve_image_path(value).exists()
                )
                skipped = int((~keep).sum())
                self.df = self.df.loc[keep].reset_index(drop=True)
                print(
                    f"[INFO] {mode}: skipped {skipped} rows with missing image files; "
                    f"{len(self.df)} samples remain."
                )
                if self.df.empty:
                    raise RuntimeError(
                        f"No {mode} samples remain after filtering missing image files: {self.csv_path}"
                    )
        if self.vision_manifest is not None:
            case_ids = self.df[self.case_id_col].astype(str).str.strip()
            foreground_by_case = self.vision_manifest[
                "actual_foreground_voxels"
            ].astype(int)
            foreground_voxels = case_ids.map(foreground_by_case).fillna(0).to_numpy()
            self.positive_indices = np.flatnonzero(
                foreground_voxels > 0
            ).astype(int).tolist()
            self.negative_indices = np.flatnonzero(
                foreground_voxels == 0
            ).astype(int).tolist()

        # 2) 데이터
        self.image_root = Path('/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/pair_data_nifti')
        
        # [수정] 모드에 따른 mask base 디렉토리 및 탐색 경로 분기 설정
        if mode == "test":
            self.mask_base = Path('/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/FUtest_data')
            self.mask_search_dirs = [
                self.mask_base / 'mask',
            ]
            print(f"[INFO] {mode} mode: 마스크 탐색 경로를 FUtest_data/mask 로 지정합니다.")
        else:
            # train 및 valid 모드는 FUdata에서 마스크를 찾습니다.
            self.mask_base  = Path('/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/FUdata')
            # 마스크 탐색 우선순위: hemo thick → hemo thin → normal
            self.mask_search_dirs = [
                self.mask_base / 'hemo_masks' / 'thick_th0.56',
                self.mask_base / 'hemo_masks' / 'thin_th0.56',
                self.mask_base / 'normal_masks',
            ]
            print(f"[INFO] {mode} mode: 마스크 탐색 경로를 FUdata 하위 폴더로 지정합니다.")


        self.deep_supervision = deep_supervision
        self.patch_size = patch_size
        self.use_dicom = bool(use_dicom)
        self.dicom_schema = (
            fit_dicom_schema(self.df)
            if self.use_dicom and dicom_schema is None
            else dicom_schema
        )
        self.dicom_numeric = None
        self.dicom_categorical = None
        if self.use_dicom:
            self.dicom_numeric, self.dicom_categorical = encode_dicom_frame(
                self.df, self.dicom_schema
            )

        # 필수 컬럼 체크
        if self.case_id_col not in self.df.columns:
            raise ValueError(f"Missing case ID column. Available columns: {list(self.df.columns)}")

        # 임베딩 프롬프트(장비/커널/spacing/thickness) 맵 준비
        # 사용자 CSV가 장비 메타 컬럼을 가진 다른 테이블일 수도 있으므로,
        # 동일 CSV에서 컬럼이 있으면 그대로 사용, 없으면 빈 프롬프트("<SEG>")로 대체.
        #csv_manufacturer_col='Manufacturer',
        if prompt_args is None:
                 # 기본 매핑: '영상일련번호ID'를 key로 사용 (ID가 FileName 역할)
             prompt_args = SimpleNamespace(
                 csv_id_col='영상일련번호ID',
                 csv_manufacturer_col='Manufacturer',
                 csv_kernel_col='ConvolutionKernel',    
                 csv_kvp_col='KVP',
                 # csv_slope_col='RescaleSlope',
                 # csv_intercept_col='RescaleIntercept',
                 # csv_spacing_col='PixelSpacingRow',
                 csv_thickness_col='SliceThickness',
                 csv_tubecurrent_col='XRayTubeCurrent',
                 csv_pixelspacing_col='PixelSpacing',
                 csv_ptnage_col='검사나이',
                 csv_ptnsex_col='성별코드',
                 csv_cc_col='extracted_cc' if include_cc else None,
                 csv_emr_col='refined_emr_v3' if include_emr else None,
                 dicom_prompt_mode=dicom_prompt_mode,
                 include_demographics=include_demographics,
             )
             if not include_emr and not include_cc:
                 print("[INFO] 🔧 DICOM-only mode: EMR/CC 텍스트 제외됨")

        requested_text_columns = []
        if include_cc:
            requested_text_columns.append("extracted_cc")
        if include_chief_complaint:
            requested_text_columns.append("chief_complaint")
        self.safe_text_columns = tuple(requested_text_columns)
        self.dicom_prompt_mode = str(dicom_prompt_mode or "none").strip().lower()
        self.prompt_map = build_safe_clinical_prompt(
            self.df,
            requested_text_columns,
            dicom_prompt_mode=self.dicom_prompt_mode,
        )
        print(
            "[DATA CONTRACT] text columns: "
            f"{requested_text_columns or 'none'}; "
            f"DICOM text mode: {self.dicom_prompt_mode}; "
            f"DICOM text fields: {sorted(_dicom_fields_for_mode(self.dicom_prompt_mode)) or 'none'}."
        )

        # === 변수명 변경: clinical/findings -> body/conclusion ===
        if include_clinical or include_findings or include_emr or include_demographics:
            print(
                "[DATA CONTRACT] report/EMR/demographic prompt flags were requested but are "
                "ignored by the safe prompt path."
            )
        self.include_body = False
        self.include_conclusion = False

        self.body_col = '검사결과본문'
        self.conclusion_col = '검사결과결론'
        self.has_body = self.body_col in self.df.columns
        self.has_conclusion = self.conclusion_col in self.df.columns

    def _resolve_image_path(self, raw_path: Union[str, Path]) -> Path:
        raw = str(raw_path).strip()
        original = Path(raw)
        if self.image_path_rewrite_from and self.image_path_rewrite_to:
            if raw == self.image_path_rewrite_from:
                raw = self.image_path_rewrite_to
            elif raw.startswith(self.image_path_rewrite_from + "/") or raw.startswith(self.image_path_rewrite_from + "\\"):
                suffix = raw[len(self.image_path_rewrite_from):].lstrip("/\\")
                raw = f"{self.image_path_rewrite_to}/{suffix}"
        rewritten = Path(raw)

        # Some spreadsheets contain a mixture of paths from both volumes.
        # Prefer the configured destination, but keep a valid original path
        # instead of turning it into a false missing-file error.
        if rewritten != original and not rewritten.exists() and original.exists():
            return original
        return rewritten

    def __len__(self):
        return len(self.df)
    @staticmethod
    def _sitk_read(path: Path):
        img_itk = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img_itk)  # (D,H,W)
        return arr
    @staticmethod
    def _sitk_read_img_and_ref(path: Path):
        img_itk = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img_itk)  # (D,H,W)
        return arr, img_itk

    @staticmethod
    def _sitk_read_mask_resample_to_ref(mask_path: Path, ref_img: sitk.Image):
        try:
            mask_itk = sitk.ReadImage(str(mask_path))
        except RuntimeError:
            # direction cosine 오류 시 nibabel로 폴백
            nib_img = nib.load(str(mask_path))
            arr = np.asarray(nib_img.dataobj)
            mask_itk = sitk.GetImageFromArray(arr)
            mask_itk.SetOrigin(ref_img.GetOrigin())
            mask_itk.SetSpacing(ref_img.GetSpacing())
            mask_itk.SetDirection(ref_img.GetDirection())
        mask_rs = sitk.Resample(
            mask_itk,
            ref_img,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,  # label
            0,
            mask_itk.GetPixelID(),
        )
        arr = sitk.GetArrayFromImage(mask_rs)  # (D,H,W)  ✅ ref와 동일 shape
        return arr

    def get_full_volume(self, idx: int) -> Dict[str, object]:
        """Load a deterministic full volume for sliding-window validation."""
        row = self.df.loc[idx]
        case_id = str(row[self.case_id_col]).strip()
        if 'image_path' in row and pd.notna(row['image_path']):
            img_path = self._resolve_image_path(row['image_path'])
        else:
            img_path = self._resolve_image_path(self.image_root / f"{case_id}.nii.gz")
        explicit_mask_path = None
        if 'mask_path' in row and pd.notna(row['mask_path']):
            explicit_mask_path = Path(str(row['mask_path']).strip())
        mask_path = find_case_mask_path(case_id, explicit_mask_path, self.mask_search_dirs)
        if not img_path.exists():
            raise FileNotFoundError(f"Missing image: {img_path}")
        if mask_path is None or not mask_path.exists():
            raise FileNotFoundError(
                f"Full-volume validation requires an explicit mask for {case_id}."
            )
        img, ref_img = self._sitk_read_img_and_ref(img_path)
        mask = self._sitk_read_mask_resample_to_ref(mask_path, ref_img)
        if img.shape != mask.shape:
            img, mask = match_img_mask_shape(
                img, mask, pad_value_img=-1024, pad_value_mask=0
            )
        image_tensor = torch.from_numpy(
            ((windowing_3ch(img) - 0.5) / 0.5).astype(np.float32, copy=False)
        )
        sample = {
            'data': image_tensor,
            'target': torch.from_numpy((mask > 0).astype(np.uint8, copy=False)),
            'case_id': case_id,
            'image_path': str(img_path),
            'mask_path': str(mask_path),
        }
        if self.use_dicom:
            sample['dicom_numeric'] = torch.from_numpy(
                np.asarray(self.dicom_numeric[idx], dtype=np.float32)
            )
            sample['dicom_categorical'] = torch.from_numpy(
                np.asarray(self.dicom_categorical[idx], dtype=np.int64)
            )
        return sample

    def __getitem__(self, idx, sample_positive=True, sampling_mode=None):
        row = self.df.loc[idx]
        case_id = str(row[self.case_id_col]).strip()

        if 'image_path' in row and pd.notna(row['image_path']):
            img_path = self._resolve_image_path(row['image_path'])
        else:
            img_path = self._resolve_image_path(self.image_root / f"{case_id}.nii.gz")

        explicit_mask_path = None
        if 'mask_path' in row and pd.notna(row['mask_path']):
            explicit_mask_path = Path(str(row['mask_path']).strip())

        mask_path = find_case_mask_path(case_id, explicit_mask_path, self.mask_search_dirs)

        if not img_path.exists():
            raise FileNotFoundError(f"이미지 없음: {img_path}")

        # image + ref
        img, ref_img = self._sitk_read_img_and_ref(img_path)

        # mask resample to ref (✅ 이걸 실제로 사용!)
        if mask_path is not None and mask_path.exists():
            mask = self._sitk_read_mask_resample_to_ref(mask_path, ref_img)
        else:
            # 마스크 파일이 존재하지 않는 경우 (예: 정상군 또는 누락된 케이스), 0으로 채워진 빈 마스크 생성
            mask = np.zeros_like(img)

        lesion_voxels = float((mask > 0).sum())
        hemorrhage_burden = (
            float(np.log1p(lesion_voxels) / np.log1p(mask.size))
            if lesion_voxels > 0 else 0.0
        )
        safe_text = " ".join(
            _clean_text(row[column])
            for column in self.safe_text_columns
            if column in row
        )
        trauma_target = float(bool(TRAUMA_PATTERN.search(safe_text)))

        # prompt
        context = self.prompt_map.get(case_id, "<SEG>")

        if self.include_body and self.has_body and pd.notna(row[self.body_col]):
            body = str(row[self.body_col]).strip()
            if body:
                context = (context.replace(" <SEG>", "") + f", Body: {body} <SEG>").strip()

        if self.include_conclusion and self.has_conclusion and pd.notna(row[self.conclusion_col]):
            conclusion = str(row[self.conclusion_col]).strip()
            if conclusion:
                context = (context.replace(" <SEG>", "") + f", Conclusion: {conclusion} <SEG>").strip()

        # (선택) shape mismatch가 그래도 생기면 padding
        if img.shape != mask.shape:
            img, mask = match_img_mask_shape(img, mask, pad_value_img=-1024, pad_value_mask=0)

        img, mask = extract_patches(img, mask, sample_positive=sample_positive, patch_size=self.patch_size)

        img = windowing_3ch(img)
        mask = (mask > 0).astype(np.uint8)[None, ...]

        sample = {
            'data': img,
            'seg': mask,
            'context': context,
            'image_path': str(img_path),
            'case_id': case_id,
            'hemorrhage_burden': np.float32(hemorrhage_burden),
            'trauma_target': np.float32(trauma_target),
            'sampling_mode': sampling_mode or (
                'foreground_attempt' if sample_positive else 'random'
            ),
            'patch_has_foreground': np.float32(bool(mask.any())),
        }
        if self.use_dicom:
            sample['dicom_numeric'] = self.dicom_numeric[idx]
            sample['dicom_categorical'] = self.dicom_categorical[idx]
        return sample

    

class HemoDataLoader3D(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size,
        transforms=None,
        positive_prob=0.8,
        balanced_sampling=False,
    ):
        super().__init__(dataset, batch_size, infinite=True)
        self.dataset = dataset
        self.transforms = transforms
        self.positive_prob = positive_prob
        self.balanced_sampling = bool(balanced_sampling)
        self._sampling_cycle = []
        if self.balanced_sampling:
            if not self.dataset.positive_indices or not self.dataset.negative_indices:
                raise ValueError(
                    "Balanced sampling requires non-empty positive and negative manifest pools."
                )
            self._refill_sampling_cycle()

    def _refill_sampling_cycle(self):
        # Exact 50/25/25 composition over every four samples. Shuffling avoids
        # a fixed foreground/background order when local batch size is two.
        self._sampling_cycle = [
            "foreground",
            "foreground",
            "positive_random",
            "normal_random",
        ]
        random.shuffle(self._sampling_cycle)

    def _next_sampling_mode(self):
        if not self._sampling_cycle:
            self._refill_sampling_cycle()
        return self._sampling_cycle.pop()

    def _sample_balanced_index(self, mode):
        pool = (
            self.dataset.negative_indices
            if mode == "normal_random"
            else self.dataset.positive_indices
        )
        return int(random.choice(pool))

    def get_indices(self):
        indices = np.random.choice(range(len(self.dataset.df)), size=self.batch_size, replace=False)
        return indices

    def generate_train_batch(self):
        if self.balanced_sampling:
            sampling_modes = [self._next_sampling_mode() for _ in range(self.batch_size)]
            indices = [self._sample_balanced_index(mode) for mode in sampling_modes]
        else:
            indices = self.get_indices()
            sampling_modes = [None] * self.batch_size
        data_all = []
        seg_all = []
        image_path_all = []
        context_all = []
        case_id_all = []
        dicom_numeric_all = []
        dicom_categorical_all = []
        hemorrhage_burden_all = []
        trauma_target_all = []
        sampling_mode_all = []
        patch_has_foreground_all = []

        for j, idx in enumerate(indices):
            sampling_mode = sampling_modes[j]
            if self.balanced_sampling:
                sample_positive = sampling_mode == "foreground"
                data_dict = self._data.__getitem__(
                    idx,
                    sample_positive=sample_positive,
                    sampling_mode=sampling_mode,
                )
            elif random.random() < self.positive_prob:
                data_dict = self._data.__getitem__(idx, sample_positive=True)
            else:
                data_dict = self._data.__getitem__(idx, sample_positive=False)
            data_all.append(data_dict['data'])
            seg_all.append(data_dict['seg'])
            image_path_all.append(data_dict['image_path'])
            context_all.append(data_dict['context'])
            case_id_all.append(data_dict['case_id'])
            hemorrhage_burden_all.append(data_dict['hemorrhage_burden'])
            trauma_target_all.append(data_dict['trauma_target'])
            sampling_mode_all.append(data_dict['sampling_mode'])
            patch_has_foreground_all.append(data_dict['patch_has_foreground'])
            if self.dataset.use_dicom:
                dicom_numeric_all.append(data_dict['dicom_numeric'])
                dicom_categorical_all.append(data_dict['dicom_categorical'])

        data_all = np.stack(data_all)
        seg_all = np.stack(seg_all)
        image_path_all = np.stack(image_path_all)
        hemorrhage_burden_all = np.asarray(hemorrhage_burden_all, dtype=np.float32)
        trauma_target_all = np.asarray(trauma_target_all, dtype=np.float32)
        patch_has_foreground_all = np.asarray(
            patch_has_foreground_all, dtype=np.float32
        )
        if self.dataset.use_dicom:
            dicom_numeric_all = np.stack(dicom_numeric_all)
            dicom_categorical_all = np.stack(dicom_categorical_all)

        # shuffle
        indices = np.arange(len(data_all))
        np.random.shuffle(indices)
        data_all = data_all[indices]
        seg_all = seg_all[indices]
        image_path_all = image_path_all[indices]
        context_all = list(np.array(context_all)[indices])
        case_id_all = list(np.array(case_id_all)[indices])
        hemorrhage_burden_all = hemorrhage_burden_all[indices]
        trauma_target_all = trauma_target_all[indices]
        sampling_mode_all = list(np.array(sampling_mode_all)[indices])
        patch_has_foreground_all = patch_has_foreground_all[indices]
        if self.dataset.use_dicom:
            dicom_numeric_all = dicom_numeric_all[indices]
            dicom_categorical_all = dicom_categorical_all[indices]

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)

                    images = []
                    segs = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})
                        tmp['image'] = (tmp['image'].clamp(0, 1) - 0.5) / 0.5 # Normalize to [-1, 1]
                        images.append(tmp['image'])
                        segs.append(tmp['segmentation'])
                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    full_resolution_seg = seg_all[0] if isinstance(seg_all, list) else seg_all
                    patch_has_foreground_all = (
                        full_resolution_seg.reshape(self.batch_size, -1)
                        .gt(0)
                        .any(dim=1)
                        .float()
                    )
                    del segs, images

            batch = {
                'data': data_all,
                'target': seg_all,
                'image_path': image_path_all,
                'case_id': case_id_all,
                'context': context_all,
                'hemorrhage_burden': torch.from_numpy(hemorrhage_burden_all).float(),
                'trauma_target': torch.from_numpy(trauma_target_all).float(),
                'sampling_mode': sampling_mode_all,
                'patch_has_foreground': patch_has_foreground_all,
            }
            if self.dataset.use_dicom:
                batch['dicom_numeric'] = torch.from_numpy(dicom_numeric_all).float()
                batch['dicom_categorical'] = torch.from_numpy(dicom_categorical_all).long()
            return batch

        batch = {
            'data': data_all,
            'target': seg_all,
            'image_path': image_path_all,
            'case_id': case_id_all,
            'context': context_all,
            'hemorrhage_burden': hemorrhage_burden_all,
            'trauma_target': trauma_target_all,
            'sampling_mode': sampling_mode_all,
            'patch_has_foreground': patch_has_foreground_all,
        }
        if self.dataset.use_dicom:
            batch['dicom_numeric'] = dicom_numeric_all
            batch['dicom_categorical'] = dicom_categorical_all
        return batch


def get_data_loaders(batch_size=4, 
                     num_workers=4, 
                     deep_supervision=False, 
                     patch_size=(32, 256, 256),
                     positive_prob=0.8,
                     spatial_only=False,
                     rater=1,
                     include_clinical=False,
                      include_findings=False,
                      include_cc=True,
                      include_chief_complaint=True,
                      include_emr=False,
                      dicom_prompt_mode="none",
                      include_demographics=False,
                      use_dicom=False,
                      train_csv=None,
                      valid_csv=None,
                      vision_manifest_path=None,
                      balanced_sampling=False,
                      filter_invalid_supervision=False,
                      labeled_validation_only=False,
                      return_metadata=False,
                     ):
    patch_size = tuple(patch_size)
    rotation_for_DA = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
    mirror_axes = (0, 1, 2)

    deep_supervision_scales = None

    if deep_supervision:
        deep_supervision_scales = [[1.0, 1.0, 1.0],
                                   [1.0, 0.5, 0.5],
                                   [1.0, 0.25, 0.25],
                                   [1.0, 0.125, 0.125],
                                   [1.0, 0.0625, 0.0625]]
    
    tr_transforms = get_training_transforms(patch_size, 
                                            rotation_for_DA, 
                                            deep_supervision_scales, 
                                            mirror_axes, 
                                            do_dummy_2d_data_aug=True,
                                            spatial_only=spatial_only)

    val_transforms = get_validation_transforms(deep_supervision_scales, False)

    valid_case_ids = set()
    if valid_csv is not None:
        valid_frame = read_dataset_table(valid_csv)
        valid_id_col = _case_id_column(valid_frame)
        valid_case_ids = set(valid_frame[valid_id_col].astype(str).str.strip())

    train_ds = HemoDataset(
        mode='train',
        deep_supervision=deep_supervision,
        patch_size=patch_size,
        rater=rater,
        include_clinical=include_clinical,
        include_findings=include_findings,
        include_cc=include_cc,
        include_chief_complaint=include_chief_complaint,
        include_emr=include_emr,
        dicom_prompt_mode=dicom_prompt_mode,
        include_demographics=include_demographics,
        use_dicom=use_dicom,
        exclude_case_ids=valid_case_ids,
        vision_manifest_path=vision_manifest_path,
        filter_invalid_supervision=filter_invalid_supervision,
        csv_path=train_csv,
    )
    val_ds = HemoDataset(
        mode='valid',
        deep_supervision=deep_supervision,
        patch_size=patch_size,
        rater=rater,
        include_clinical=include_clinical,
        include_findings=include_findings,
        include_cc=include_cc,
        include_chief_complaint=include_chief_complaint,
        include_emr=include_emr,
        dicom_prompt_mode=dicom_prompt_mode,
        include_demographics=include_demographics,
        use_dicom=use_dicom,
        dicom_schema=train_ds.dicom_schema,
        vision_manifest_path=vision_manifest_path,
        filter_invalid_supervision=filter_invalid_supervision,
        labeled_validation_only=labeled_validation_only,
        csv_path=valid_csv,
    )
    train_loader = HemoDataLoader3D(
        train_ds,
        batch_size,
        transforms=tr_transforms,
        positive_prob=positive_prob,
        balanced_sampling=balanced_sampling,
    )
    val_loader = HemoDataLoader3D(val_ds, batch_size, transforms=val_transforms, positive_prob=1)

    mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=train_loader, transform=None,
                                                num_processes=num_workers,
                                                num_cached=max(6, num_workers // 2), seeds=None,
                                                pin_memory=True, wait_time=0.002)
    
    mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=val_loader,
                                              transform=None, num_processes=max(1, num_workers // 2),
                                              num_cached=max(3, num_workers // 4), seeds=None,
                                              pin_memory=True,
                                              wait_time=0.002)

    if return_metadata:
        metadata = {
            "dicom_schema": train_ds.dicom_schema,
            "train_samples": len(train_ds),
            "valid_samples": len(val_ds),
            "train_dataset": train_ds,
            "valid_dataset": val_ds,
            "safe_text_columns": list(SAFE_TEXT_COLUMNS),
        }
        return mt_gen_train, mt_gen_val, metadata
    return mt_gen_train, mt_gen_val
