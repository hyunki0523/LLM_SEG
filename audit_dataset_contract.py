#!/usr/bin/env python3
"""Audit the leakage-sensitive multimodal dataset contract before training."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

CASE_ID_COLUMN = "영상일련번호ID"
SAFE_TEXT_COLUMNS = ("extracted_cc", "chief_complaint")
DICOM_CATEGORICAL_COLUMNS = ("Manufacturer", "ConvolutionKernel")
DICOM_PROMPT_FIELD_MODES = {
    "full": {
        "manufacturer", "kernel", "slice_thickness", "kvp",
        "tube_current", "pixel_spacing",
    },
    "extended": {
        "manufacturer", "kernel", "slice_thickness", "kvp",
        "tube_current", "pixel_spacing", "spacing_between_slices",
        "series_description", "contrast",
    },
    "limited": {"kernel", "slice_thickness", "pixel_spacing"},
    "geometry": {"slice_thickness", "pixel_spacing", "spacing_between_slices"},
    "kernel_only": {"kernel"},
    "spacing_only": {"slice_thickness", "pixel_spacing", "spacing_between_slices"},
    "scanner_only": {"manufacturer", "kvp", "tube_current"},
    "protocol_only": {"kernel", "series_description", "contrast", "kvp"},
    "none": set(),
}


REQUIRED_DICOM_COLUMNS = (
    "KVP",
    "PixelSpacing",
    "SliceThickness",
    "XRayTubeCurrent",
    *DICOM_CATEGORICAL_COLUMNS,
)
PROHIBITED_TEXT_COLUMNS = (
    "검사결과본문",
    "검사결과결론",
    "History\n(판독문)",
    "초진기록지(EMR)",
    "refined_emr_v3",
    "class",
    "subclass",
)


def read_dataset_table(path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    for encoding in ("utf-8-sig", "cp949", "euc-kr", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            pass
    raise RuntimeError(f"Could not decode dataset table: {path}")


def _case_id_column(frame: pd.DataFrame) -> str:
    for candidate in (CASE_ID_COLUMN, "case_id", "CaseID", "id"):
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"Missing case ID column. Available: {list(frame.columns)}")


def _number(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _spacing(value):
    try:
        if isinstance(value, str):
            parsed = ast.literal_eval(value) if value.strip().startswith("[") else value.split(",")
        else:
            parsed = value
        if not isinstance(parsed, (list, tuple, np.ndarray)):
            parsed = [parsed]
        values = [_number(item) for item in parsed]
        values = [item for item in values if np.isfinite(item)]
        if not values:
            return np.nan, np.nan
        return values[0], values[1] if len(values) > 1 else values[0]
    except (ValueError, SyntaxError):
        return np.nan, np.nan


def _dicom_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    spacing = frame.get("PixelSpacing", pd.Series(np.nan, index=frame.index)).map(_spacing)
    return pd.DataFrame({
        "KVP": frame.get("KVP", pd.Series(np.nan, index=frame.index)).map(_number),
        "PixelSpacingX": spacing.map(lambda value: value[0]),
        "PixelSpacingY": spacing.map(lambda value: value[1]),
        "SliceThickness": frame.get(
            "SliceThickness", pd.Series(np.nan, index=frame.index)
        ).map(_number),
        "XRayTubeCurrent": frame.get(
            "XRayTubeCurrent", pd.Series(np.nan, index=frame.index)
        ).map(_number),
    })


def summarize(frame: pd.DataFrame, name: str) -> dict:
    id_col = _case_id_column(frame)
    ids = frame[id_col].astype(str).str.strip()
    numeric = _dicom_numeric_frame(frame)
    return {
        "name": name,
        "rows": len(frame),
        "unique_case_ids": int(ids.nunique()),
        "duplicate_case_rows": int(ids.duplicated().sum()),
        "missing_safe_text": {
            column: int(frame[column].isna().sum()) if column in frame else len(frame)
            for column in SAFE_TEXT_COLUMNS
        },
        "missing_dicom": {
            column: int(frame[column].isna().sum()) if column in frame else len(frame)
            for column in REQUIRED_DICOM_COLUMNS
        },
        "numeric_dicom_summary": numeric.describe().to_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--valid-csv", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--dicom-prompt-mode",
        default="none",
        choices=sorted(DICOM_PROMPT_FIELD_MODES),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when duplicates or train/valid overlap are found.",
    )
    args = parser.parse_args()

    train = read_dataset_table(args.train_csv)
    valid = read_dataset_table(args.valid_csv)
    train_id_col = _case_id_column(train)
    valid_id_col = _case_id_column(valid)
    train_ids = set(train[train_id_col].astype(str).str.strip())
    valid_ids = set(valid[valid_id_col].astype(str).str.strip())
    overlap = sorted(train_ids & valid_ids)

    missing_required = {
        "train": sorted(
            {CASE_ID_COLUMN, *SAFE_TEXT_COLUMNS, *REQUIRED_DICOM_COLUMNS} - set(train.columns)
        ),
        "valid": sorted(
            {CASE_ID_COLUMN, *SAFE_TEXT_COLUMNS, *REQUIRED_DICOM_COLUMNS} - set(valid.columns)
        ),
    }
    report = {
        "contract": {
            "allowed_text_columns": list(SAFE_TEXT_COLUMNS),
            "prohibited_text_columns": list(PROHIBITED_TEXT_COLUMNS),
            "dicom_prompt_mode": args.dicom_prompt_mode,
            "dicom_text_fields": sorted(
                DICOM_PROMPT_FIELD_MODES[args.dicom_prompt_mode]
            ),
            "dicom_transport": (
                "LLM text serialization from an explicit allow-list"
                if args.dicom_prompt_mode != "none"
                else "not serialized into LLM text; optional FiLM path remains separate"
            ),
        },
        "train": summarize(train, "train"),
        "valid": summarize(valid, "valid"),
        "cross_split_overlap_count": len(overlap),
        "cross_split_overlap_examples": overlap[:20],
        "missing_required_columns": missing_required,
    }

    rendered = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")

    has_duplicates = (
        report["train"]["duplicate_case_rows"] > 0
        or report["valid"]["duplicate_case_rows"] > 0
    )
    has_missing = any(missing_required.values())
    if args.strict and (has_duplicates or overlap or has_missing):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
