#!/usr/bin/env python3
"""Remove leakage, duplicate rows, and required-field gaps from train/valid Excel files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


CASE_ID_CANDIDATES = ("영상일련번호ID", "case_id", "CaseID", "id")
REQUIRED_NONEMPTY_COLUMNS = ("XRayTubeCurrent", "extracted_cc")
MISSING_STRINGS = {"", "nan", "none", "null"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def case_id_column(frame: pd.DataFrame) -> str:
    for candidate in CASE_ID_CANDIDATES:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"No case-ID column found. Available columns: {list(frame.columns)}")


def normalized_ids(frame: pd.DataFrame, id_column: str) -> pd.Series:
    return frame[id_column].astype(str).str.strip()


def missing_mask(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(str).str.strip().str.lower().isin(
        MISSING_STRINGS
    )


def removal_masks(
    frame: pd.DataFrame,
    id_column: str,
    overlap_ids: set[str],
    remove_overlap: bool,
) -> dict[str, pd.Series]:
    ids = normalized_ids(frame, id_column)
    masks = {
        "duplicate_after_first": ids.duplicated(keep="first"),
        "missing_XRayTubeCurrent": missing_mask(frame["XRayTubeCurrent"]),
        "missing_extracted_cc": missing_mask(frame["extracted_cc"]),
    }
    masks["train_valid_overlap"] = (
        ids.isin(overlap_ids)
        if remove_overlap
        else pd.Series(False, index=frame.index)
    )
    return masks


def summarize_removals(
    frame: pd.DataFrame, id_column: str, masks: dict[str, pd.Series]
) -> dict[str, object]:
    ids = normalized_ids(frame, id_column)
    combined = pd.Series(False, index=frame.index)
    by_reason = {}
    for reason, mask in masks.items():
        combined |= mask
        by_reason[reason] = {
            "rows": int(mask.sum()),
            "case_ids": sorted(set(ids[mask].tolist())),
        }
    removed_rows = []
    for index in frame.index[combined]:
        reasons = [reason for reason, mask in masks.items() if bool(mask.loc[index])]
        removed_rows.append(
            {
                "excel_row": int(index) + 2,
                "case_id": ids.loc[index],
                "reasons": reasons,
            }
        )
    return {
        "removed_row_count": int(combined.sum()),
        "removed_unique_case_count": int(ids[combined].nunique()),
        "by_reason": by_reason,
        "removed_rows": removed_rows,
        "keep_mask": ~combined,
    }


def validate_cleaned(train: pd.DataFrame, valid: pd.DataFrame) -> dict[str, int]:
    train_id_column = case_id_column(train)
    valid_id_column = case_id_column(valid)
    train_ids = normalized_ids(train, train_id_column)
    valid_ids = normalized_ids(valid, valid_id_column)
    metrics = {
        "train_rows": len(train),
        "valid_rows": len(valid),
        "train_duplicate_rows": int(train_ids.duplicated().sum()),
        "valid_duplicate_rows": int(valid_ids.duplicated().sum()),
        "cross_split_overlap_count": len(set(train_ids) & set(valid_ids)),
    }
    for split, frame in (("train", train), ("valid", valid)):
        for column in REQUIRED_NONEMPTY_COLUMNS:
            metrics[f"{split}_missing_{column}"] = int(
                missing_mask(frame[column]).sum()
            )
    failures = {key: value for key, value in metrics.items() if key not in {
        "train_rows", "valid_rows"
    } and value != 0}
    if failures:
        raise AssertionError(f"Cleaned split validation failed: {failures}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--valid", required=True, type=Path)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Replace the source workbooks after backup. Otherwise audit only.",
    )
    parser.add_argument("--timestamp", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_path = args.train.resolve()
    valid_path = args.valid.resolve()
    for path in (train_path, valid_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    train = pd.read_excel(train_path)
    valid = pd.read_excel(valid_path)
    train_id_column = case_id_column(train)
    valid_id_column = case_id_column(valid)
    for split, frame in (("train", train), ("valid", valid)):
        missing_columns = set(REQUIRED_NONEMPTY_COLUMNS) - set(frame.columns)
        if missing_columns:
            raise ValueError(f"{split} is missing columns: {sorted(missing_columns)}")

    train_ids = normalized_ids(train, train_id_column)
    valid_ids = normalized_ids(valid, valid_id_column)
    overlap_ids = set(train_ids) & set(valid_ids)

    train_summary = summarize_removals(
        train,
        train_id_column,
        removal_masks(train, train_id_column, overlap_ids, remove_overlap=True),
    )
    valid_summary = summarize_removals(
        valid,
        valid_id_column,
        removal_masks(valid, valid_id_column, overlap_ids, remove_overlap=False),
    )
    train_clean = train.loc[train_summary.pop("keep_mask")].reset_index(drop=True)
    valid_clean = valid.loc[valid_summary.pop("keep_mask")].reset_index(drop=True)
    post_cleanup = validate_cleaned(train_clean, valid_clean)

    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = train_path.parent / f"split_cleanup_report_{timestamp}.json"
    report = {
        "timestamp": timestamp,
        "applied": bool(args.apply),
        "policy": {
            "cross_split_overlap": "keep validation row; remove matching train row",
            "duplicates": "keep first row within each split",
            "missing_values": list(REQUIRED_NONEMPTY_COLUMNS),
            "missing_string_sentinels": sorted(MISSING_STRINGS),
        },
        "source": {
            "train": {
                "path": str(train_path),
                "rows": len(train),
                "sha256": sha256(train_path),
            },
            "valid": {
                "path": str(valid_path),
                "rows": len(valid),
                "sha256": sha256(valid_path),
            },
        },
        "original_cross_split_overlap_count": len(overlap_ids),
        "original_cross_split_overlap_case_ids": sorted(overlap_ids),
        "train_removals": train_summary,
        "valid_removals": valid_summary,
        "post_cleanup": post_cleanup,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.apply:
        print("[DRY RUN] No workbook was modified. Add --apply to write changes.")
        return 0

    train_temp = train_path.with_name(f".{train_path.stem}.{timestamp}.tmp.xlsx")
    valid_temp = valid_path.with_name(f".{valid_path.stem}.{timestamp}.tmp.xlsx")
    train_backup = train_path.with_name(
        f"{train_path.stem}.backup_before_cleanup_{timestamp}{train_path.suffix}"
    )
    valid_backup = valid_path.with_name(
        f"{valid_path.stem}.backup_before_cleanup_{timestamp}{valid_path.suffix}"
    )
    for path in (train_temp, valid_temp, train_backup, valid_backup, report_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")

    try:
        train_clean.to_excel(train_temp, index=False, sheet_name="Sheet1")
        valid_clean.to_excel(valid_temp, index=False, sheet_name="Sheet1")
        reloaded_train = pd.read_excel(train_temp)
        reloaded_valid = pd.read_excel(valid_temp)
        reloaded_metrics = validate_cleaned(reloaded_train, reloaded_valid)
        if reloaded_metrics != post_cleanup:
            raise AssertionError(
                f"Round-trip workbook metrics changed: {reloaded_metrics} != {post_cleanup}"
            )

        shutil.copy2(train_path, train_backup)
        shutil.copy2(valid_path, valid_backup)
        os.replace(train_temp, train_path)
        os.replace(valid_temp, valid_path)
        report["backup"] = {
            "train": str(train_backup),
            "valid": str(valid_backup),
        }
        report["cleaned"] = {
            "train_sha256": sha256(train_path),
            "valid_sha256": sha256(valid_path),
        }
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    finally:
        train_temp.unlink(missing_ok=True)
        valid_temp.unlink(missing_ok=True)

    print(f"[DONE] Cleaned train workbook: {train_path}")
    print(f"[DONE] Cleaned valid workbook: {valid_path}")
    print(f"[BACKUP] {train_backup}")
    print(f"[BACKUP] {valid_backup}")
    print(f"[REPORT] {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
