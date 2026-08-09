#!/usr/bin/env python
"""Build the supervision manifest used by Vision-Balanced-v1."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import label

from utils.mask_paths import is_mask_file, select_preferred_mask_path


DEFAULT_MASK_DIRS = (
    "/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/FUdata/hemo_masks/thick_th0.56",
    "/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/FUdata/hemo_masks/thin_th0.56",
    "/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/FUdata/normal_masks",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--valid-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--mask-dir",
        action="append",
        dest="mask_dirs",
        help="Mask search directory. Repeat to override the default three directories.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_dataset_table(path: str) -> pd.DataFrame:
    table_path = Path(path)
    if table_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(table_path)
    last_error = None
    for encoding in ("utf-8-sig", "cp949", "euc-kr", "latin-1"):
        try:
            return pd.read_csv(table_path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise RuntimeError(f"Could not decode dataset table: {table_path}") from last_error


def case_id_column(frame: pd.DataFrame) -> str:
    for candidate in ("영상일련번호ID", "case_id", "CaseID", "id"):
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"No case-id column found; columns={list(frame.columns)}")


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mask_statistics(path: str) -> tuple[int, int, str]:
    if not path:
        return 0, 0, "missing"
    try:
        try:
            array = np.asanyarray(nib.load(path).dataobj)
        except Exception:
            array = sitk.GetArrayFromImage(sitk.ReadImage(path))
        foreground = np.asarray(array > 0, dtype=bool)
        voxels = int(foreground.sum())
        components = int(label(foreground)[1]) if voxels else 0
        return voxels, components, "ok"
    except Exception as exc:
        return 0, 0, f"read_error:{type(exc).__name__}"


def mask_case_id(path: Path) -> str:
    name = path.name
    lowered = name.lower()
    for suffix in (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd"):
        if lowered.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def build_mask_index(mask_dirs: tuple[str, ...]) -> dict[str, Path]:
    """Index each search root once, preserving the configured directory priority."""
    index: dict[str, Path] = {}
    for raw_directory in mask_dirs:
        directory = Path(raw_directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"Mask directory not found: {directory}")
        directory_count = 0
        # Production FUdata mask roots are flat. Avoid recursive UNC/NAS walks,
        # which can take minutes even when only a few thousand files exist.
        for candidate in sorted(directory.iterdir()):
            # Extension filtering avoids one network stat call per file. The
            # three contracted production roots contain files, not subfolders.
            if not is_mask_file(candidate):
                continue
            case_id = mask_case_id(candidate)
            if case_id.lower() in {"final", "mask", "mask_final"}:
                case_id = candidate.parent.name
            index.setdefault(case_id, candidate)
            directory_count += 1
        print(
            f"[MANIFEST] indexed={directory_count} root={directory}",
            flush=True,
        )
    return index


def prepare_rows(
    path: str,
    split: str,
    mask_index: dict[str, Path],
) -> list[dict[str, object]]:
    frame = read_dataset_table(path)
    case_column = case_id_column(frame)
    rows = []
    for _, row in frame.iterrows():
        case_id = str(row[case_column]).strip()
        if not case_id or case_id.lower() in {"nan", "none", "null"}:
            raise ValueError(f"Invalid case ID in {path}: {case_id!r}")
        explicit = None
        if "mask_path" in row and pd.notna(row["mask_path"]):
            explicit = Path(str(row["mask_path"]).strip())
        mask_path = select_preferred_mask_path(explicit)
        if mask_path is None:
            mask_path = mask_index.get(case_id)
        class_value = row.get("class", None)
        if pd.isna(class_value) or not str(class_value).strip():
            raise ValueError(f"Missing class label for {case_id} in {path}")
        class_text = str(class_value).strip()
        class_is_positive = class_text.lower() != "normal"
        rows.append(
            {
                "case_id": case_id,
                "split": split,
                "image_path": str(row.get("image_path", "")),
                "mask_path": str(mask_path) if mask_path is not None else "",
                "mask_exists": bool(mask_path is not None),
                "class_text": class_text,
                "class_is_positive": class_is_positive,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Manifest already exists: {output}; use --overwrite")
    mask_dirs = tuple(args.mask_dirs or DEFAULT_MASK_DIRS)
    mask_index = build_mask_index(mask_dirs)
    rows = prepare_rows(args.train_csv, "train", mask_index) + prepare_rows(
        args.valid_csv, "valid", mask_index
    )
    tasks = [str(row["mask_path"]) for row in rows]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        statistics = executor.map(mask_statistics, tasks, chunksize=16)
        for index, (row, stats) in enumerate(zip(rows, statistics)):
            voxels, components, read_status = stats
            row["actual_foreground_voxels"] = voxels
            row["foreground_components"] = components
            row["mask_read_status"] = read_status
            class_positive = bool(row["class_is_positive"])
            mask_exists = bool(row["mask_exists"])
            if read_status.startswith("read_error"):
                supervision_status = "invalid_mask_read_error"
                train_eligible = validation_eligible = False
            elif class_positive and voxels > 0:
                supervision_status = "positive_mask"
                train_eligible = validation_eligible = True
            elif class_positive and not mask_exists:
                supervision_status = "invalid_positive_missing_mask"
                train_eligible = validation_eligible = False
            elif class_positive:
                supervision_status = "invalid_positive_empty_mask"
                train_eligible = validation_eligible = False
            elif voxels > 0:
                supervision_status = "invalid_normal_nonempty_mask"
                train_eligible = validation_eligible = False
            elif mask_exists:
                supervision_status = "negative_explicit_mask"
                train_eligible = validation_eligible = True
            else:
                supervision_status = "negative_implicit_class_label"
                train_eligible = True
                validation_eligible = False
            row["supervision_status"] = supervision_status
            row["train_eligible"] = train_eligible
            row["validation_eligible"] = validation_eligible
            if (index + 1) % 256 == 0 or index + 1 == len(rows):
                print(f"[MANIFEST] processed={index + 1}/{len(rows)}", flush=True)

    frame = pd.DataFrame(rows)
    duplicates = int(frame.duplicated("case_id").sum())
    if duplicates:
        raise RuntimeError(f"Manifest contains {duplicates} duplicate case IDs.")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    summary = (
        frame.groupby(["split", "supervision_status"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    summary.to_csv(
        output.with_name(output.stem + "_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    metadata = {
        "schema_version": 1,
        "train_csv": str(args.train_csv),
        "valid_csv": str(args.valid_csv),
        "train_csv_sha256": file_sha256(args.train_csv),
        "valid_csv_sha256": file_sha256(args.valid_csv),
        "mask_dirs": list(mask_dirs),
        "rows": int(len(frame)),
        "train_eligible": int(frame["train_eligible"].sum()),
        "validation_eligible": int(frame["validation_eligible"].sum()),
    }
    with open(
        output.with_name(output.stem + "_metadata.json"),
        "w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(metadata, metadata_file, indent=2, ensure_ascii=False)
    print(summary.to_string(index=False))
    print(f"[DONE] manifest={output}")


if __name__ == "__main__":
    main()
