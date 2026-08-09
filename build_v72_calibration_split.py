#!/usr/bin/env python
"""Create a locked, lesion-size-stratified V7.2 calibration split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-csv", required=True)
    parser.add_argument("--vision-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stable_order(case_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{case_id}".encode("utf-8")).hexdigest()


def strict_bool(series: pd.Series, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "1": True, "false": False, "0": False}
    if (~normalized.isin(mapping)).any():
        raise ValueError(f"Invalid boolean values in {column}")
    return normalized.map(mapping).astype(bool)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.calibration_fraction < 1.0:
        raise ValueError("calibration-fraction must be between zero and one.")
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Split already exists: {output}; use --overwrite")

    annotation = pd.read_csv(args.annotation_csv, encoding="utf-8-sig")
    manifest = pd.read_csv(args.vision_manifest, encoding="utf-8-sig")
    required_annotation = {"case_id", "probability_path", "gt_mask_path"}
    required_manifest = {
        "case_id", "actual_foreground_voxels", "validation_eligible",
        "supervision_status",
    }
    if required_annotation - set(annotation.columns):
        raise ValueError("Annotation CSV lacks the probability/GT case contract.")
    if required_manifest - set(manifest.columns):
        raise ValueError("Vision manifest lacks validation eligibility or voxel counts.")
    annotation["case_id"] = annotation["case_id"].astype(str).str.strip()
    manifest["case_id"] = manifest["case_id"].astype(str).str.strip()
    if annotation["case_id"].duplicated().any() or manifest["case_id"].duplicated().any():
        raise ValueError("Duplicate case IDs are not allowed in split inputs.")

    labeled = annotation[
        annotation["probability_path"].notna() & annotation["gt_mask_path"].notna()
    ][["case_id"]]
    manifest = manifest.copy()
    manifest["validation_eligible"] = strict_bool(
        manifest["validation_eligible"], "validation_eligible"
    )
    eligible = manifest[manifest["validation_eligible"]][
        ["case_id", "actual_foreground_voxels", "supervision_status"]
    ]
    table = labeled.merge(eligible, on="case_id", how="inner", validate="one_to_one")
    if table.empty:
        raise RuntimeError("No manifest-eligible labeled validation cases were found.")

    table["gt_is_positive"] = table["actual_foreground_voxels"].gt(0).astype(int)
    table["lesion_size_stratum"] = "normal"
    positive = table[table["gt_is_positive"].eq(1)].sort_values(
        ["actual_foreground_voxels", "case_id"]
    )
    if len(positive):
        positive_indices = np.array_split(positive.index.to_numpy(), 4)
        for bin_index, indices in enumerate(positive_indices, start=1):
            table.loc[indices, "lesion_size_stratum"] = f"positive_q{bin_index}"

    table["cohort"] = ""
    for stratum, group in table.groupby("lesion_size_stratum", sort=True):
        ordered = group.assign(
            _order=group["case_id"].map(lambda value: stable_order(value, args.seed))
        ).sort_values("_order")
        calibration_count = int(round(len(ordered) * args.calibration_fraction))
        calibration_count = min(max(1, calibration_count), len(ordered) - 1)
        table.loc[ordered.index[:calibration_count], "cohort"] = "calibration"
        table.loc[ordered.index[calibration_count:], "cohort"] = "locked"

    if table["cohort"].eq("").any():
        raise RuntimeError("Some cases were not assigned to a cohort.")
    table = table.sort_values("case_id").reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False, encoding="utf-8-sig")
    summary = (
        table.groupby(["cohort", "lesion_size_stratum", "gt_is_positive"])
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
        "seed": args.seed,
        "calibration_fraction": args.calibration_fraction,
        "eligible_labeled_cases": len(table),
        "annotation_labeled_cases": len(labeled),
        "excluded_by_supervision_contract": len(labeled) - len(table),
        "calibration_cases": int(table["cohort"].eq("calibration").sum()),
        "locked_cases": int(table["cohort"].eq("locked").sum()),
    }
    output.with_name(output.stem + "_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"[DONE] split={output}")


if __name__ == "__main__":
    main()
