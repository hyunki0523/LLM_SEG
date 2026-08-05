#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge preview inference summaries.")
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    records = []
    for path in sorted(root.glob("*/evaluation_summary.json")):
        with path.open(encoding="utf-8") as f:
            record = json.load(f)
        record["experiment"] = path.parent.name
        records.append(record)

    if not records:
        raise SystemExit(f"No evaluation_summary.json files found under {root}")

    columns = [
        "experiment",
        "processed",
        "n_positive",
        "n_normal",
        "mean_dice_positive",
        "mean_sensitivity_positive",
        "mean_hd95_positive_mm",
        "normal_no_fp_rate",
        "model_path",
    ]
    comparison = pd.DataFrame(records)
    comparison = comparison[[c for c in columns if c in comparison.columns]]
    output_path = root / "comparison.csv"
    comparison.to_csv(output_path, index=False, encoding="utf-8-sig")
    print("\n" + comparison.drop(columns=["model_path"], errors="ignore").to_string(index=False))
    print(f"\n[DONE] Comparison: {output_path}")

    per_case = None
    metric_columns = [
        "case_dice",
        "case_sensitivity",
        "fp_voxels",
        "gt_voxels",
        "pred_voxels",
        "normal_no_fp",
        "hemorrhage_volume_ml",
        "pred_path",
    ]
    for summary_path in sorted(root.glob("*/evaluation_summary.json")):
        experiment = summary_path.parent.name
        annotation_path = summary_path.parent / "annotation.csv"
        if not annotation_path.exists():
            continue
        annotation = pd.read_csv(annotation_path, encoding="utf-8-sig")
        identity_columns = [c for c in ["case_id", "class", "subclass"] if c in annotation]
        available_metrics = [c for c in metric_columns if c in annotation]
        annotation = annotation[identity_columns + available_metrics].copy()
        annotation = annotation.rename(
            columns={column: f"{experiment}__{column}" for column in available_metrics}
        )
        if per_case is None:
            per_case = annotation
        else:
            per_case = per_case.merge(
                annotation,
                on=[c for c in ["case_id", "class", "subclass"] if c in per_case and c in annotation],
                how="outer",
            )
    if per_case is not None:
        per_case_path = root / "per_case_comparison.csv"
        per_case.to_csv(per_case_path, index=False, encoding="utf-8-sig")
        print(f"[DONE] Per-case comparison: {per_case_path}")


if __name__ == "__main__":
    main()
