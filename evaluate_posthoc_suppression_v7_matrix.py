#!/usr/bin/env python
"""Evaluate the paired five-condition V7 DICOM/CC ablation matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

from evaluate_posthoc_suppression_v7 import (
    binary_metrics,
    load_ground_truth,
    parse_float_list,
    summarize,
    suppress_probability,
)


V7_SCORE_COLUMNS = {
    "v7_dicom_empty_cc": "p_normal_empty_llm",
    "v7_dicom_shuffled_cc": "p_normal_shuffled_llm",
    "v7_real_cc_proposed": "p_normal_real_llm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision-annotation-csv", required=True)
    parser.add_argument("--dicom-annotation-csv", required=True)
    parser.add_argument("--normal-scores-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--p-min", type=float, default=0.2)
    parser.add_argument("--p-protect", type=float, default=0.85)
    parser.add_argument(
        "--betas", type=parse_float_list, default=parse_float_list("0.25,0.5,1,2,3")
    )
    parser.add_argument(
        "--text-thresholds", type=parse_float_list, default=parse_float_list("0")
    )
    parser.add_argument("--prob-threshold", type=float, default=0.5)
    parser.add_argument("--max-sensitivity-drop", type=float, default=0.01)
    return parser.parse_args()


def load_annotation(path: str, probability_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"case_id", "probability_path", "gt_mask_path", "image_path"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks columns: {sorted(missing)}")
    frame = frame[
        frame["probability_path"].notna()
        & frame["gt_mask_path"].notna()
        & frame["image_path"].notna()
    ].copy()
    frame = frame[["case_id", "probability_path", "gt_mask_path", "image_path"]]
    return frame.rename(columns={"probability_path": probability_name})


def main() -> None:
    args = parse_args()
    if not 0 <= args.p_min < args.p_protect <= 1:
        raise ValueError("Require 0 <= p_min < p_protect <= 1.")
    vision = load_annotation(args.vision_annotation_csv, "vision_probability_path")
    dicom = load_annotation(args.dicom_annotation_csv, "dicom_probability_path")
    scores = pd.read_csv(args.normal_scores_csv, encoding="utf-8-sig")
    required_scores = {"case_id", *V7_SCORE_COLUMNS.values()}
    missing_scores = required_scores - set(scores.columns)
    if missing_scores:
        raise ValueError(f"Normal score CSV lacks columns: {sorted(missing_scores)}")

    vision_cases = set(vision["case_id"].astype(str))
    dicom_cases = set(dicom["case_id"].astype(str))
    if vision_cases != dicom_cases:
        raise RuntimeError(
            "Vision and DICOM inference cohorts differ; paired comparison is invalid. "
            f"vision_only={len(vision_cases - dicom_cases)} "
            f"dicom_only={len(dicom_cases - vision_cases)}"
        )
    table = vision.merge(
        dicom[["case_id", "dicom_probability_path"]],
        on="case_id",
        how="inner",
        validate="one_to_one",
    ).merge(
        scores[["case_id", *V7_SCORE_COLUMNS.values()]],
        on="case_id",
        how="inner",
        validate="one_to_one",
    )
    if len(table) != len(vision):
        raise RuntimeError("Normal-score coverage does not match the paired labeled cohort.")

    records: list[dict[str, object]] = []
    for index, row in table.iterrows():
        vision_probability = np.load(
            Path(str(row["vision_probability_path"])), allow_pickle=False
        ).astype(np.float32)
        dicom_probability = np.load(
            Path(str(row["dicom_probability_path"])), allow_pickle=False
        ).astype(np.float32)
        target, spacing_zyx = load_ground_truth(
            str(row["gt_mask_path"]), str(row["image_path"])
        )
        if vision_probability.shape != target.shape or dicom_probability.shape != target.shape:
            raise ValueError(
                f"Shape mismatch for {row['case_id']}: vision={vision_probability.shape}, "
                f"dicom={dicom_probability.shape}, target={target.shape}"
            )
        target_points = None
        target_tree = None
        if target.any():
            surface = target & ~binary_erosion(
                target,
                structure=np.ones((3, 3, 3), dtype=bool),
                border_value=0,
            )
            target_points = np.argwhere(surface).astype(np.float32)
            target_points *= np.asarray(spacing_zyx, dtype=np.float32)
            target_tree = cKDTree(target_points)

        for configuration, score_mode, probability in (
            ("vision_only", "vision_only", vision_probability),
            ("vision_dicom_film", "dicom_film", dicom_probability),
        ):
            records.append(
                {
                    "case_id": row["case_id"],
                    "configuration": configuration,
                    "score_mode": score_mode,
                    "beta": 0.0,
                    "text_threshold": np.nan,
                    "p_normal": np.nan,
                    **binary_metrics(
                        probability >= args.prob_threshold,
                        target,
                        spacing_zyx,
                        target_points,
                        target_tree,
                    ),
                }
            )

        for condition, score_column in V7_SCORE_COLUMNS.items():
            p_normal = float(row[score_column])
            for threshold in args.text_thresholds:
                for beta in args.betas:
                    final_probability = suppress_probability(
                        dicom_probability,
                        p_normal,
                        beta,
                        args.p_min,
                        args.p_protect,
                        threshold,
                    )
                    records.append(
                        {
                            "case_id": row["case_id"],
                            "configuration": f"{condition}__tau{threshold:g}__beta{beta:g}",
                            "score_mode": condition,
                            "beta": beta,
                            "text_threshold": threshold,
                            "p_normal": p_normal,
                            **binary_metrics(
                                final_probability >= args.prob_threshold,
                                target,
                                spacing_zyx,
                                target_points,
                                target_tree,
                            ),
                        }
                    )
        if (index + 1) % 50 == 0 or index + 1 == len(table):
            print(f"[MATRIX] processed={index + 1}/{len(table)}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_case = pd.DataFrame(records)
    summary = summarize(per_case, args.max_sensitivity_drop)
    dicom_sensitivity = float(
        summary.loc[
            summary["configuration"] == "vision_dicom_film",
            "mean_sensitivity_positive",
        ].iloc[0]
    )
    summary["sensitivity_delta_vs_dicom_film"] = (
        summary["mean_sensitivity_positive"] - dicom_sensitivity
    )
    summary["safety_vs_dicom_film"] = (
        dicom_sensitivity - summary["mean_sensitivity_positive"]
        <= args.max_sensitivity_drop + 1e-12
    )
    summary["five_condition_group"] = summary["score_mode"].map(
        {
            "vision_only": "Vision-Only (Baseline)",
            "dicom_film": "Vision + DICOM FiLM",
            "v7_dicom_empty_cc": "V7: DICOM + Empty CC",
            "v7_dicom_shuffled_cc": "V7: DICOM + Shuffled CC",
            "v7_real_cc_proposed": "V7: DICOM + Real CC (Proposed)",
        }
    )
    per_case_path = output_dir / "per_case_v7_five_condition_metrics.csv"
    summary_path = output_dir / "v7_five_condition_summary.csv"
    per_case.to_csv(per_case_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    (output_dir / "v7_five_condition_metadata.json").write_text(
        json.dumps(
            {
                "p_min": args.p_min,
                "p_protect": args.p_protect,
                "betas": args.betas,
                "text_thresholds": args.text_thresholds,
                "paired_labeled_cases": len(table),
                "v7_probability_base": "frozen DICOM-FiLM",
                "safety_reference": ["vision_only", "vision_dicom_film"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"[DONE] summary={summary_path}")
    print(f"[DONE] per_case={per_case_path}")


if __name__ == "__main__":
    main()
