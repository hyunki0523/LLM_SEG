#!/usr/bin/env python
"""Audit V7 suppression for overlap and clinically important tail failures.

This script intentionally does not choose a deployment operating point from a
single mean metric.  It consumes the per-case matrix output and produces:

* candidate-level mean and tail-risk safety metrics;
* risk stratification by lesion size and real-CC P(normal);
* a review table of false-normal positive cases with the original prompt; and
* a compact list of candidates that pass every configured safety guard.

No image inference is repeated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SCORE_COLUMN = "p_normal_real_llm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-case-csv", required=True)
    parser.add_argument("--normal-scores-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vision-configuration", default="vision_only")
    parser.add_argument("--dicom-configuration", default="vision_dicom_film")
    parser.add_argument("--score-column", default=DEFAULT_SCORE_COLUMN)
    parser.add_argument("--false-normal-threshold", type=float, default=0.5)
    parser.add_argument("--max-mean-sensitivity-drop-vs-vision", type=float, default=0.01)
    parser.add_argument(
        "--max-q95-incremental-drop-vs-dicom", type=float, default=0.01
    )
    parser.add_argument(
        "--max-catastrophic-drop-rate", type=float, default=0.001,
        help="Maximum fraction with >5%% incremental sensitivity loss vs DICOM.",
    )
    parser.add_argument(
        "--max-newly-missed-cases", type=int, default=0,
        help="Maximum positive cases changed from DICOM TP>0 to candidate TP=0.",
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: set[str], source: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{source} lacks required columns: {sorted(missing)}")


def lesion_size_group(voxels: pd.Series) -> pd.Categorical:
    return pd.cut(
        voxels,
        bins=[-1, 100, 1_000, 10_000, np.inf],
        labels=["tiny_le_100", "small_101_1k", "medium_1k_10k", "large_gt_10k"],
    )


def p_normal_group(values: pd.Series) -> pd.Categorical:
    return pd.cut(
        values,
        bins=[-np.inf, 0.25, 0.5, 0.7, 0.9, np.inf],
        labels=["lt_025", "025_050", "050_070", "070_090", "ge_090"],
        right=False,
    )


def main() -> None:
    args = parse_args()
    per_case = pd.read_csv(args.per_case_csv, encoding="utf-8-sig")
    scores = pd.read_csv(args.normal_scores_csv, encoding="utf-8-sig")
    require_columns(
        per_case,
        {
            "case_id", "configuration", "gt_is_positive", "gt_voxels",
            "tp_voxels", "fp_voxels", "fn_voxels", "dice", "sensitivity",
        },
        args.per_case_csv,
    )
    require_columns(scores, {"case_id", "prompt", args.score_column}, args.normal_scores_csv)
    per_case["case_id"] = per_case["case_id"].astype(str).str.strip()
    scores["case_id"] = scores["case_id"].astype(str).str.strip()

    def indexed(configuration: str) -> pd.DataFrame:
        frame = per_case[per_case["configuration"] == configuration].copy()
        if frame.empty:
            raise ValueError(f"Missing configuration: {configuration}")
        if frame["case_id"].duplicated().any():
            raise ValueError(f"Duplicate cases in configuration: {configuration}")
        return frame.set_index("case_id")

    vision = indexed(args.vision_configuration)
    dicom = indexed(args.dicom_configuration)
    if set(vision.index) != set(dicom.index):
        raise ValueError("Vision and DICOM case cohorts differ.")

    score_table = scores.set_index("case_id")[["prompt", args.score_column]]
    candidates = [
        name for name in per_case["configuration"].drop_duplicates()
        if name not in {args.vision_configuration, args.dicom_configuration}
    ]
    safety_rows: list[dict[str, object]] = []
    stratum_rows: list[dict[str, object]] = []
    review_rows: list[pd.DataFrame] = []

    for configuration in candidates:
        candidate = indexed(configuration)
        if set(candidate.index) != set(vision.index):
            raise ValueError(f"Candidate cohort differs: {configuration}")
        positive_ids = candidate.index[candidate["gt_is_positive"].eq(1)]
        normal_ids = candidate.index[candidate["gt_is_positive"].eq(0)]
        positive = candidate.loc[positive_ids].join(score_table, how="left")
        if positive[args.score_column].isna().any():
            raise ValueError(f"Missing normal score in candidate: {configuration}")

        delta_vs_vision = positive["sensitivity"] - vision.loc[positive_ids, "sensitivity"]
        delta_vs_dicom = positive["sensitivity"] - dicom.loc[positive_ids, "sensitivity"]
        incremental_drop = (-delta_vs_dicom).clip(lower=0.0)
        newly_missed = (
            dicom.loc[positive_ids, "tp_voxels"].gt(0)
            & positive["tp_voxels"].eq(0)
        )
        catastrophic = incremental_drop.gt(0.05)
        fp_reduction_vs_vision = (
            vision.loc[normal_ids, "fp_voxels"] - candidate.loc[normal_ids, "fp_voxels"]
        )
        fp_reduction_vs_dicom = (
            dicom.loc[normal_ids, "fp_voxels"] - candidate.loc[normal_ids, "fp_voxels"]
        )
        mean_drop = float(-delta_vs_vision.mean())
        q95_drop = float(incremental_drop.quantile(0.95))
        catastrophic_rate = float(catastrophic.mean())
        newly_missed_count = int(newly_missed.sum())
        mean_pass = mean_drop <= args.max_mean_sensitivity_drop_vs_vision + 1e-12
        q95_pass = q95_drop <= args.max_q95_incremental_drop_vs_dicom + 1e-12
        catastrophic_pass = catastrophic_rate <= args.max_catastrophic_drop_rate + 1e-12
        newly_missed_pass = newly_missed_count <= args.max_newly_missed_cases

        safety_rows.append(
            {
                "configuration": configuration,
                "n_positive": len(positive_ids),
                "mean_dice_positive": float(positive["dice"].mean()),
                "mean_sensitivity_drop_vs_vision": mean_drop,
                "mean_incremental_sensitivity_drop_vs_dicom": float(-delta_vs_dicom.mean()),
                "q50_incremental_drop_vs_dicom": float(incremental_drop.quantile(0.50)),
                "q90_incremental_drop_vs_dicom": float(incremental_drop.quantile(0.90)),
                "q95_incremental_drop_vs_dicom": q95_drop,
                "q99_incremental_drop_vs_dicom": float(incremental_drop.quantile(0.99)),
                "max_incremental_drop_vs_dicom": float(incremental_drop.max()),
                "drop_gt_1pp_count": int(incremental_drop.gt(0.01).sum()),
                "drop_gt_2pp_count": int(incremental_drop.gt(0.02).sum()),
                "drop_gt_5pp_count": int(catastrophic.sum()),
                "catastrophic_drop_rate": catastrophic_rate,
                "newly_missed_vs_dicom_count": newly_missed_count,
                "normal_fp_reduction_vs_vision": int(fp_reduction_vs_vision.sum()),
                "normal_fp_reduction_vs_dicom": int(fp_reduction_vs_dicom.sum()),
                "normal_cases_with_fp_reduction_vs_dicom_rate": float(
                    fp_reduction_vs_dicom.gt(0).mean()
                ),
                "mean_safety_pass": mean_pass,
                "q95_safety_pass": q95_pass,
                "catastrophic_safety_pass": catastrophic_pass,
                "newly_missed_safety_pass": newly_missed_pass,
                "all_safety_guards_pass": bool(
                    mean_pass and q95_pass and catastrophic_pass and newly_missed_pass
                ),
            }
        )

        positive = positive.assign(
            lesion_size_group=lesion_size_group(positive["gt_voxels"]),
            p_normal_group=p_normal_group(positive[args.score_column]),
            sensitivity_delta_vs_vision=delta_vs_vision,
            sensitivity_delta_vs_dicom=delta_vs_dicom,
            incremental_drop_vs_dicom=incremental_drop,
            newly_missed_vs_dicom=newly_missed,
        )
        for stratum_type, column in (
            ("lesion_size", "lesion_size_group"),
            ("p_normal", "p_normal_group"),
        ):
            for stratum, group in positive.groupby(column, observed=True):
                stratum_rows.append(
                    {
                        "configuration": configuration,
                        "stratum_type": stratum_type,
                        "stratum": str(stratum),
                        "n": len(group),
                        "mean_dice": float(group["dice"].mean()),
                        "mean_sensitivity": float(group["sensitivity"].mean()),
                        "mean_sensitivity_delta_vs_vision": float(
                            group["sensitivity_delta_vs_vision"].mean()
                        ),
                        "mean_sensitivity_delta_vs_dicom": float(
                            group["sensitivity_delta_vs_dicom"].mean()
                        ),
                        "q95_incremental_drop_vs_dicom": float(
                            group["incremental_drop_vs_dicom"].quantile(0.95)
                        ),
                        "newly_missed_vs_dicom_count": int(
                            group["newly_missed_vs_dicom"].sum()
                        ),
                    }
                )

        false_normal = positive[
            positive[args.score_column].ge(args.false_normal_threshold)
        ].copy()
        false_normal["configuration"] = configuration
        false_normal["vision_sensitivity"] = vision.loc[
            false_normal.index, "sensitivity"
        ]
        false_normal["dicom_sensitivity"] = dicom.loc[
            false_normal.index, "sensitivity"
        ]
        false_normal["vision_dice"] = vision.loc[false_normal.index, "dice"]
        false_normal["dicom_dice"] = dicom.loc[false_normal.index, "dice"]
        false_normal["fn_delta_vs_vision"] = (
            false_normal["fn_voxels"] - vision.loc[false_normal.index, "fn_voxels"]
        )
        false_normal["fn_delta_vs_dicom"] = (
            false_normal["fn_voxels"] - dicom.loc[false_normal.index, "fn_voxels"]
        )
        review_rows.append(false_normal.reset_index())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safety = pd.DataFrame(safety_rows).sort_values(
        ["all_safety_guards_pass", "normal_fp_reduction_vs_dicom"],
        ascending=[False, False],
    )
    strata = pd.DataFrame(stratum_rows)
    review = pd.concat(review_rows, ignore_index=True)
    review = review.sort_values(
        [args.score_column, "incremental_drop_vs_dicom"], ascending=[False, False]
    )
    safety.to_csv(output_dir / "v7_candidate_tail_safety.csv", index=False, encoding="utf-8-sig")
    strata.to_csv(output_dir / "v7_candidate_stratified_safety.csv", index=False, encoding="utf-8-sig")
    review.to_csv(output_dir / "v7_false_normal_case_review.csv", index=False, encoding="utf-8-sig")

    positive_scores = score_table.join(vision[["gt_is_positive"]], how="inner")
    overlap_rows = []
    for threshold in (0.5, 0.7, 0.8, 0.9, 0.95):
        selected = positive_scores[args.score_column].ge(threshold)
        is_positive = positive_scores["gt_is_positive"].eq(1)
        overlap_rows.append(
            {
                "p_normal_threshold": threshold,
                "positive_selected_count": int((selected & is_positive).sum()),
                "positive_selected_rate": float(selected[is_positive].mean()),
                "normal_selected_count": int((selected & ~is_positive).sum()),
                "normal_selected_rate": float(selected[~is_positive].mean()),
            }
        )
    pd.DataFrame(overlap_rows).to_csv(
        output_dir / "v7_p_normal_overlap.csv", index=False, encoding="utf-8-sig"
    )
    print(safety.to_string(index=False))
    print(f"[DONE] safety={output_dir / 'v7_candidate_tail_safety.csv'}")
    print(f"[DONE] strata={output_dir / 'v7_candidate_stratified_safety.csv'}")
    print(f"[DONE] false_normal_review={output_dir / 'v7_false_normal_case_review.csv'}")


if __name__ == "__main__":
    main()
