#!/usr/bin/env python
"""Select V7 parameters on a calibration split and evaluate once on holdout.

The split is deterministic from case_id.  Parameters are selected separately
for Real, Shuffled, and Empty CC using only calibration cases.  The holdout
partition is never consulted during selection.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-case-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vision-configuration", default="vision_only")
    parser.add_argument("--dicom-configuration", default="vision_dicom_film")
    parser.add_argument("--calibration-fraction", type=float, default=0.7)
    parser.add_argument("--split-seed", default="v72-safety-20260809")
    parser.add_argument("--max-mean-sensitivity-drop-vs-vision", type=float, default=0.01)
    parser.add_argument("--max-q95-incremental-drop-vs-dicom", type=float, default=0.01)
    parser.add_argument("--max-catastrophic-drop-rate", type=float, default=0.001)
    parser.add_argument("--max-newly-missed-cases", type=int, default=0)
    return parser.parse_args()


def split_for_case(case_id: str, seed: str, calibration_fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}:{case_id}".encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64)
    return "calibration" if fraction < calibration_fraction else "holdout"


def main() -> None:
    args = parse_args()
    if not 0.0 < args.calibration_fraction < 1.0:
        raise ValueError("--calibration-fraction must be between zero and one.")
    data = pd.read_csv(args.per_case_csv, encoding="utf-8-sig")
    required = {
        "case_id", "configuration", "score_mode", "beta", "text_threshold",
        "gt_is_positive", "tp_voxels", "fp_voxels", "dice", "sensitivity",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing per-case columns: {sorted(missing)}")
    data["case_id"] = data["case_id"].astype(str).str.strip()
    case_split = {
        case_id: split_for_case(case_id, args.split_seed, args.calibration_fraction)
        for case_id in data["case_id"].unique()
    }
    data["partition"] = data["case_id"].map(case_split)

    def indexed(configuration: str, partition: str) -> pd.DataFrame:
        result = data[
            data["configuration"].eq(configuration)
            & data["partition"].eq(partition)
        ].set_index("case_id")
        if result.empty:
            raise ValueError(f"Missing {configuration} in {partition}")
        return result

    def summarize(configuration: str, partition: str) -> dict[str, object]:
        vision = indexed(args.vision_configuration, partition)
        dicom = indexed(args.dicom_configuration, partition)
        candidate = indexed(configuration, partition)
        if set(candidate.index) != set(vision.index) or set(candidate.index) != set(dicom.index):
            raise ValueError(f"Cohort mismatch for {configuration} in {partition}")
        positive_ids = candidate.index[candidate["gt_is_positive"].eq(1)]
        normal_ids = candidate.index[candidate["gt_is_positive"].eq(0)]
        delta_vision = (
            candidate.loc[positive_ids, "sensitivity"]
            - vision.loc[positive_ids, "sensitivity"]
        )
        delta_dicom = (
            candidate.loc[positive_ids, "sensitivity"]
            - dicom.loc[positive_ids, "sensitivity"]
        )
        incremental_drop = (-delta_dicom).clip(lower=0.0)
        newly_missed = (
            dicom.loc[positive_ids, "tp_voxels"].gt(0)
            & candidate.loc[positive_ids, "tp_voxels"].eq(0)
        )
        catastrophic_rate = float(incremental_drop.gt(0.05).mean())
        mean_drop = float(-delta_vision.mean())
        q95_drop = float(incremental_drop.quantile(0.95))
        newly_missed_count = int(newly_missed.sum())
        safety_pass = bool(
            mean_drop <= args.max_mean_sensitivity_drop_vs_vision + 1e-12
            and q95_drop <= args.max_q95_incremental_drop_vs_dicom + 1e-12
            and catastrophic_rate <= args.max_catastrophic_drop_rate + 1e-12
            and newly_missed_count <= args.max_newly_missed_cases
        )
        first = candidate.iloc[0]
        return {
            "partition": partition,
            "configuration": configuration,
            "score_mode": first["score_mode"],
            "beta": first["beta"],
            "text_threshold": first["text_threshold"],
            "n_positive": len(positive_ids),
            "n_normal": len(normal_ids),
            "mean_dice_positive": float(candidate.loc[positive_ids, "dice"].mean()),
            "mean_sensitivity_drop_vs_vision": mean_drop,
            "mean_incremental_sensitivity_drop_vs_dicom": float(-delta_dicom.mean()),
            "q95_incremental_drop_vs_dicom": q95_drop,
            "catastrophic_drop_rate": catastrophic_rate,
            "newly_missed_vs_dicom_count": newly_missed_count,
            "normal_fp_reduction_vs_dicom": int(
                (
                    dicom.loc[normal_ids, "fp_voxels"]
                    - candidate.loc[normal_ids, "fp_voxels"]
                ).sum()
            ),
            "safety_pass": safety_pass,
        }

    candidate_names = data.loc[
        ~data["configuration"].isin(
            [args.vision_configuration, args.dicom_configuration]
        ),
        "configuration",
    ].drop_duplicates()
    calibration_rows = [summarize(name, "calibration") for name in candidate_names]
    calibration = pd.DataFrame(calibration_rows)
    selected_rows: list[dict[str, object]] = []
    for score_mode, group in calibration.groupby("score_mode", sort=False):
        eligible = group[group["safety_pass"]].sort_values(
            ["normal_fp_reduction_vs_dicom", "mean_dice_positive"],
            ascending=[False, False],
        )
        if eligible.empty:
            selected_rows.append(
                {
                    "score_mode": score_mode,
                    "selected_configuration": "",
                    "selection_status": "no_safe_calibration_candidate",
                }
            )
            continue
        selected_name = str(eligible.iloc[0]["configuration"])
        holdout = summarize(selected_name, "holdout")
        holdout["selected_configuration"] = selected_name
        holdout["selection_status"] = "selected_on_calibration_only"
        selected_rows.append(holdout)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration = calibration.sort_values(
        ["score_mode", "safety_pass", "normal_fp_reduction_vs_dicom"],
        ascending=[True, False, False],
    )
    selection = pd.DataFrame(selected_rows)
    split_frame = pd.DataFrame(
        sorted(case_split.items()), columns=["case_id", "partition"]
    )
    calibration.to_csv(
        output_dir / "v72_calibration_candidate_ranking.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selection.to_csv(
        output_dir / "v72_locked_holdout_evaluation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    split_frame.to_csv(
        output_dir / "v72_case_partition.csv", index=False, encoding="utf-8-sig"
    )
    print(selection.to_string(index=False))
    print(f"[DONE] calibration={output_dir / 'v72_calibration_candidate_ranking.csv'}")
    print(f"[DONE] holdout={output_dir / 'v72_locked_holdout_evaluation.csv'}")


if __name__ == "__main__":
    main()
