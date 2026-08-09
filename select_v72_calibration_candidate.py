#!/usr/bin/env python
"""Select one V7.2 operating point using calibration results only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROPOSED_MODE = "v7_real_cc_proposed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-env", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = []
    all_rows = []
    for raw_path in args.summary:
        path = Path(raw_path)
        summary = pd.read_csv(path, encoding="utf-8-sig")
        dicom_rows = summary[summary["configuration"].eq("vision_dicom_film")]
        if len(dicom_rows) != 1:
            raise ValueError(f"Missing unique DICOM baseline in {path}")
        dicom_fp = int(dicom_rows.iloc[0]["normal_total_fp_voxels"])
        proposed = summary[summary["score_mode"].eq(PROPOSED_MODE)].copy()
        proposed["normal_fp_reduction_vs_dicom"] = (
            dicom_fp - proposed["normal_total_fp_voxels"]
        )
        if {"p_min", "p_protect"} - set(proposed.columns):
            raise ValueError(f"Summary lacks multi-band columns: {path}")
        proposed["source_summary"] = str(path)
        candidates.append(proposed)
        summary["source_summary"] = str(path)
        all_rows.append(summary)
    table = pd.concat(candidates, ignore_index=True)
    safety_vs_vision = table["safety_pass"].astype(str).str.lower().eq("true")
    safety_vs_dicom = (
        table["safety_vs_dicom_film"].astype(str).str.lower().eq("true")
    )
    safe = table[safety_vs_vision & safety_vs_dicom].copy()
    if safe.empty:
        raise RuntimeError(
            "No Real-CC candidate passes both Vision and DICOM sensitivity guards."
        )
    safe = safe.sort_values(
        [
            "normal_fp_reduction_vs_dicom",
            "normal_no_fp_rate",
            "mean_dice_positive",
            "beta",
        ],
        ascending=[False, False, False, True],
    )
    selected = safe.iloc[0]
    combined = pd.concat(all_rows, ignore_index=True)
    matched_controls = combined[
        combined["source_summary"].eq(selected["source_summary"])
        & combined["p_min"].eq(selected["p_min"])
        & combined["p_protect"].eq(selected["p_protect"])
        & combined["beta"].eq(selected["beta"])
        & combined["text_threshold"].eq(selected["text_threshold"])
        & combined["score_mode"].isin(
            {"v7_dicom_empty_cc", "v7_dicom_shuffled_cc"}
        )
    ]
    control_metrics = {
        row["score_mode"]: {
            "mean_dice_positive": float(row["mean_dice_positive"]),
            "mean_sensitivity_positive": float(row["mean_sensitivity_positive"]),
            "normal_total_fp_voxels": int(row["normal_total_fp_voxels"]),
        }
        for _, row in matched_controls.iterrows()
    }
    result = {
        "selection_cohort": "calibration",
        "configuration": selected["configuration"],
        "p_min": float(selected["p_min"]),
        "p_protect": float(selected["p_protect"]),
        "beta": float(selected["beta"]),
        "text_threshold": float(selected["text_threshold"]),
        "calibration_mean_dice_positive": float(selected["mean_dice_positive"]),
        "calibration_mean_sensitivity_positive": float(
            selected["mean_sensitivity_positive"]
        ),
        "calibration_sensitivity_delta_vs_vision": float(
            selected["sensitivity_delta_vs_vision"]
        ),
        "calibration_sensitivity_delta_vs_dicom_film": float(
            selected["sensitivity_delta_vs_dicom_film"]
        ),
        "calibration_normal_fp_reduction_vs_dicom": int(
            selected["normal_fp_reduction_vs_dicom"]
        ),
        "source_summary": selected["source_summary"],
        "matched_negative_controls": control_metrics,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output_env = Path(args.output_env)
    output_env.write_text(
        "\n".join(
            [
                f"SELECTED_P_MIN={result['p_min']}",
                f"SELECTED_P_PROTECT={result['p_protect']}",
                f"SELECTED_BETA={result['beta']}",
                f"SELECTED_TEXT_THRESHOLD={result['text_threshold']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[DONE] selected={output_json}")


if __name__ == "__main__":
    main()
