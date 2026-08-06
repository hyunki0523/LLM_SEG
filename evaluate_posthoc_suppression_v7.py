#!/usr/bin/env python
"""Evaluate frozen V7 negative-only suppression from saved vision probabilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import binary_erosion, label
from scipy.spatial import cKDTree


SCORE_COLUMNS = {
    "real_cc": "p_normal_real_llm",
    "shuffled_cc": "p_normal_shuffled_llm",
    "empty_cc": "p_normal_empty_llm",
    "tfidf": "p_normal_tfidf",
}


def parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected a non-empty comma-separated list.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-csv", required=True)
    parser.add_argument("--normal-scores-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--p-min", type=float, default=0.2)
    parser.add_argument("--p-protect", type=float, default=0.85)
    parser.add_argument(
        "--betas", type=parse_float_list, default=parse_float_list("0.25,0.5,1,2,3")
    )
    parser.add_argument(
        "--text-thresholds",
        type=parse_float_list,
        default=parse_float_list("0"),
        help=(
            "Normal-confidence thresholds. Zero reproduces s=P(normal); values "
            "above zero activate only high-confidence normal predictions."
        ),
    )
    parser.add_argument("--prob-threshold", type=float, default=0.5)
    parser.add_argument("--max-sensitivity-drop", type=float, default=0.01)
    return parser.parse_args()


def band_pass_weight(probability: np.ndarray, p_min: float, p_protect: float) -> np.ndarray:
    weight = np.zeros_like(probability, dtype=np.float32)
    active = (probability >= p_min) & (probability < p_protect)
    weight[active] = (
        (p_protect - probability[active]) / (p_protect - p_min)
    ).astype(np.float32)
    return weight


def suppression_strength(p_normal: float, threshold: float) -> float:
    if threshold >= 1.0:
        return 0.0
    return float(np.clip((p_normal - threshold) / (1.0 - threshold), 0.0, 1.0))


def suppress_probability(
    vision_probability: np.ndarray,
    p_normal: float,
    beta: float,
    p_min: float,
    p_protect: float,
    text_threshold: float,
) -> np.ndarray:
    clipped = np.clip(vision_probability.astype(np.float32), 1e-6, 1.0 - 1e-6)
    vision_logit = np.log(clipped) - np.log1p(-clipped)
    spatial_weight = band_pass_weight(clipped, p_min, p_protect)
    strength = suppression_strength(p_normal, text_threshold)
    final_logit = vision_logit - float(beta) * strength * spatial_weight
    return 1.0 / (1.0 + np.exp(-final_logit))


def same_geometry(a: sitk.Image, b: sitk.Image, atol: float = 1e-5) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=atol, rtol=0)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=atol, rtol=0)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=atol, rtol=0)
    )


def load_ground_truth(mask_path: str, image_path: str) -> tuple[np.ndarray, tuple[float, ...]]:
    reference = sitk.ReadImage(str(image_path))
    try:
        mask = sitk.ReadImage(str(mask_path))
    except RuntimeError:
        image = nib.load(str(mask_path))
        mask = sitk.GetImageFromArray(np.asarray(image.dataobj))
        if mask.GetSize() == reference.GetSize():
            mask.CopyInformation(reference)
    if not same_geometry(mask, reference):
        if mask.GetSize() == reference.GetSize():
            aligned = sitk.Image(mask)
            aligned.CopyInformation(reference)
            mask = aligned
        else:
            mask = sitk.Resample(
                mask,
                reference,
                sitk.Transform(),
                sitk.sitkNearestNeighbor,
                0,
                mask.GetPixelID(),
            )
    spacing_zyx = tuple(float(value) for value in reversed(reference.GetSpacing()))
    return (sitk.GetArrayFromImage(mask) > 0), spacing_zyx


def surface_hd95_mm(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing_zyx: tuple[float, ...],
    target_surface_points: np.ndarray | None = None,
    target_tree: cKDTree | None = None,
) -> float:
    if not prediction.any() or not target.any():
        return float("inf")
    structure = np.ones((3, 3, 3), dtype=bool)
    prediction_surface = prediction & ~binary_erosion(
        prediction, structure=structure, border_value=0
    )
    prediction_points = np.argwhere(prediction_surface).astype(np.float32)
    prediction_points *= np.asarray(spacing_zyx, dtype=np.float32)
    if target_surface_points is None:
        target_surface = target & ~binary_erosion(
            target, structure=structure, border_value=0
        )
        target_surface_points = np.argwhere(target_surface).astype(np.float32)
        target_surface_points *= np.asarray(spacing_zyx, dtype=np.float32)
    if target_tree is None:
        target_tree = cKDTree(target_surface_points)
    prediction_tree = cKDTree(prediction_points)
    distance_to_target = target_tree.query(prediction_points, k=1, workers=-1)[0]
    distance_to_prediction = prediction_tree.query(
        target_surface_points, k=1, workers=-1
    )[0]
    return float(np.percentile(np.concatenate([distance_to_target, distance_to_prediction]), 95))


def binary_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing_zyx: tuple[float, ...],
    target_surface_points: np.ndarray | None = None,
    target_tree: cKDTree | None = None,
) -> dict[str, float | int]:
    prediction = prediction.astype(bool, copy=False)
    target = target.astype(bool, copy=False)
    pred_voxels = int(prediction.sum())
    gt_voxels = int(target.sum())
    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    labeled, components = label(
        prediction, structure=np.ones((3, 3, 3), dtype=bool)
    )
    sizes = np.bincount(labeled.ravel()) if components else np.asarray([0])
    largest_component = int(sizes[1:].max()) if components else 0
    positive = gt_voxels > 0
    return {
        "gt_is_positive": int(positive),
        "gt_voxels": gt_voxels,
        "pred_voxels": pred_voxels,
        "tp_voxels": tp,
        "fp_voxels": fp,
        "fn_voxels": fn,
        "largest_fp_component_voxels": largest_component if not positive else 0,
        "dice": float(2 * tp / (pred_voxels + gt_voxels)) if positive else np.nan,
        "sensitivity": float(tp / gt_voxels) if positive else np.nan,
        "hd95_mm": (
            surface_hd95_mm(
                prediction,
                target,
                spacing_zyx,
                target_surface_points=target_surface_points,
                target_tree=target_tree,
            )
            if positive
            else np.nan
        ),
        "normal_no_fp": float(pred_voxels == 0) if not positive else np.nan,
    }


def summarize(per_case: pd.DataFrame, max_sensitivity_drop: float) -> pd.DataFrame:
    baseline = per_case[per_case["configuration"] == "vision_only"]
    baseline_positive = baseline[baseline["gt_is_positive"] == 1]
    baseline_normal = baseline[baseline["gt_is_positive"] == 0]
    baseline_sensitivity = float(baseline_positive["sensitivity"].mean())
    baseline_fp_total = int(baseline_normal["fp_voxels"].sum())
    baseline_largest_total = int(baseline_normal["largest_fp_component_voxels"].sum())
    rows = []
    for configuration, group in per_case.groupby("configuration", sort=False):
        positive = group[group["gt_is_positive"] == 1]
        normal = group[group["gt_is_positive"] == 0]
        sensitivity = float(positive["sensitivity"].mean())
        finite_hd95 = positive.loc[np.isfinite(positive["hd95_mm"]), "hd95_mm"]
        row = {
            "configuration": configuration,
            "score_mode": group["score_mode"].iloc[0],
            "beta": group["beta"].iloc[0],
            "text_threshold": group["text_threshold"].iloc[0],
            "n_positive": len(positive),
            "n_normal": len(normal),
            "mean_dice_positive": float(positive["dice"].mean()),
            "mean_sensitivity_positive": sensitivity,
            "sensitivity_delta_vs_vision": sensitivity - baseline_sensitivity,
            "mean_hd95_positive_mm_finite": (
                float(finite_hd95.mean()) if len(finite_hd95) else float("inf")
            ),
            "positive_infinite_hd95_count": int((~np.isfinite(positive["hd95_mm"])).sum()),
            "mean_fn_delta_vs_vision": np.nan,
            "normal_no_fp_rate": float(normal["normal_no_fp"].mean()),
            "normal_total_fp_voxels": int(normal["fp_voxels"].sum()),
            "normal_total_fp_voxel_reduction": baseline_fp_total - int(normal["fp_voxels"].sum()),
            "normal_mean_largest_fp_component_voxels": float(
                normal["largest_fp_component_voxels"].mean()
            ),
            "normal_largest_component_voxel_reduction_total": (
                baseline_largest_total - int(normal["largest_fp_component_voxels"].sum())
            ),
            "safety_pass": bool(
                baseline_sensitivity - sensitivity <= max_sensitivity_drop + 1e-12
            ),
        }
        if configuration == "vision_only":
            row["mean_fn_delta_vs_vision"] = 0.0
        else:
            paired = positive[["case_id", "fn_voxels"]].merge(
                baseline_positive[["case_id", "fn_voxels"]],
                on="case_id",
                suffixes=("", "_vision"),
            )
            row["mean_fn_delta_vs_vision"] = float(
                (paired["fn_voxels"] - paired["fn_voxels_vision"]).mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if not 0 <= args.p_min < args.p_protect <= 1:
        raise ValueError("Require 0 <= p_min < p_protect <= 1.")
    if any(beta < 0 for beta in args.betas):
        raise ValueError("All beta values must be non-negative.")
    if any(not 0 <= value < 1 for value in args.text_thresholds):
        raise ValueError("All text thresholds must satisfy 0 <= threshold < 1.")

    annotation = pd.read_csv(args.annotation_csv, encoding="utf-8-sig")
    scores = pd.read_csv(args.normal_scores_csv, encoding="utf-8-sig")
    required_annotation = {
        "case_id", "probability_path", "gt_mask_path", "image_path"
    }
    missing = required_annotation - set(annotation.columns)
    if missing:
        raise ValueError(
            f"Inference annotation lacks columns {sorted(missing)}. Run the updated "
            "inference_eval.py with --save_probabilities."
        )
    missing_scores = set(SCORE_COLUMNS.values()) - set(scores.columns)
    if missing_scores:
        raise ValueError(f"Normal score CSV lacks columns: {sorted(missing_scores)}")
    table = annotation.merge(scores, on="case_id", how="inner", validate="one_to_one")
    table = table[
        table["probability_path"].notna()
        & table["gt_mask_path"].notna()
        & table["image_path"].notna()
    ].reset_index(drop=True)
    if table.empty:
        raise RuntimeError("No labeled cases with saved probability volumes were found.")

    records: list[dict[str, object]] = []
    for index, row in table.iterrows():
        probability_path = Path(str(row["probability_path"]))
        if not probability_path.is_file():
            raise FileNotFoundError(probability_path)
        probability = np.load(probability_path, allow_pickle=False).astype(np.float32)
        target, spacing_zyx = load_ground_truth(
            str(row["gt_mask_path"]), str(row["image_path"])
        )
        if probability.shape != target.shape:
            raise ValueError(
                f"Shape mismatch for {row['case_id']}: probability={probability.shape}, "
                f"target={target.shape}"
            )
        target_surface_points = None
        target_tree = None
        if target.any():
            target_surface = target & ~binary_erosion(
                target,
                structure=np.ones((3, 3, 3), dtype=bool),
                border_value=0,
            )
            target_surface_points = np.argwhere(target_surface).astype(np.float32)
            target_surface_points *= np.asarray(spacing_zyx, dtype=np.float32)
            target_tree = cKDTree(target_surface_points)
        baseline_prediction = probability >= args.prob_threshold
        baseline_metrics = binary_metrics(
            baseline_prediction,
            target,
            spacing_zyx,
            target_surface_points,
            target_tree,
        )
        records.append(
            {
                "case_id": row["case_id"],
                "configuration": "vision_only",
                "score_mode": "vision_only",
                "beta": 0.0,
                "text_threshold": np.nan,
                "p_normal": np.nan,
                **baseline_metrics,
            }
        )
        for score_mode, score_column in SCORE_COLUMNS.items():
            p_normal = float(row[score_column])
            for text_threshold in args.text_thresholds:
                for beta in args.betas:
                    final_probability = suppress_probability(
                        probability,
                        p_normal,
                        beta,
                        args.p_min,
                        args.p_protect,
                        text_threshold,
                    )
                    prediction = final_probability >= args.prob_threshold
                    configuration = (
                        f"{score_mode}__tau{text_threshold:g}__beta{beta:g}"
                    )
                    records.append(
                        {
                            "case_id": row["case_id"],
                            "configuration": configuration,
                            "score_mode": score_mode,
                            "beta": beta,
                            "text_threshold": text_threshold,
                            "p_normal": p_normal,
                            **binary_metrics(
                                prediction,
                                target,
                                spacing_zyx,
                                target_surface_points,
                                target_tree,
                            ),
                        }
                    )
        if (index + 1) % 50 == 0 or index + 1 == len(table):
            print(f"[SWEEP] processed={index + 1}/{len(table)}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_case = pd.DataFrame(records)
    summary = summarize(per_case, args.max_sensitivity_drop)
    per_case_path = output_dir / "per_case_suppression_metrics.csv"
    summary_path = output_dir / "suppression_sweep_summary.csv"
    per_case.to_csv(per_case_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    metadata = {
        "master_equation": "Zf=stopgrad(Zv)-beta*s*w(stopgrad(Pv))",
        "p_min": args.p_min,
        "p_protect": args.p_protect,
        "betas": args.betas,
        "text_thresholds": args.text_thresholds,
        "prob_threshold": args.prob_threshold,
        "max_sensitivity_drop": args.max_sensitivity_drop,
        "processed_labeled_cases": len(table),
    }
    (output_dir / "sweep_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    printable = summary.sort_values(
        ["safety_pass", "normal_total_fp_voxel_reduction", "mean_dice_positive"],
        ascending=[False, False, False],
    )
    print(printable.head(25).to_string(index=False))
    print(f"[DONE] summary={summary_path}")
    print(f"[DONE] per_case={per_case_path}")


if __name__ == "__main__":
    main()
