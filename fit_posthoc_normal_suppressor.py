#!/usr/bin/env python
"""Fit V7 case-level normal classifiers without touching the vision model.

The primary classifier consumes the last valid cached LLM token (the contextual
<SEG> state).  A TF-IDF classifier is trained on the exact same safe prompt and
labels.  Only cases with a readable GT mask are used as supervised examples;
normal means that the GT mask is exactly empty.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import joblib
import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from data.dataset import _case_id_column, build_safe_clinical_prompt, read_dataset_table
from model_custom.text_feature_cache import TextFeatureCache
from utils.mask_paths import find_case_mask_path


DEFAULT_MASK_DIRS = (
    "/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/FUtest_data/mask",
    "/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/FUdata/hemo_masks/thick_th0.56",
    "/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/FUdata/hemo_masks/thin_th0.56",
    "/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/FUdata/normal_masks",
    "/mnt/nas206/forGPU/lhyunki/NeuroCAD/data/FUdata/mask",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--valid-csv", required=True)
    parser.add_argument("--text-feature-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--vision-manifest",
        default=None,
        help="Optional supervision contract; invalid train rows are excluded from fitting.",
    )
    parser.add_argument("--mask-dir", action="append", default=None)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--calibration-fraction", type=float, default=0.1)
    parser.add_argument("--llm-alpha", type=float, default=1e-4)
    parser.add_argument("--tfidf-alpha", type=float, default=1e-5)
    parser.add_argument("--tfidf-max-features", type=int, default=50000)
    return parser.parse_args()


def explicit_mask_path(row: pd.Series) -> Path | None:
    if "mask_path" not in row or pd.isna(row["mask_path"]):
        return None
    value = str(row["mask_path"]).strip()
    return Path(value) if value else None


def mask_is_empty(path: Path) -> bool:
    try:
        image = sitk.ReadImage(str(path))
        return not bool(np.any(sitk.GetArrayViewFromImage(image) > 0))
    except RuntimeError:
        image = nib.load(str(path))
        return not bool(np.any(np.asanyarray(image.dataobj) > 0))


def collect_rows(
    csv_path: str,
    mask_dirs: Iterable[str],
) -> tuple[pd.DataFrame, dict[str, str], dict[str, int], dict[str, str]]:
    frame = read_dataset_table(csv_path)
    case_column = _case_id_column(frame)
    frame = frame.drop_duplicates(case_column, keep="first").reset_index(drop=True)
    prompts = build_safe_clinical_prompt(
        frame, ("extracted_cc", "chief_complaint"), dicom_prompt_mode="none"
    )
    targets: dict[str, int] = {}
    mask_paths: dict[str, str] = {}
    for _, row in frame.iterrows():
        case_id = str(row[case_column]).strip()
        path = find_case_mask_path(
            case_id,
            explicit_path=explicit_mask_path(row),
            search_dirs=mask_dirs,
        )
        if path is None:
            continue
        targets[case_id] = int(mask_is_empty(path))
        mask_paths[case_id] = str(path)
    return frame, prompts, targets, mask_paths


def pooled_features(
    cache: TextFeatureCache,
    prompts: list[str],
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for prompt in prompts:
        # build_safe_clinical_prompt appends <SEG>; the final valid hidden state
        # is therefore a contextual case representation, not a mean over pads.
        hidden = cache.get(prompt)
        rows.append(hidden[-1].float().numpy())
    matrix = np.stack(rows).astype(np.float32, copy=False)
    return normalize(matrix, norm="l2", axis=1, copy=False)


def fit_platt_classifier(
    x_fit,
    y_fit: np.ndarray,
    x_calibration,
    y_calibration: np.ndarray,
    alpha: float,
    seed: int,
) -> tuple[SGDClassifier, LogisticRegression]:
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        class_weight="balanced",
        max_iter=2000,
        tol=1e-5,
        average=True,
        random_state=seed,
    )
    classifier.fit(x_fit, y_fit)
    calibration_logits = classifier.decision_function(x_calibration).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", random_state=seed)
    calibrator.fit(calibration_logits, y_calibration)
    return classifier, calibrator


def calibrated_probability(classifier, calibrator, features) -> np.ndarray:
    logits = classifier.decision_function(features).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def classification_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    if len(np.unique(target)) != 2:
        return {}
    return {
        "roc_auc": float(roc_auc_score(target, probability)),
        "average_precision": float(average_precision_score(target, probability)),
        "brier": float(brier_score_loss(target, probability)),
        "balanced_accuracy_at_0.5": float(
            balanced_accuracy_score(target, probability >= 0.5)
        ),
    }


def main() -> None:
    args = parse_args()
    if not 0.05 <= args.calibration_fraction <= 0.4:
        raise ValueError("--calibration-fraction must be in [0.05, 0.4].")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dirs = tuple(args.mask_dir or DEFAULT_MASK_DIRS)

    train_frame, train_prompts, train_targets, train_masks = collect_rows(
        args.train_csv, mask_dirs
    )
    valid_frame, valid_prompts, valid_targets, valid_masks = collect_rows(
        args.valid_csv, mask_dirs
    )
    train_case_column = _case_id_column(train_frame)
    valid_case_column = _case_id_column(valid_frame)
    train_cases = [
        str(value).strip()
        for value in train_frame[train_case_column]
        if str(value).strip() in train_targets
    ]
    if args.vision_manifest:
        manifest = pd.read_csv(args.vision_manifest, encoding="utf-8-sig")
        required = {"case_id", "train_eligible"}
        if required - set(manifest.columns):
            raise ValueError("Vision manifest lacks case_id/train_eligible.")
        manifest["case_id"] = manifest["case_id"].astype(str).str.strip()
        eligible_text = manifest["train_eligible"].astype(str).str.strip().str.lower()
        if (~eligible_text.isin({"true", "false", "1", "0"})).any():
            raise ValueError("Vision manifest train_eligible contains invalid booleans.")
        eligible = manifest.loc[
            eligible_text.isin({"true", "1"}), "case_id"
        ]
        eligible_cases = set(eligible)
        before = len(train_cases)
        train_cases = [case_id for case_id in train_cases if case_id in eligible_cases]
        print(
            f"[VISION MANIFEST] classifier train cases {before} -> {len(train_cases)}"
        )
    valid_cases = [str(value).strip() for value in valid_frame[valid_case_column]]
    if len(train_cases) < 20 or len(set(train_targets[c] for c in train_cases)) != 2:
        raise RuntimeError("Insufficient labeled train cases or only one GT class was found.")

    cache = TextFeatureCache(args.text_feature_cache, read_only=True)
    cache_mode = str(cache.metadata.get("dicom_prompt_mode", "none")).lower()
    if cache_mode != "none":
        raise ValueError(
            "V7 CC suppression requires a CC-only cache (dicom_prompt_mode=none), "
            f"got {cache_mode!r}."
        )
    if not cache.contains("<SEG>"):
        raise RuntimeError(
            "The cache lacks the explicit empty-context '<SEG>' control. Re-run "
            "precompute_text_features.py against this existing cache; current code "
            "adds that one control prompt without replacing cached cases."
        )

    train_texts = [train_prompts[case_id] for case_id in train_cases]
    train_y = np.asarray([train_targets[case_id] for case_id in train_cases], dtype=np.int64)
    indices = np.arange(len(train_cases))
    fit_indices, calibration_indices = train_test_split(
        indices,
        test_size=args.calibration_fraction,
        random_state=args.seed,
        stratify=train_y,
    )

    print(f"[LABEL] train_labeled={len(train_cases)} valid_all={len(valid_cases)}")
    print(
        f"[LABEL] train_normal={int(train_y.sum())} "
        f"train_positive={int((1 - train_y).sum())}"
    )
    train_features = pooled_features(cache, train_texts)
    valid_texts = [valid_prompts[case_id] for case_id in valid_cases]
    valid_features = pooled_features(cache, valid_texts)
    empty_feature = pooled_features(cache, ["<SEG>"])

    llm_classifier, llm_calibrator = fit_platt_classifier(
        train_features[fit_indices],
        train_y[fit_indices],
        train_features[calibration_indices],
        train_y[calibration_indices],
        args.llm_alpha,
        args.seed,
    )
    real_llm_probability = calibrated_probability(
        llm_classifier, llm_calibrator, valid_features
    )
    rng = np.random.default_rng(args.seed)
    shuffled_indices = rng.permutation(len(valid_cases))
    shuffled_llm_probability = calibrated_probability(
        llm_classifier, llm_calibrator, valid_features[shuffled_indices]
    )
    empty_llm_probability = np.repeat(
        calibrated_probability(llm_classifier, llm_calibrator, empty_feature)[0],
        len(valid_cases),
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=args.tfidf_max_features,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    tfidf_fit = vectorizer.fit_transform([train_texts[index] for index in fit_indices])
    tfidf_calibration = vectorizer.transform(
        [train_texts[index] for index in calibration_indices]
    )
    tfidf_classifier, tfidf_calibrator = fit_platt_classifier(
        tfidf_fit,
        train_y[fit_indices],
        tfidf_calibration,
        train_y[calibration_indices],
        args.tfidf_alpha,
        args.seed,
    )
    valid_tfidf = vectorizer.transform(valid_texts)
    tfidf_probability = calibrated_probability(
        tfidf_classifier, tfidf_calibrator, valid_tfidf
    )

    score_frame = pd.DataFrame(
        {
            "case_id": valid_cases,
            "prompt": valid_texts,
            "p_normal_real_llm": real_llm_probability,
            "p_normal_shuffled_llm": shuffled_llm_probability,
            "shuffled_source_case_id": [valid_cases[i] for i in shuffled_indices],
            "p_normal_empty_llm": empty_llm_probability,
            "p_normal_tfidf": tfidf_probability,
            "gt_is_normal": [valid_targets.get(case_id, np.nan) for case_id in valid_cases],
            "gt_mask_path": [valid_masks.get(case_id, "") for case_id in valid_cases],
        }
    )
    score_path = output_dir / "valid_normal_scores.csv"
    score_frame.to_csv(score_path, index=False, encoding="utf-8-sig")

    labeled = score_frame["gt_is_normal"].notna().to_numpy()
    valid_y = score_frame.loc[labeled, "gt_is_normal"].to_numpy(dtype=np.int64)
    report = {
        "target": "GT mask exactly empty = 1",
        "train_labeled": len(train_cases),
        "valid_labeled": int(labeled.sum()),
        "train_normal": int(train_y.sum()),
        "train_positive": int((1 - train_y).sum()),
        "cache": str(Path(args.text_feature_cache).resolve()),
        "cache_metadata": cache.metadata,
        "vision_manifest": args.vision_manifest,
        "metrics": {
            "real_llm": classification_metrics(
                valid_y, real_llm_probability[labeled]
            ),
            "shuffled_llm": classification_metrics(
                valid_y, shuffled_llm_probability[labeled]
            ),
            "empty_llm": classification_metrics(
                valid_y, empty_llm_probability[labeled]
            ),
            "tfidf": classification_metrics(valid_y, tfidf_probability[labeled]),
        },
    }
    report_path = output_dir / "normal_classifier_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    joblib.dump(
        {
            "llm_classifier": llm_classifier,
            "llm_calibrator": llm_calibrator,
            "tfidf_vectorizer": vectorizer,
            "tfidf_classifier": tfidf_classifier,
            "tfidf_calibrator": tfidf_calibrator,
            "seed": args.seed,
            "target": report["target"],
        },
        output_dir / "posthoc_normal_classifiers.joblib",
    )
    cache.close()
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"[DONE] scores={score_path}")
    print(f"[DONE] report={report_path}")


if __name__ == "__main__":
    main()
