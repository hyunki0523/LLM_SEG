#!/usr/bin/env python
"""Verify that a checkpoint came from the requested modality configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Cannot interpret boolean value: {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expect-context", required=True)
    parser.add_argument("--expect-dicom", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--reference-checkpoint",
        default=None,
        help="Require the main training recipe to match this checkpoint.",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    args_path = checkpoint.with_name("args.json")
    if not args_path.is_file():
        raise FileNotFoundError(
            f"{args.name}: missing sibling args.json; cannot prove modality contract: "
            f"{args_path}"
        )
    training_args = json.loads(args_path.read_text(encoding="utf-8"))
    actual_context = parse_bool(training_args.get("context", False))
    actual_dicom = parse_bool(training_args.get("use_dicom", False))
    expected_context = parse_bool(args.expect_context)
    expected_dicom = parse_bool(args.expect_dicom)
    mismatches = {}
    if actual_context != expected_context:
        mismatches["context"] = {"actual": actual_context, "expected": expected_context}
    if actual_dicom != expected_dicom:
        mismatches["use_dicom"] = {"actual": actual_dicom, "expected": expected_dicom}
    if mismatches:
        raise RuntimeError(
            f"{args.name} checkpoint violates the requested experiment contract:\n"
            + json.dumps(mismatches, ensure_ascii=False, indent=2)
        )
    if expected_dicom:
        schema_path = checkpoint.with_name("dicom_schema.json")
        if not schema_path.is_file():
            raise FileNotFoundError(
                f"{args.name}: DICOM FiLM checkpoint lacks dicom_schema.json: {schema_path}"
            )
    if args.reference_checkpoint:
        reference_args_path = Path(args.reference_checkpoint).with_name("args.json")
        if not reference_args_path.is_file():
            raise FileNotFoundError(
                f"Missing reference args.json for recipe comparison: {reference_args_path}"
            )
        reference_args = json.loads(reference_args_path.read_text(encoding="utf-8"))
        recipe_fields = (
            "train_csv",
            "valid_csv",
            "patch_size",
            "batch_size",
            "grad_accum",
            "lr",
            "epochs",
            "n_iter_per_epoch",
            "n_iter_valid",
            "positive_prob",
            "loss_fct",
            "deep_supervision",
            "deep_supervision_weights",
            "use_ema",
            "ema_decay",
        )
        recipe_mismatches = {
            field: {
                "actual": training_args.get(field),
                "reference": reference_args.get(field),
            }
            for field in recipe_fields
            if training_args.get(field) != reference_args.get(field)
        }
        if recipe_mismatches:
            raise RuntimeError(
                f"{args.name} training recipe does not match the Vision baseline; "
                "the ablation would be confounded:\n"
                + json.dumps(recipe_mismatches, ensure_ascii=False, indent=2)
            )
    print(
        f"[PASS] {args.name}: context={actual_context} use_dicom={actual_dicom} "
        f"checkpoint={checkpoint}"
    )


if __name__ == "__main__":
    main()
