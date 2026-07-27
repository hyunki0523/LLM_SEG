#!/usr/bin/env python3
"""Preflight train/valid image paths before allocating GPUs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    for encoding in ("utf-8-sig", "cp949", "euc-kr", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def path_candidates(raw_value, rewrite_from: str, rewrite_to: str):
    raw = str(raw_value).strip()
    original = Path(raw)
    rewritten_raw = raw
    if rewrite_from and rewrite_to:
        if raw == rewrite_from:
            rewritten_raw = rewrite_to
        elif raw.startswith(rewrite_from + "/") or raw.startswith(rewrite_from + "\\"):
            suffix = raw[len(rewrite_from):].lstrip("/\\")
            rewritten_raw = f"{rewrite_to}/{suffix}"
    rewritten = Path(rewritten_raw)
    return [original] if rewritten == original else [rewritten, original]


def check_table(
    label: str,
    table_path: Path,
    rewrite_from: str,
    rewrite_to: str,
    skip_missing: bool,
) -> bool:
    if not table_path.exists():
        print(f"[ERROR] {label} not found: {table_path}")
        return False

    frame = read_table(table_path)
    if "image_path" not in frame.columns:
        print(f"[WARN] {label} has no image_path column; skipping image-path preflight.")
        return True

    values = frame["image_path"].dropna().astype(str)
    examples = []
    missing_count = 0
    used_original_count = 0
    for raw in values:
        candidates = path_candidates(raw, rewrite_from, rewrite_to)
        existing = next((candidate for candidate in candidates if candidate.exists()), None)
        if existing is None:
            missing_count += 1
            if len(examples) < 10:
                examples.append((raw, candidates))
        elif len(candidates) > 1 and existing == candidates[1]:
            used_original_count += 1

    if missing_count:
        level = "WARN" if skip_missing else "ERROR"
        action = "will skip" if skip_missing else "is missing"
        print(f"[{level}] {label} {action} {missing_count}/{len(values)} image files.")
        for raw, candidates in examples:
            print(f"  - raw={raw}")
            for candidate in candidates:
                print(f"    checked={candidate}")
    else:
        print(f"[CHECK] {label}: all {len(values)} image_path files exist.")

    if used_original_count:
        print(
            f"[INFO] {label}: kept the original CSV path for "
            f"{used_original_count} files missing from the rewrite destination."
        )
    return skip_missing or missing_count == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--valid-csv", required=True)
    parser.add_argument("--rewrite-from", default="")
    parser.add_argument("--rewrite-to", default="")
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()

    rewrite_from = args.rewrite_from.strip().rstrip("/\\")
    rewrite_to = args.rewrite_to.strip().rstrip("/\\")
    if bool(rewrite_from) ^ bool(rewrite_to):
        print("[ERROR] Set both --rewrite-from and --rewrite-to, or neither.")
        return 2

    valid = True
    valid &= check_table(
        "TRAIN_CSV",
        Path(args.train_csv),
        rewrite_from,
        rewrite_to,
        args.skip_missing,
    )
    valid &= check_table(
        "VALID_CSV",
        Path(args.valid_csv),
        rewrite_from,
        rewrite_to,
        args.skip_missing,
    )
    if not valid:
        print(
            "[HINT] Mount/stage the image volume, configure the rewrite paths, "
            "or explicitly enable missing-image skipping."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
