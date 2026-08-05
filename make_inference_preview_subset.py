#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a deterministic positive/normal validation subset."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--positive", type=int, default=64)
    parser.add_argument("--normal", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    df = read_table(input_path)
    if "class" not in df.columns:
        raise KeyError("The input table must contain a 'class' column.")
    if "영상일련번호ID" not in df.columns:
        raise KeyError("The input table must contain a '영상일련번호ID' column.")

    labels = df["class"].fillna("").astype(str).str.strip()
    positive_df = df[labels.str.contains("Intracranial hemorrhage or trauma", case=False)]
    normal_df = df[labels.str.fullmatch("Normal", case=False)]
    if positive_df.empty or normal_df.empty:
        raise RuntimeError(
            f"Could not make a balanced subset: positive={len(positive_df)}, "
            f"normal={len(normal_df)}"
        )

    positive_n = min(args.positive, len(positive_df))
    normal_n = min(args.normal, len(normal_df))
    positive_sample = positive_df.sample(n=positive_n, random_state=args.seed)
    normal_sample = normal_df.sample(n=normal_n, random_state=args.seed + 1)
    subset = pd.concat([positive_sample, normal_sample], ignore_index=True)
    subset = subset.sample(frac=1.0, random_state=args.seed + 2).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(
        f"[DONE] subset={output_path} total={len(subset)} "
        f"positive={positive_n} normal={normal_n} seed={args.seed}"
    )


if __name__ == "__main__":
    main()
