#!/usr/bin/env python3
"""Merge independently written SQLite text-feature shards into one cache."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from model_custom.text_feature_cache import TextFeatureCache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--source", required=True, action="append", type=Path)
    return parser.parse_args()


def comparable_metadata(metadata: dict[str, object]) -> dict[str, object]:
    result = dict(metadata)
    result.setdefault("dicom_prompt_mode", "none")
    result.setdefault("dicom_prompt_fields", [])
    return result


def main() -> int:
    args = parse_args()
    sources = [path.resolve() for path in args.source]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing cache shards: {missing}")
    target_path = args.target.resolve()
    if target_path in sources:
        raise ValueError("Target cache cannot also be a source shard.")

    source_caches = [TextFeatureCache(path, read_only=True) for path in sources]
    try:
        expected_metadata = comparable_metadata(source_caches[0].metadata)
        for cache in source_caches[1:]:
            if comparable_metadata(cache.metadata) != expected_metadata:
                raise ValueError(
                    f"Shard metadata mismatch: {cache.path} does not match {sources[0]}"
                )
    finally:
        for cache in source_caches:
            cache.close()

    target = TextFeatureCache(target_path, read_only=False)
    if target.metadata:
        if comparable_metadata(target.metadata) != expected_metadata:
            target.close()
            raise ValueError("Target metadata does not match the source shards.")
    else:
        target.set_metadata(expected_metadata)

    before = len(target)
    for index, source in enumerate(sources):
        alias = f"shard_{index}"
        target.connection.execute(f"ATTACH DATABASE ? AS {alias}", (str(source),))
        try:
            target.connection.execute(
                f"INSERT OR IGNORE INTO main.features "
                f"SELECT cache_key, prompt, seq_len, hidden_dim, dtype, data "
                f"FROM {alias}.features"
            )
            target.connection.commit()
        finally:
            target.connection.execute(f"DETACH DATABASE {alias}")
        print(f"[MERGE] source={source} target_entries={len(target)}")
    after = len(target)
    target.close()
    print(f"[DONE] merged={after - before} total={after} target={target_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
