#!/usr/bin/env python
from __future__ import annotations

import argparse
import random

import torch

from model_custom.text_feature_cache import TextFeatureCache
from precompute_text_features import (
    collect_prompts,
    encode_prompt_batch,
    load_frozen_text_encoder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Numerically compare online and cached no-soft Llama features."
    )
    parser.add_argument("--csv", action="append", required=True)
    parser.add_argument("--llm-repo", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = TextFeatureCache(args.cache, read_only=True)
    prompts = collect_prompts(args.csv)
    random.Random(args.seed).shuffle(prompts)
    prompts = prompts[: min(args.samples, len(prompts))]
    device = torch.device(args.device)
    tokenizer, encoder, hidden_dim = load_frozen_text_encoder(
        args.llm_repo, device
    )
    if hidden_dim != int(cache.metadata["hidden_dim"]):
        raise ValueError("Online LLM and cache hidden dimensions differ.")

    max_error = 0.0
    sum_error = 0.0
    element_count = 0
    for start in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[start : start + args.batch_size]
        online, online_mask = encode_prompt_batch(
            tokenizer,
            encoder,
            batch_prompts,
            int(cache.metadata["max_length"]),
            device,
        )
        cached, cached_mask = cache.get_batch(batch_prompts, device)
        for row in range(len(batch_prompts)):
            online_valid = online[row][online_mask[row].bool()]
            cached_valid = cached[row][cached_mask[row].bool()]
            if online_valid.shape != cached_valid.shape:
                raise AssertionError(
                    f"Shape mismatch: {online_valid.shape} != {cached_valid.shape}"
                )
            error = (online_valid - cached_valid).abs().float()
            max_error = max(max_error, float(error.max().item()))
            sum_error += float(error.sum().item())
            element_count += error.numel()

    mean_error = sum_error / max(1, element_count)
    print(
        f"[EQUIVALENCE] prompts={len(prompts)} max_abs={max_error:.8g} "
        f"mean_abs={mean_error:.8g} atol={args.atol:.8g}"
    )
    if max_error > args.atol:
        raise SystemExit(
            "[FAIL] Cached features are not numerically equivalent at the "
            "requested tolerance."
        )
    print("[PASS] Online and cached valid-token hidden states are equivalent.")


if __name__ == "__main__":
    main()
