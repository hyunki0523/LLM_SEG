#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from data.dataset import build_safe_clinical_prompt, read_dataset_table
from model_custom.text_encoder import TextContextEncoder
from model_custom.text_feature_cache import TextFeatureCache


def collect_prompts(csv_paths: list[str]) -> list[str]:
    prompts: set[str] = set()
    for csv_path in csv_paths:
        frame = read_dataset_table(csv_path)
        prompts.update(
            build_safe_clinical_prompt(
                frame, ("extracted_cc", "chief_complaint")
            ).values()
        )
    return sorted(prompts)


def load_frozen_text_encoder(
    llm_repo: str,
    device: torch.device,
) -> tuple[AutoTokenizer, TextContextEncoder, int]:
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            llm_repo, trust_remote_code=True
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            llm_repo, trust_remote_code=True, use_fast=False
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<SEG>"]}
    )
    model = AutoModel.from_pretrained(
        llm_repo,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    model.eval().requires_grad_(False)
    model.to(device)

    encoder = TextContextEncoder()
    encoder.llm = True
    encoder.transformer = model
    encoder.token_embedding = model.get_input_embeddings()
    encoder.pad_id = int(tokenizer.pad_token_id)
    encoder.eval()
    hidden_dim = int(model.config.hidden_size)
    return tokenizer, encoder, hidden_dim


def encode_prompt_batch(
    tokenizer,
    encoder: TextContextEncoder,
    prompts: list[str],
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        prompts,
        add_special_tokens=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    with torch.inference_mode():
        return encoder(
            encoded["input_ids"].to(device),
            context=None,
            attn_mask=encoded["attention_mask"].to(device),
            position_ids_from_mask=True,
        )


def repo_fingerprint(llm_repo: str) -> str:
    config_path = Path(llm_repo) / "config.json"
    if config_path.is_file():
        return hashlib.sha256(config_path.read_bytes()).hexdigest()
    return hashlib.sha256(llm_repo.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute no-soft-prompt frozen Llama token features."
    )
    parser.add_argument("--csv", action="append", required=True)
    parser.add_argument("--llm-repo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--commit-every", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    device = torch.device(args.device)
    prompts = collect_prompts(args.csv)
    print(f"[CACHE] unique safe prompts: {len(prompts)}")

    tokenizer, encoder, hidden_dim = load_frozen_text_encoder(
        args.llm_repo, device
    )
    expected_metadata = {
        "llm_repo": str(Path(args.llm_repo).resolve()),
        "llm_config_sha256": repo_fingerprint(args.llm_repo),
        "hidden_dim": hidden_dim,
        "max_length": int(args.max_length),
        "soft_prompt_mode": "disabled",
        "position_ids": "attention_mask_cumsum",
        "text_columns": ["extracted_cc", "chief_complaint"],
        "tokenizer_length": len(tokenizer),
    }

    cache = TextFeatureCache(args.output, read_only=False)
    if cache.metadata:
        mismatches = {
            key: (cache.metadata.get(key), value)
            for key, value in expected_metadata.items()
            if cache.metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "Existing cache metadata does not match this run:\n"
                + json.dumps(mismatches, indent=2, ensure_ascii=False)
            )
    else:
        cache.set_metadata(expected_metadata)

    pending = [prompt for prompt in prompts if not cache.contains(prompt)]
    print(
        f"[CACHE] existing={len(cache)} pending={len(pending)} "
        f"output={cache.path}"
    )
    for batch_index, start in enumerate(
        range(0, len(pending), args.batch_size), start=1
    ):
        batch_prompts = pending[start : start + args.batch_size]
        hidden, mask = encode_prompt_batch(
            tokenizer, encoder, batch_prompts, args.max_length, device
        )
        for row, prompt in enumerate(batch_prompts):
            valid = hidden[row][mask[row].bool()].contiguous()
            cache.put(prompt, valid)
        if batch_index % args.commit_every == 0:
            cache.commit()
            print(
                f"[CACHE] written={min(start + len(batch_prompts), len(pending))}"
                f"/{len(pending)}"
            )
    cache.commit()
    print(f"[DONE] cache entries={len(cache)} path={cache.path}")
    cache.close()


if __name__ == "__main__":
    main()
