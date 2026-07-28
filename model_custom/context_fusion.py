"""Grounded, vision-preserving text fusion blocks for 3D segmentation."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch import nn


def _stable_multihead_attention(module, query, key, value, **kwargs):
    """Run only multimodal MHA in FP32 to avoid unstable BF16 batched GEMM."""
    if os.environ.get("LLMSEG_FORCE_FP32_MHA", "0") != "1":
        return module(query, key, value, **kwargs)

    output_dtype = query.dtype
    with torch.autocast(device_type=query.device.type, enabled=False):
        output, weights = module(
            query.float(),
            key.float(),
            value.float(),
            **kwargs,
        )
    return output.to(output_dtype), weights


if os.environ.get("LLMSEG_FORCE_FP32_MHA", "0") == "1":
    print(
        "[INFO] Multimodal attention precision: FP32 "
        "(LLMSEG_FORCE_FP32_MHA=1)"
    )


class DICOMMetadataEncoder(nn.Module):
    """Encode normalized DICOM numbers and categorical acquisition metadata."""

    def __init__(
        self,
        numeric_dim: int,
        category_sizes,
        output_dim: int = 256,
        category_embedding_dim: int = 16,
    ):
        super().__init__()
        self.numeric_dim = int(numeric_dim)
        self.category_sizes = tuple(int(size) for size in category_sizes)
        self.category_embeddings = nn.ModuleList(
            [
                nn.Embedding(size, category_embedding_dim, padding_idx=1)
                for size in self.category_sizes
            ]
        )
        input_dim = self.numeric_dim + len(self.category_sizes) * category_embedding_dim
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, numeric, categorical):
        if numeric is None or categorical is None:
            raise ValueError("Both numeric and categorical DICOM tensors are required.")
        categorical = categorical.long()
        embedded = [
            embedding(categorical[:, index].clamp(0, embedding.num_embeddings - 1))
            for index, embedding in enumerate(self.category_embeddings)
        ]
        features = torch.cat([numeric.float(), *embedded], dim=-1)
        return self.encoder(features)


class ZeroResidualFiLM3D(nn.Module):
    """Apply acquisition conditioning as an initially exact identity residual."""

    def __init__(self, channels: int, metadata_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels, affine=False)
        self.to_gamma_beta = nn.Linear(metadata_dim, channels * 2)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)
        self.logit_alpha = nn.Parameter(torch.tensor(-4.59511985))  # sigmoid=0.01

    def forward(self, vision, metadata, enabled: bool = True):
        if not enabled or metadata is None:
            zero = vision.new_zeros(())
            return vision, {"alpha": zero, "residual_rms": zero}
        gamma, beta = self.to_gamma_beta(metadata).chunk(2, dim=-1)
        broadcast_shape = (vision.shape[0], vision.shape[1], 1, 1, 1)
        delta = (
            gamma.view(broadcast_shape) * self.norm(vision)
            + beta.view(broadcast_shape)
        )
        alpha = torch.sigmoid(self.logit_alpha)
        residual = alpha * delta
        return vision + residual, {
            "alpha": alpha,
            "residual_rms": residual.float().square().mean().sqrt(),
        }


class HybridConceptExtractor(nn.Module):
    """Compress LLM tokens into interpretable and open-ended concept queries."""

    STRUCTURED_CONCEPTS = (
        "hemorrhage_burden",
        "trauma",
        "neurologic_symptom",
        "anatomy",
        "laterality",
        "uncertainty",
    )

    def __init__(self, text_dim: int, num_open_queries: int = 2, num_heads: int = 8):
        super().__init__()
        self.num_open_queries = int(num_open_queries)
        self.num_queries = len(self.STRUCTURED_CONCEPTS) + self.num_open_queries
        self.queries = nn.Parameter(torch.empty(1, self.num_queries, text_dim))
        nn.init.normal_(self.queries, std=0.02)
        self.cross_attn = nn.MultiheadAttention(
            text_dim, num_heads=num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(text_dim)
        self.ffn = nn.Sequential(
            nn.Linear(text_dim, text_dim * 2),
            nn.GELU(),
            nn.Linear(text_dim * 2, text_dim),
        )
        self.norm2 = nn.LayerNorm(text_dim)

    def forward(self, text_tokens, attention_mask=None):
        # expand() creates a stride-0 batch dimension. Materialize it before
        # BF16 GEMM/MHA: the cu132 Blackwell stack has produced
        # CUBLAS_STATUS_INTERNAL_ERROR in backward for batch sizes > 1 when
        # this view reaches the attention projections.
        queries = self.queries.expand(
            text_tokens.shape[0], -1, -1
        ).contiguous()
        text_tokens = text_tokens.contiguous()
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        # need_weights=False dispatches through fused SDPA in recent PyTorch
        # releases. Torch 2.12/cu132 intermittently raises
        # cudaErrorIllegalAddress for this variable-length BF16 path on sm_120.
        # The explicit attention-weight path uses stable matmul/softmax kernels;
        # the matrix is tiny here (concept queries x <=128 text tokens).
        attended, weights = _stable_multihead_attention(
            self.cross_attn,
            queries,
            text_tokens,
            text_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        concepts = self.norm1(queries + attended)
        concepts = self.norm2(concepts + self.ffn(concepts))
        if self.training:
            # Do not keep the diagnostic attention matrix in context_aux.
            weights = None
        return concepts, weights


class GroundedResidualFusion3D(nn.Module):
    """Inject concepts as a small, confidence-controlled spatial residual."""

    def __init__(self, channels: int, text_dim: int, num_heads: int = 8):
        super().__init__()
        self.concept_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, channels),
        )
        self.cross_attn = nn.MultiheadAttention(
            channels, num_heads=num_heads, batch_first=True
        )
        self.residual_proj = nn.Linear(channels, channels)
        nn.init.normal_(self.residual_proj.weight, std=1e-3)
        nn.init.zeros_(self.residual_proj.bias)

        self.compatibility = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.GELU(),
            nn.Linear(channels, 1),
        )
        nn.init.zeros_(self.compatibility[-1].weight)
        nn.init.zeros_(self.compatibility[-1].bias)

        # A small non-zero start preserves gradients without perturbing vision much.
        self.logit_alpha = nn.Parameter(torch.tensor(-4.59511985))  # sigmoid=0.01
        self.norm = nn.LayerNorm(channels)

    def forward(
        self,
        vision,
        concepts,
        enabled=True,
        sample_gate=None,
        force_full_strength: bool = False,
    ):
        if not enabled or concepts is None:
            zero = vision.new_zeros(())
            return vision, {"confidence": zero, "residual_rms": zero}

        batch, channels, depth, height, width = vision.shape
        # transpose() is non-contiguous. Explicit materialization avoids
        # backend-dependent BF16 GEMM layout handling in MultiheadAttention.
        spatial = vision.flatten(2).transpose(1, 2).contiguous()
        concept_features = self.concept_proj(concepts).contiguous()
        delta, _ = _stable_multihead_attention(
            self.cross_attn,
            spatial,
            concept_features,
            concept_features,
        )
        delta = self.residual_proj(self.norm(delta))

        pooled_vision = spatial.mean(dim=1)
        pooled_concept = concept_features.mean(dim=1)
        confidence = torch.sigmoid(
            self.compatibility(torch.cat([pooled_vision, pooled_concept], dim=-1))
        )
        if pooled_concept.shape[0] > 1:
            shuffled_concept = pooled_concept.roll(shifts=1, dims=0)
        elif dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            gathered = [torch.zeros_like(pooled_concept) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, pooled_concept.detach())
            shuffled_concept = gathered[(dist.get_rank() + 1) % dist.get_world_size()]
        else:
            shuffled_concept = torch.zeros_like(pooled_concept)
        shuffled_confidence = torch.sigmoid(
            self.compatibility(torch.cat([pooled_vision, shuffled_concept], dim=-1))
        )
        effective_confidence = confidence
        if sample_gate is not None:
            effective_confidence = effective_confidence * sample_gate.reshape(
                -1, 1
            ).to(confidence.dtype)
        alpha = torch.sigmoid(self.logit_alpha)
        # Keep logit_alpha in the DDP autograd graph during forced-fusion
        # warmup while holding the forward value exactly at 1.0. Replacing
        # alpha with a newly-created constant made one parameter in every
        # fusion scale unused and caused DDP reduction failures.
        effective_alpha = (
            vision.new_ones(()) + alpha * 0.0
            if force_full_strength
            else alpha
        )
        if force_full_strength:
            effective_confidence = torch.ones_like(effective_confidence)
        residual = effective_alpha * effective_confidence.unsqueeze(1) * delta
        fused = spatial + residual
        fused = fused.transpose(1, 2).reshape(batch, channels, depth, height, width)
        stats = {
            "confidence": confidence.mean(),
            "confidence_vector": confidence.squeeze(-1),
            "shuffled_confidence_vector": shuffled_confidence.squeeze(-1),
            "residual_rms": residual.float().square().mean().sqrt(),
            "effective_alpha": effective_alpha,
        }
        return fused, stats
