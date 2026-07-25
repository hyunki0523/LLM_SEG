# -*- coding: utf-8 -*-
from __future__ import annotations
from collections.abc import Sequence
import importlib.machinery
import os
import sys
import types

os.environ.setdefault("USE_TORCHAUDIO", "0")
os.environ.setdefault("TRANSFORMERS_NO_TORCHAUDIO", "1")


def _install_torchaudio_stub():
    torchaudio_stub = types.ModuleType("torchaudio")
    torchaudio_stub.__spec__ = importlib.machinery.ModuleSpec("torchaudio", loader=None)

    transforms_stub = types.ModuleType("torchaudio.transforms")
    transforms_stub.__spec__ = importlib.machinery.ModuleSpec("torchaudio.transforms", loader=None)

    class _UnavailableRNNTLoss:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("torchaudio RNNTLoss is unavailable in this segmentation runtime")

    transforms_stub.RNNTLoss = _UnavailableRNNTLoss
    torchaudio_stub.transforms = transforms_stub

    sys.modules["torchaudio"] = torchaudio_stub
    sys.modules["torchaudio.transforms"] = transforms_stub


if os.environ.get("LLMSEG_DISABLE_TORCHAUDIO", "1") == "1":
    _install_torchaudio_stub()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks import UnetrBasicBlock
from monai.utils import ensure_tuple_rep

from transformers import AutoTokenizer, AutoModel
try:
    from transformers.utils import import_utils as _transformers_import_utils
    _transformers_import_utils._torchaudio_available = False
except Exception:
    pass
from .modules import ContextUnetrUpBlock, UnetOutUpBlock
from .sam import TwoWayTransformer
from .text_encoder import TextContextEncoder
from .context_fusion import (
    DICOMMetadataEncoder,
    GroundedResidualFusion3D,
    HybridConceptExtractor,
    ZeroResidualFiLM3D,
)
from .pe import PositionalEncodingGenerator3D
from transformers import BitsAndBytesConfig
import traceback


def _load_auto_model(repo, **kwargs):
    attn_impl = os.environ.get("LLM_ATTN_IMPLEMENTATION", "sdpa").strip()
    if attn_impl:
        try:
            return AutoModel.from_pretrained(
                repo,
                attn_implementation=attn_impl,
                **kwargs,
            )
        except (TypeError, ValueError) as e:
            if "attn" not in str(e).lower():
                raise
            print(f"[WARN] attn_implementation={attn_impl} is not supported here. Falling back to default attention. ({e})")
    return AutoModel.from_pretrained(repo, **kwargs)


class STUNetEncoder(nn.Module):
    def __init__(
        self,
        input_channels,
        depth=[1] * 6,
        dims=[16 * x for x in [1, 2, 4, 8, 16, 16]],
        conv_kernel_sizes=[[3, 3, 3]] * 6,
        pool_op_kernel_sizes=[[1, 1, 1], [1, 2, 2], [1, 2, 2], [1, 2, 2], [2, 2, 2]],
    ):
        super().__init__()
        self.conv_op = nn.Conv3d
        self.input_channels = input_channels
        self.dims = dims
        self.depth = depth

        # padding 설정
        self.conv_pad_sizes = []
        for krnl in conv_kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])

        num_pool = len(pool_op_kernel_sizes)
        assert num_pool == len(dims) - 1

        # encoder block 생성
        self.conv_blocks_context = nn.ModuleList()
        stage = nn.Sequential(
            BasicResBlock(input_channels, dims[0], conv_kernel_sizes[0], self.conv_pad_sizes[0], use_1x1conv=True),
            *[BasicResBlock(dims[0], dims[0], conv_kernel_sizes[0], self.conv_pad_sizes[0]) for _ in range(depth[0] - 1)]
        )
        self.conv_blocks_context.append(stage)

        for d in range(1, num_pool + 1):
            stage = nn.Sequential(
                BasicResBlock(
                    dims[d - 1],
                    dims[d],
                    conv_kernel_sizes[d],
                    self.conv_pad_sizes[d],
                    stride=pool_op_kernel_sizes[d - 1],
                    use_1x1conv=True,
                ),
                *[
                    BasicResBlock(dims[d], dims[d], conv_kernel_sizes[d], self.conv_pad_sizes[d])
                    for _ in range(depth[d] - 1)
                ]
            )
            self.conv_blocks_context.append(stage)

    def forward(self, x):
        skips = []
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)  # skip connection 저장
        x = self.conv_blocks_context[-1](x)
        return x, skips  # 최종 feature map과 skip connections 반환


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deep_supervision = True


class STUNet(nn.Module):
    def __init__(
        self,
        input_channels,
        num_classes,
        depth=[1, 1, 1, 1, 1, 1],
        dims=[32, 64, 128, 256, 512, 512],
        pool_op_kernel_sizes=None,
        conv_kernel_sizes=None,
        enable_deep_supervision=True,
        return_features=False,
        context=True,
        text_encoder="auto",
        llm_repo: str = "meta-llama/Llama-2-7b-chat-hf",
        use_lora: bool = False,
        use_cls_head: bool = False,
        use_dicom: bool = False,
        dicom_numeric_dim: int = 10,
        dicom_category_sizes=(2, 2),
        dicom_embedding_dim: int = 256,
    ):
        super().__init__()
        self.conv_op = nn.Conv3d
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.final_nonlin = lambda x: x
        self.decoder = Decoder()
        self.decoder.deep_supervision = enable_deep_supervision
        self.upscale_logits = False

        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_pad_sizes = []
        self.return_features = return_features
        self.context = context
        self.text_encoder_name = text_encoder
        self.llm_repo = llm_repo
        self.use_lora = use_lora
        self.use_cls_head = use_cls_head
        self.use_dicom = bool(use_dicom)
        if str(self.llm_repo).lower() in ["False","false", "none", ""]:
            print("[INFO] Vision-Only.")
            self.tokenizer = None
            self.text_encoder = None
            self.contexts = None
            self.max_length = 0
            self.context_length = 0
            self.text_encoder = TextContextEncoder(embed_dim=512)
        else:
            # 기존 LLM 로딩 코드 (AutoModel.from_pretrained 등...)
            print(f"[INFO] LLM 백본 로딩 시도: {self.llm_repo}")


        self.dims = dims
        self.cls_head = None
        if self.use_cls_head:
            self.cls_head = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(1),
                nn.LayerNorm(dims[-1]),
                nn.Linear(dims[-1], 1),
            )
        
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])

        num_pool = len(pool_op_kernel_sizes)
        assert num_pool == len(dims) - 1

        # encoder
        self.conv_blocks_context = nn.ModuleList()
        stage = nn.Sequential(
            BasicResBlock(input_channels, dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0], use_1x1conv=True),
            *[BasicResBlock(dims[0], dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0]) for _ in range(depth[0] - 1)]
        )
        self.conv_blocks_context.append(stage)
        for d in range(1, num_pool + 1):
            stage = nn.Sequential(
                BasicResBlock(
                    dims[d - 1],
                    dims[d],
                    self.conv_kernel_sizes[d],
                    self.conv_pad_sizes[d],
                    stride=self.pool_op_kernel_sizes[d - 1],
                    use_1x1conv=True,
                ),
                *[
                    BasicResBlock(dims[d], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d])
                    for _ in range(depth[d] - 1)
                ]
            )
            self.conv_blocks_context.append(stage)

        # upsample_layers
        self.upsample_layers = nn.ModuleList()
        for u in range(num_pool):
            upsample_layer = Upsample_Layer_nearest(dims[-1 - u], dims[-2 - u], pool_op_kernel_sizes[-1 - u])
            self.upsample_layers.append(upsample_layer)

        # decoder
        self.conv_blocks_localization = nn.ModuleList()
        for u in range(num_pool):
            stage = nn.Sequential(
                BasicResBlock(
                    dims[-2 - u] * 2,
                    dims[-2 - u],
                    self.conv_kernel_sizes[-2 - u],
                    self.conv_pad_sizes[-2 - u],
                    use_1x1conv=True,
                ),
                *[
                    BasicResBlock(dims[-2 - u], dims[-2 - u], self.conv_kernel_sizes[-2 - u], self.conv_pad_sizes[-2 - u])
                    for _ in range(depth[-2 - u] - 1)
                ]
            )
            self.conv_blocks_localization.append(stage)
        
        # outputs
        self.seg_outputs = nn.ModuleList()
        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(nn.Conv3d(dims[-2 - ds], num_classes, kernel_size=1))

        self.upscale_logits_ops = []
        for usl in range(num_pool - 1):
            self.upscale_logits_ops.append(lambda x: x)

        # DICOM metadata is never serialized as LLM text. It conditions the
        # bottleneck and semantic decoder scales through identity-initialized FiLM.
        self.dicom_encoder = None
        self.dicom_bottleneck_film = None
        self.dicom_decoder_films = nn.ModuleList()
        self.dicom_aux = {}
        if self.use_dicom:
            self.dicom_encoder = DICOMMetadataEncoder(
                numeric_dim=dicom_numeric_dim,
                category_sizes=dicom_category_sizes,
                output_dim=dicom_embedding_dim,
            )
            self.dicom_bottleneck_film = ZeroResidualFiLM3D(
                dims[-1], dicom_embedding_dim
            )
            decoder_dims = list(reversed(dims[:-1]))
            self.dicom_decoder_films = nn.ModuleList(
                [
                    ZeroResidualFiLM3D(channels, dicom_embedding_dim)
                    for channels in decoder_dims[:3]
                ]
            )

        # multimodal text encoder
        if self.context and str(self.llm_repo).lower() not in ["false", "none", ""]:
            print(f"[LLM] llm_repo = {self.llm_repo}")
            # 1) 텍스트 인코더 래퍼
            self.text_encoder = TextContextEncoder(embed_dim=None)
            self.context_length = 8
            self.n_prompts = 2
            self.max_length = 128
            

            # freeze wrapper params by default
            for _, p in self.text_encoder.named_parameters():
                p.requires_grad_(False)

            # 2) LLM 토크나이저 & 모델 (Fallback 포함)
            repos_to_try = [self.llm_repo]
            model = None
            
            for repo in repos_to_try:
                if not repo: continue
                try:
                    print(f"[INFO] LLM 백본 로딩 시도: {repo}")
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
                    except Exception as e:
                        print(f"[WARN] Tokenizer 로딩 에러 ({e}), use_fast=False 로 재시도합니다.")
                        self.tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True, use_fast=False)
                    
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                    self.tokenizer.padding_side = "left"
                    self.tokenizer.add_special_tokens({"additional_special_tokens": ["<SEG>"]})
                    
                    #Accelerate (Multi-GPU) 환경을 위한 안전한 Device 할당
                    local_rank = int(os.environ.get("LOCAL_RANK", 0))
                    
                    # GPU 아키텍처 동적 확인 (Blackwell 호환성 체크)
                    is_blackwell = False
                    if torch.cuda.is_available():
                        major, _ = torch.cuda.get_device_capability(local_rank)
                        if major >= 12: # sm_120 (Blackwell) 이상
                            is_blackwell = True
                            
                    if is_blackwell:
                        print(f"[INFO] Blackwell(sm_120) 아키텍처 감지. GPU 커널 에러 우회를 위해 CPU에서 bfloat16 로딩 후 GPU로 직접 이동합니다.")
                        # 1) device_map=None으로 설정하여 Accelerate의 CPU 고정 훅(hook)이 걸리지 않게 순수 CPU 메모리에 올립니다.
                        model = _load_auto_model(
                            repo,
                            torch_dtype=torch.bfloat16,
                            device_map=None, 
                            trust_remote_code=True,
                            low_cpu_mem_usage=True,
                        )
                        # 2) [경고] 여기서 수동으로 model.to(local_rank)를 호출하면 
                        # LLM만 GPU에 가고 U-Net은 CPU에 남는 "이산가족(Mixed-device)" 상태가 되어 
                        # NCCL(DDP 통신) 에러가 무조건 발생합니다!
                        # 따라서 절대 GPU로 미리 옮기지 말고 CPU에 둔 상태로 리턴합니다.
                        # 이후 train.py의 accelerator.prepare()가 모델 전체를 안전하게 GPU로 동기화합니다.
                        load_msg = "bfloat16 네이티브 (CPU 우회 후 자동 할당)"
                    else:
                        print(f"[INFO] 이전 GPU 아키텍처 감지. 8-bit 양자화를 적용하여 로딩을 수행합니다.")
                        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                        model = _load_auto_model(
                            repo,
                            quantization_config=bnb_config,
                            device_map={"": local_rank},
                            trust_remote_code=True,
                            low_cpu_mem_usage=True,
                        )
                        load_msg = "8-bit 양자화"
                    
                    # VRAM 폭증 방지
                    # Frozen prompt tuning uses short sequences and ample VRAM.
                    # Checkpointed LLM execution was unstable on Blackwell, so
                    # keep it opt-in unless LoRA training requires it.
                    use_llm_checkpointing = self.use_lora or (
                        os.environ.get("LLMSEG_LLM_GRADIENT_CHECKPOINTING", "0") == "1"
                    )
                    if use_llm_checkpointing:
                        model.gradient_checkpointing_enable()
                    elif hasattr(model, "gradient_checkpointing_disable"):
                        model.gradient_checkpointing_disable()
                    if hasattr(model, "config"):
                        model.config.use_cache = False
                    
                    if self.use_lora:
                        from peft import get_peft_model, LoraConfig

                        print(f"[INFO] PEFT LoRA 설정 적용 중...")
                        peft_config = LoraConfig(
                            inference_mode=False,
                            r=8,
                            lora_alpha=16,
                            lora_dropout=0.05,
                            target_modules=["q_proj", "v_proj"]
                        )
                        model = get_peft_model(model, peft_config)
                        model.print_trainable_parameters()
                    
                    print(f"[INFO] LLM 백본 로딩 성공 ({load_msg}): {repo}")
                    self.llm_repo = repo
                    break




                    # #  4-bit 양자화 설정 추가
                    # bnb_config = BitsAndBytesConfig(
                    #     load_in_4bit=True,
                    #     bnb_4bit_use_double_quant=True,
                    #     bnb_4bit_quant_type="nf4",
                    #     bnb_4bit_compute_dtype=torch.float16
                    # )

                    # 모델 로드 시 양자화 적용 및 device_map 변경
                    # model = AutoModel.from_pretrained(
                    #     repo,
                    #     quantization_config=bnb_config,  # 양자화 설정 주입
                    #     device_map="auto",               # "cpu"에서 "auto"로 반드시 변경!
                    #     trust_remote_code=True,
                    #     low_cpu_mem_usage=True,
                    # )
                    
                    # print(f"[INFO] LLM 백본 로딩 성공 (4-bit 양자화): {repo}")
                    # self.llm_repo = repo
                    # break
                except Exception as e:
                    print(f"[WARN] LLM 백본 '{repo}' 로딩 실패. 원인: {e}")
                    print("[WARN] Full LLM loading traceback:")
                    traceback.print_exc()
                    model = None
                    continue
            
            if model is None:
                raise RuntimeError(f"지정된 LLM 백본을 로드할 수 없습니다: {repos_to_try}")

            try:
                model.resize_token_embeddings(len(self.tokenizer))
            except Exception:
                pass

            # ✅ backbone만 사용
            self.text_encoder.llm = True
            self.text_encoder.transformer = model
            self.text_encoder.token_embedding = model.get_input_embeddings()
            # IMPORTANT: align pad id with tokenizer (train.py must also pad with this id)
            self.text_encoder.pad_id = int(self.tokenizer.pad_token_id)

            # for _, p in self.text_encoder.transformer.named_parameters():
            #     p.requires_grad_(False)

            # 3) 임베딩 차원 동적 획득
            token_embed_dim = getattr(self.text_encoder.transformer.config, "hidden_size", None)
            if token_embed_dim is None:
                token_embed_dim = getattr(self.text_encoder.token_embedding, "embedding_dim", None)

            # 4) learnable prompt contexts
            self.context_length = 16  # 필요에 따라 16~32 정도로 조절 가능
            self.contexts = nn.Parameter(torch.randn(1, self.context_length, token_embed_dim))
            self.max_length = 128



            # 5) 텍스트→비전 projection & cross-attn
            self.concept_extractor = HybridConceptExtractor(
                token_embed_dim, num_open_queries=2, num_heads=8
            )
            self.text_burden_head = nn.Sequential(
                nn.LayerNorm(token_embed_dim),
                nn.Linear(token_embed_dim, 1),
            )
            self.text_trauma_head = nn.Sequential(
                nn.LayerNorm(token_embed_dim),
                nn.Linear(token_embed_dim, 1),
            )
            decoder_dims = list(reversed(dims[:-1]))
            self.context_fusions = nn.ModuleList(
                [
                    GroundedResidualFusion3D(C, token_embed_dim, num_heads=8)
                    for C in decoder_dims[:3]
                ]
            )
            self.context_aux = {}
    def _align_and_concat(self, x, skip):
        if x.shape[2:] != skip.shape[2:]:
            Dx, Hx, Wx = x.shape[2:]
            Ds, Hs, Ws = skip.shape[2:]

            D = min(Dx, Ds)
            H = min(Hx, Hs)
            W = min(Wx, Ws)

            def center_crop(t, D, H, W):
                _, _, d, h, w = t.shape
                sd = (d - D) // 2
                sh = (h - H) // 2
                sw = (w - W) // 2
                return t[:, :, sd:sd + D, sh:sh + H, sw:sw + W]

            x = center_crop(x, D, H, W)
            skip = center_crop(skip, D, H, W)

        return torch.cat((x, skip), dim=1)

    def forward(
        self,
        x,
        report_in=None,
        attn_mask=None,
        dicom_numeric=None,
        dicom_categorical=None,
        cfg_scale=1.0,
        return_cls: bool = False,
        return_context_variants: bool = False,
        force_text_fusion: bool = False,
        context_patch_mask_probability: float = 0.0,
        context_hard_bypass_threshold: float = 0.0,
    ):
        if return_context_variants and report_in is not None and self.context:
            fused_x = x
            observed_mask_rate = x.new_zeros(())
            if self.training and context_patch_mask_probability > 0:
                grid_shape = (
                    max(1, x.shape[2] // 4),
                    max(1, x.shape[3] // 32),
                    max(1, x.shape[4] // 32),
                )
                keep = (
                    torch.rand(
                        x.shape[0], 1, *grid_shape, device=x.device
                    ) > context_patch_mask_probability
                ).to(x.dtype)
                keep = F.interpolate(keep, size=x.shape[2:], mode="nearest")
                fused_x = x * keep
                observed_mask_rate = 1.0 - keep.float().mean()
            raw_result = self._forward_impl(
                fused_x,
                report_in,
                attn_mask,
                dicom_numeric,
                dicom_categorical,
                force_drop_text=False,
                return_cls=return_cls,
                force_text_fusion=force_text_fusion,
            )
            raw_fused, cls_logits = raw_result if return_cls else (raw_result, None)
            saved_context_aux = self.context_aux
            saved_dicom_aux = self.dicom_aux
            vision_result = self._forward_impl(
                x,
                report_in,
                attn_mask,
                dicom_numeric,
                dicom_categorical,
                force_drop_text=True,
                return_cls=return_cls,
                force_text_fusion=False,
            )
            vision_logits, vision_cls_logits = (
                vision_result if return_cls else (vision_result, None)
            )
            self.context_aux = saved_context_aux
            self.context_aux["vision_patch_mask_rate"] = observed_mask_rate
            self.dicom_aux = saved_dicom_aux
            confidence_vectors = [
                stats["confidence_vector"]
                for name, stats in self.context_aux.items()
                if name.startswith("fusion_")
                and isinstance(stats, dict)
                and "confidence_vector" in stats
            ]
            hard_gate = None
            if confidence_vectors:
                aggregate_confidence = torch.stack(confidence_vectors).mean(dim=0)
                self.context_aux["aggregate_confidence"] = aggregate_confidence
                if not self.training and context_hard_bypass_threshold > 0:
                    hard_gate = (
                        aggregate_confidence >= context_hard_bypass_threshold
                    ).to(aggregate_confidence.dtype)

            def uncertainty_safe_fuse(vision_value, fused_value):
                if isinstance(vision_value, tuple):
                    return tuple(
                        uncertainty_safe_fuse(v, f)
                        for v, f in zip(vision_value, fused_value)
                    )
                if vision_value.shape[1] == 1:
                    probability = torch.sigmoid(vision_value)
                else:
                    probability = torch.softmax(vision_value, dim=1)[:, 1:2]
                uncertainty = 4.0 * probability * (1.0 - probability)
                if hard_gate is not None:
                    gate = hard_gate.view(-1, 1, 1, 1, 1)
                    uncertainty = uncertainty * gate
                return vision_value + uncertainty * (fused_value - vision_value)

            output = {
                "vision_logits": vision_logits,
                "raw_fused_logits": raw_fused,
                "fused_logits": uncertainty_safe_fuse(vision_logits, raw_fused),
            }
            if return_cls:
                output["cls_logits"] = vision_cls_logits
                output["raw_fused_cls_logits"] = cls_logits
            return output

        if False:
            # --- Classifier-Free Guidance (CFG) Inference ---
            # 1. Conditional Pass (Text = Normal)
            out_cond = self._forward_impl(
                x, report_in, attn_mask, dicom_numeric, dicom_categorical,
                force_drop_text=False, return_cls=return_cls
            )
            
            # 2. Unconditional Pass (Text = Dropped)
            out_uncond = self._forward_impl(
                x, report_in, attn_mask, dicom_numeric, dicom_categorical,
                force_drop_text=True, return_cls=return_cls
            )

            if return_cls:
                logits_cond, cls_logits = out_cond
                logits_uncond, _ = out_uncond
            else:
                logits_cond = out_cond
                logits_uncond = out_uncond
            
            # 3. CFG Extrapolation
            # deep_supervision이 켜져 있으면 tuple로 반환되므로 각각 계산
            if isinstance(logits_cond, tuple):
                guided_logits = tuple(uncond + cfg_scale * (cond - uncond) for cond, uncond in zip(logits_cond, logits_uncond))
            else:
                guided_logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)

            if return_cls:
                return guided_logits, cls_logits
            return guided_logits
        return self._forward_impl(
            x,
            report_in,
            attn_mask,
            dicom_numeric,
            dicom_categorical,
            force_drop_text=False,
            return_cls=return_cls,
            force_text_fusion=force_text_fusion,
        )

    def _forward_impl(
        self,
        x,
        report_in=None,
        attn_mask=None,
        dicom_numeric=None,
        dicom_categorical=None,
        force_drop_text=False,
        return_cls: bool = False,
        force_text_fusion: bool = False,
    ):
            skips = []
            seg_outputs = []
            features = []

            # ----- 인코더 -----
            for d in range(len(self.conv_blocks_context) - 1):
                x = self.conv_blocks_context[d](x)
                skips.append(x)
            x = self.conv_blocks_context[-1](x)

            dicom_embedding = None
            self.dicom_aux = {}
            if self.dicom_encoder is not None:
                if dicom_numeric is None or dicom_categorical is None:
                    raise ValueError(
                        "STUNet(use_dicom=True) requires DICOM tensors in every batch."
                    )
                dicom_embedding = self.dicom_encoder(
                    dicom_numeric.to(x.device), dicom_categorical.to(x.device)
                )
                x, film_stats = self.dicom_bottleneck_film(x, dicom_embedding)
                self.dicom_aux["bottleneck"] = film_stats

            # 인코더에서 나온 순수 이미지 피처맵들을 리스트로 묶어둠
            hidden_states_out = skips + [x]
            
            # ----- 텍스트-비전 정렬 -----
            concepts = None
            context_keep = None
            self.context_aux = {}
            if (
                not force_drop_text
                and getattr(self, "text_encoder", None) is not None
                and report_in is not None
            ):
                # LLM + Vision 모드: 항상 텍스트 브랜치를 실행하여 DDP 연산 그래프 유지
                emb_txt, full_attn_mask = self.text_encoder(
                    report_in.to(x.device), self.contexts, attn_mask=attn_mask
                )
                concepts, concept_attention = self.concept_extractor(
                    emb_txt, full_attn_mask
                )
                if self.training:
                    context_keep = (
                        torch.rand(concepts.shape[0], 1, 1, device=concepts.device) > 0.1
                    ).to(concepts.dtype)
                self.context_aux["concept_attention"] = concept_attention
                self.context_aux["text_burden_logits"] = self.text_burden_head(
                    concepts[:, 0]
                ).squeeze(-1)
                self.context_aux["text_trauma_logits"] = self.text_trauma_head(
                    concepts[:, 1]
                ).squeeze(-1)


            # 디코더에 넣기 위해 다시 분해
            cls_logits = None
            if return_cls:
                if self.cls_head is None:
                    raise RuntimeError("return_cls=True requires STUNet(use_cls_head=True).")
                cls_logits = self.cls_head(x).squeeze(1)

            # ----- 디코더 -----
            for u in range(len(self.conv_blocks_localization)):
                x = self.upsample_layers[u](x)
                skip = skips[-(u + 1)]
                
                x = self._align_and_concat(x, skip)
                x = self.conv_blocks_localization[u](x)
                if dicom_embedding is not None and u < len(self.dicom_decoder_films):
                    x, film_stats = self.dicom_decoder_films[u](x, dicom_embedding)
                    self.dicom_aux[f"decoder_{u}"] = film_stats
                if concepts is not None and u < len(self.context_fusions):
                    x, fusion_stats = self.context_fusions[u](
                        x,
                        concepts,
                        sample_gate=context_keep,
                        force_full_strength=force_text_fusion,
                    )
                    self.context_aux[f"fusion_{u}"] = fusion_stats
                
                seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))
                if getattr(self, "return_features", False):
                    features.append(x)

            if getattr(self.decoder, "deep_supervision", False):
                final_outs = tuple(
                    [seg_outputs[-1]] + [i(j) for i, j in zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])]
                )
            elif getattr(self, "return_features", False):
                final_outs = features[-1]
            else:
                final_outs = seg_outputs[-1]

            if return_cls:
                return final_outs, cls_logits
            return final_outs

    def interactive_alignment(self, hidden_states_out, report_in, attn_mask, x_in, force_drop_text=False):
        # emb_txt: [B, Seq_Len, C] 형태로 단어별 정보가 넘어옴
        
        emb_txt, full_attn_mask = self.text_encoder(report_in.to(x_in.device), self.contexts, attn_mask=attn_mask)
        
        # [Modality Dropout & Token Masking] 
        # 이렇게 하면 txt2vis, cross-attention 등의 하위 모듈이 항상 다 실행되므로 DDP 안전성 보장
        if self.training:
            # 1. 10% 확률로 문장 전체를 날림 (CFG를 위한 Unconditional 학습)
            use_text_global = (torch.rand(1, device=emb_txt.device) > 0.1).float()
            
            # 2. 문장이 살아있을 때, 15% 확률로 개별 단어(토큰)를 랜덤하게 가림 (Token Masking)
            B, Seq_Len, C = emb_txt.shape
            token_mask = (torch.rand(B, Seq_Len, 1, device=emb_txt.device) > 0.15).float()
            
            emb_txt = emb_txt * use_text_global * token_mask
        elif force_drop_text:
            # CFG Inference 시 Unconditional 패스를 위해 강제로 텍스트를 지움
            emb_txt = emb_txt * 0.0
        
        # [FP16 Safety] 정규화 전 강제로 float32로 캐스팅하여 overflow/underflow(Inf/NaN) 방지
        emb_txt_fp32 = emb_txt.float()
        emb_txt_fp32 = F.layer_norm(emb_txt_fp32, (emb_txt_fp32.shape[-1],))
        emb_txt_fp32 = emb_txt_fp32 / (emb_txt_fp32.norm(dim=-1, keepdim=True) + 1e-6)
        emb_txt = emb_txt_fp32.to(emb_txt.dtype)

        report_l = []
        for j, proj_layer in enumerate(self.txt2vis):
            # U-Net 채널에 맞게 Projection -> [B, Seq_Len, dims[j]]
            projected_txt = proj_layer(emb_txt)
            report_l.append(F.layer_norm(projected_txt, normalized_shape=[self.dims[j]]))

        h_offset = 0
        for j, text_vis in enumerate(zip(report_l[h_offset:], hidden_states_out[h_offset:])):
            txt, vis = text_vis
            
            image_pe = self.pe_layers[j + h_offset](vis)

            # Inference 시 배치 크기가 다를 경우 (예: Sliding window로 이미지는 B개인데 텍스트는 1개일 때)
            if txt.shape[0] != vis.shape[0]:
                txt = txt.expand(vis.shape[0], -1, -1) # repeat_interleave보다 메모리 효율적
                if full_attn_mask is not None:
                    curr_mask = full_attn_mask.expand(vis.shape[0], -1)
                else:
                    curr_mask = None
            else:
                curr_mask = full_attn_mask

            # Cross-Attention: 텍스트(queries)와 이미지(keys)를 양방향으로 결합하여 둘 다 획득
            queries_out, vis_out = self.attn_transformer[j + h_offset](
                image_embedding=vis, image_pe=image_pe, point_embedding=txt, attn_mask=curr_mask
            )
            
            # <SEG> 토큰의 최종 임베딩 추출 (padding_side="left" 및 <SEG>가 문장 맨 뒤에 오므로 -1 인덱스 추출) -> [B, C]
            seg_feat = queries_out[:, -1, :]
            
            # Gating 가중치 생성 -> [B, C]
            # 2 * sigmoid(0) = 1.0, so zero-initialized gates start as identity.
            gate = 2.0 * torch.sigmoid(self.gating_layers[j + h_offset](seg_feat))
            
            # 이미지 피쳐맵 vis_out([B, C, D, H, W])에 채널별 게이팅 적용하여 저장!
            hidden_states_out[j + h_offset] = vis_out * gate.view(vis_out.shape[0], vis_out.shape[1], 1, 1, 1)

        return hidden_states_out, emb_txt, None

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x


class BasicResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, padding=1, stride=1, use_1x1conv=False):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act2 = nn.LeakyReLU(inplace=True)

        if use_1x1conv:
            self.conv3 = nn.Conv3d(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="leaky_relu")
        nn.init.constant_(self.conv1.bias, 0.0)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="leaky_relu")
        nn.init.constant_(self.conv2.bias, 0.0)
        if self.conv3 is not None:
            nn.init.kaiming_normal_(self.conv3.weight, nonlinearity="leaky_relu")
            nn.init.constant_(self.conv3.bias, 0.0)

    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)


class Upsample_Layer_nearest(nn.Module):
    def __init__(self, input_channels, output_channels, pool_op_kernel_size, mode="nearest"):
        super().__init__()
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode

        if isinstance(self.pool_op_kernel_size, (list, tuple)):
            self.scale_factor = tuple(self.pool_op_kernel_size)
        else:
            self.scale_factor = (self.pool_op_kernel_size,) * 3

        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="leaky_relu")
        nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


def get_stunet_small(
    num_input_channels,
    num_classes,
    strides=[[1, 1, 1], [1, 2, 2], [1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]],
    enable_deep_supervision=True,
    return_features=False,
    context=True,
    llm_repo: str = "meta-llama/Llama-2-7b-chat-hf",
    use_lora: bool = False,
    use_cls_head: bool = False,
    use_dicom: bool = False,
    dicom_numeric_dim: int = 10,
    dicom_category_sizes=(2, 2),
):
    kernel_sizes = [[3, 3, 3]] * 6
    if len(strides) > 5:
        strides = strides[:5]
    while len(strides) < 5:
        strides.append([1, 1, 1])

    return STUNet(
        num_input_channels,
        num_classes,
        depth=[1] * 6,
        dims=[16 * x for x in [1, 2, 4, 8, 16, 16]],
        pool_op_kernel_sizes=strides,
        conv_kernel_sizes=kernel_sizes,
        enable_deep_supervision=enable_deep_supervision,
        return_features=return_features,
        context=context,
        llm_repo=llm_repo,
        use_lora=use_lora,
        use_cls_head=use_cls_head,
        use_dicom=use_dicom,
        dicom_numeric_dim=dicom_numeric_dim,
        dicom_category_sizes=dicom_category_sizes,
    )


def get_stunet_base(
    num_input_channels,
    num_classes,
    strides=[[1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]],
    enable_deep_supervision=True,
    return_features=False,
    context=True,
    llm_repo: str = "meta-llama/Llama-2-7b-chat-hf",
    use_lora: bool = False,
    use_cls_head: bool = False,
    use_dicom: bool = False,
    dicom_numeric_dim: int = 10,
    dicom_category_sizes=(2, 2),
):
    kernel_sizes = [[3, 3, 3]] * 6
    if len(strides) > 5:
        strides = strides[:5]
    while len(strides) < 5:
        strides.append([1, 1, 1])

    return STUNet(
        num_input_channels,
        num_classes,
        depth=[1] * 6,
        dims=[32 * x for x in [1, 2, 4, 8, 16, 16]],
        pool_op_kernel_sizes=strides,
        conv_kernel_sizes=kernel_sizes,
        enable_deep_supervision=enable_deep_supervision,
        return_features=return_features,
        context=context,
        llm_repo=llm_repo,
        use_lora=use_lora,
        use_cls_head=use_cls_head,
        use_dicom=use_dicom,
        dicom_numeric_dim=dicom_numeric_dim,
        dicom_category_sizes=dicom_category_sizes,
    )


def get_stunet_large(
    num_input_channels,
    num_classes,
    strides=[[1, 1, 1], [1, 2, 2], [1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]],
    enable_deep_supervision=True,
    return_features=False,
    context=True,
    llm_repo: str = "meta-llama/Llama-2-7b-chat-hf",
    use_lora: bool = False,
    use_cls_head: bool = False,
    use_dicom: bool = False,
    dicom_numeric_dim: int = 10,
    dicom_category_sizes=(2, 2),
):
    kernel_sizes = [[3, 3, 3]] * 6
    if len(strides) > 5:
        strides = strides[:5]
    while len(strides) < 5:
        strides.append([1, 1, 1])

    return STUNet(
        num_input_channels,
        num_classes,
        depth=[2] * 6,
        dims=[64 * x for x in [1, 2, 4, 8, 16, 16]],
        pool_op_kernel_sizes=strides,
        conv_kernel_sizes=kernel_sizes,
        enable_deep_supervision=enable_deep_supervision,
        return_features=return_features,
        context=context,
        llm_repo=llm_repo,
        use_lora=use_lora,
        use_cls_head=use_cls_head,
        use_dicom=use_dicom,
        dicom_numeric_dim=dicom_numeric_dim,
        dicom_category_sizes=dicom_category_sizes,
    )
