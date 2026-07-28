# text_encoder.py
# -*- coding: utf-8 -*-
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange


# ---------------------------------------------------------------------
# Optional no-op helpers for backward compatibility with older imports
# ---------------------------------------------------------------------
def set_hf_tokenizer(_tok):
    """
    Backward-compat shim. Older code may import and call this.
    We no longer need a global tokenizer here, so this is a no-op.
    """
    return None


def tokenize(*args, **kwargs):
    """
    If some legacy path still tries to import `tokenize` from this module,
    make it explicit that tokenization is not handled here anymore.
    """
    raise NotImplementedError(
        "tokenize() is no longer provided in text_encoder.py. "
        "Use your HF tokenizer directly before calling the model."
    )

class TextContextEncoder(nn.Module):
    """
    Wrapper around a HuggingFace transformer used as a frozen text backbone.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.transformer = None
        self.token_embedding = None
        self.llm = False
        self.pad_id = kwargs.get("pad_id", 0)

    def forward(
        self,
        text: torch.LongTensor,                 # [B, Seq_Len]
        context: torch.Tensor | None = None,    # [1, Prompt_Len, C]
        attn_mask: torch.Tensor | None = None,  # [B, Seq_Len]
    ):
        if not self.llm or self.transformer is None or self.token_embedding is None:
            raise ValueError("TextContextEncoder is not configured for LLM usage.")

        # 🌟 Qwen 등 최신 모델의 fp16 오버플로우(NaN)를 막기 위해, 강제 fp16 대신 텐서의 고유 dtype(bf16 등)을 따르도록 동적 할당
        model_dtype = context.dtype if context is not None else torch.bfloat16

        # 🌟 [OOM & Attention Dilution 완벽 해결] Dynamic Left-Padding Truncation
        # padding_side = "left" 이므로, 패딩 토큰들은 항상 시퀀스 맨 왼쪽에 뭉쳐 있습니다.
        # 배치 전체에서 실제 유효 단어가 시작되는 최초의 인덱스를 찾아 그 이상만 물리적으로 남겨둡니다.
        is_not_pad = (text != self.pad_id)
        non_pad_indices = is_not_pad.nonzero()
        if non_pad_indices.numel() > 0:
            min_start_idx = non_pad_indices[:, 1].min().item()
        else:
            min_start_idx = 0
            
        # 진짜 의미 있는 EMR 단어들과 우측 <SEG> 특수 토큰만 물리적으로 슬라이싱하여 시퀀스를 절반 이상으로 압축!
        text = text[:, min_start_idx:]
        if attn_mask is not None:
            attn_mask = attn_mask[:, min_start_idx:]

        # 1. 입력 텍스트 임베딩 -> [B, Seq_Len, C]
        x_text = self.token_embedding(text).to(model_dtype)
        
        # 🌟 누락되었던 Shape 추출 코드 추가!
        B, N1, C = x_text.shape

        # 2. Learnable Soft Prompt(context) 결합 및 마스킹 계산
        if context is not None:
            # context: [1, N2, C] -> Batch 사이즈에 맞게 확장
            ctx_exp = context.expand(B, -1, -1).to(model_dtype) # [B, N2, C]
            
            # Soft Prompt를 원래 텍스트 시퀀스 맨 앞에 이어 붙임
            x = torch.cat([ctx_exp, x_text], dim=1).contiguous()  # [B, N2 + N1, C]
            
            # 🌟 마스크 확장: 앞에 붙은 Soft Prompt(길이 N2)는 무조건 어텐션(1) 적용
            if attn_mask is not None:
                ctx_mask = torch.ones((B, context.shape[1]), dtype=attn_mask.dtype, device=attn_mask.device)
                full_attn_mask = torch.cat([ctx_mask, attn_mask], dim=1)
            else:
                full_attn_mask = None
        else:
            x = x_text
            full_attn_mask = attn_mask

        # 3. LLM 통과 (단 1번의 포워드 패스로 연산량 최소화)
        # attention_mask가 넘어가면 텍스트 모델(LLM) 자체도 PAD 토큰을 무시함
        # [BUGFIX] custom Gemma 모델에서 input_ids=None일 때 inputs_embeds의 차원을 잘못 유추하여 160GB OOM(Broadcasting bug)이
        # 발생하는 것을 막기 위해 명시적으로 position_ids를 생성해서 넘겨줍니다.
        B_val, seq_len, _ = x.shape
        position_ids = (
            torch.arange(0, seq_len, dtype=torch.long, device=x.device)
            .unsqueeze(0)
            .expand(B_val, -1)
            .contiguous()
        )
        
        # [CRITICAL BUGFIX] Gemma4 모델 내부의 get_per_layer_inputs 함수에서 llm_input_ids가 None일 때 
        # 차원 유추 실패로 160GB짜리 텐서를 생성해버리는(Broadcasting) 치명적 버그를 우회하는 Monkey Patch
        def apply_gemma4_patch(module):
            if hasattr(module, "get_per_layer_inputs") and callable(getattr(module, "get_per_layer_inputs")):
                orig_func = getattr(module, "get_per_layer_inputs")
                if not getattr(orig_func, "_is_patched", False):
                    def patched_get_per_layer_inputs(llm_input_ids, llm_inputs_embeds, *args, **kwargs):
                        if llm_input_ids is None and llm_inputs_embeds is not None:
                            dummy_ids = torch.zeros(llm_inputs_embeds.shape[:2], dtype=torch.long, device=llm_inputs_embeds.device)
                            return orig_func(dummy_ids, llm_inputs_embeds, *args, **kwargs)
                        return orig_func(llm_input_ids, llm_inputs_embeds, *args, **kwargs)
                    patched_get_per_layer_inputs._is_patched = True
                    module.get_per_layer_inputs = patched_get_per_layer_inputs
            for child in module.children():
                apply_gemma4_patch(child)
                
        apply_gemma4_patch(self.transformer)

        outputs = self.transformer(
            inputs_embeds=x, 
            attention_mask=full_attn_mask,
            position_ids=position_ids
        )
        
        # 4. 마지막 Hidden States 전체 시퀀스 반환 (Word-level Feature)
        hidden_states = outputs.last_hidden_state.contiguous()

        # SAM 단에서 똑같이 PAD를 무시할 수 있도록 확장된 전체 마스크(full_attn_mask)를 함께 반환합니다!
        return hidden_states, full_attn_mask
