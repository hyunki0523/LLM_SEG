from __future__ import annotations
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn

from monai.networks.blocks import UnetrBasicBlock
from monai.utils import ensure_tuple_rep
from transformers import LlamaTokenizer

from .modules import ContextUnetrUpBlock, UnetOutUpBlock
from .sam import TwoWayTransformer
from .text_encoder_legacy import tokenize, TextContextEncoder
from .llama2.llama_custom import LlamaForCausalLM

class STUNetEncoder(nn.Module):
    def __init__(self, 
                 input_channels, 
                 depth = [1]*6, 
                 dims = [16 * x for x in [1, 2, 4, 8, 16, 16]],  
                 conv_kernel_sizes = [[3,3,3]] * 6, 
                 pool_op_kernel_sizes = [[1,1,1],[1,2,2],[1,2,2],[1,2,2],[2,2,2]]):
        
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
            *[BasicResBlock(dims[0], dims[0], conv_kernel_sizes[0], self.conv_pad_sizes[0]) for _ in range(depth[0]-1)]
        )
        self.conv_blocks_context.append(stage)

        for d in range(1, num_pool + 1):
            stage = nn.Sequential(
                BasicResBlock(dims[d-1], dims[d], conv_kernel_sizes[d], self.conv_pad_sizes[d],
                              stride=pool_op_kernel_sizes[d-1], use_1x1conv=True),
                *[BasicResBlock(dims[d], dims[d], conv_kernel_sizes[d], self.conv_pad_sizes[d]) for _ in range(depth[d]-1)]
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
    def __init__(self, input_channels, num_classes, depth=[1,1,1,1,1,1], dims=[32, 64, 128, 256, 512, 512],
                 pool_op_kernel_sizes=None, conv_kernel_sizes=None, enable_deep_supervision=True,
                 return_features=False, context=True, text_encoder='llama2', llama_rep='/mnt/nas206/forGPU/lhyunki/NeuroCAD/LLM_seg_jhk/model_custom/llama2/Llama-2-7b-chat-hf/'):
        super().__init__()
        self.conv_op = nn.Conv3d
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        self.final_nonlin = lambda x:x 
        self.decoder = Decoder()
        self.decoder.deep_supervision = enable_deep_supervision
        self.upscale_logits = False

        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_pad_sizes = []
        self.return_features = return_features
        self.context = context
        self.text_encoder = text_encoder
        self.llama_rep = llama_rep

        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])

        num_pool  = len(pool_op_kernel_sizes)
        
        assert num_pool == len(dims) - 1
        
        # encoder
        self.conv_blocks_context = nn.ModuleList()
        stage = nn.Sequential(BasicResBlock(input_channels, dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0], use_1x1conv=True), 
                              *[BasicResBlock(dims[0], dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0]) for _ in range(depth[0]-1)])
        self.conv_blocks_context.append(stage)
        for d in range(1, num_pool+1):
            stage = nn.Sequential(BasicResBlock(dims[d-1], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d], stride=self.pool_op_kernel_sizes[d-1], use_1x1conv=True),
                *[BasicResBlock(dims[d], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d]) for _ in range(depth[d]-1)])
            self.conv_blocks_context.append(stage)

        # upsample_layers
        self.upsample_layers = nn.ModuleList()
        for u in range(num_pool):
            upsample_layer = Upsample_Layer_nearest(dims[-1-u], dims[-2-u], pool_op_kernel_sizes[-1-u])
            self.upsample_layers.append(upsample_layer)

        # decoder
        self.conv_blocks_localization = nn.ModuleList()
        for u in range(num_pool):
            stage = nn.Sequential(BasicResBlock(dims[-2-u] * 2, dims[-2-u], self.conv_kernel_sizes[-2-u], self.conv_pad_sizes[-2-u], use_1x1conv=True),
                *[BasicResBlock(dims[-2-u], dims[-2-u], self.conv_kernel_sizes[-2-u], self.conv_pad_sizes[-2-u]) for _ in range(depth[-2-u]-1)])
            self.conv_blocks_localization.append(stage)
            
        # outputs    
        self.seg_outputs = nn.ModuleList()
        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(nn.Conv3d(dims[-2-ds], num_classes, kernel_size=1))

        self.upscale_logits_ops = []
        for usl in range(num_pool - 1):
            self.upscale_logits_ops.append(lambda x: x)

        # multiomdal text encoder
        if self.context:
            if self.text_encoder in ["llama2", "llama2_13b"]:
                self.txt_embed_dim = 4096 if self.text_encoder == "llama2" else 5120
            else:
                self.txt_embed_dim = 512  

            from .text_encoder_legacy import TextContextEncoder
            self.text_encoder = TextContextEncoder(embed_dim=self.txt_embed_dim)
            self.context_length = 0
            self.n_prompts = 0 
            self.contexts = None
            self.max_length = 77 

            for _, p in self.text_encoder.named_parameters():
                p.requires_grad_(False)

            if isinstance(self.text_encoder, TextContextEncoder):
                from transformers import LlamaTokenizer
                from .llama2.llama_custom import LlamaForCausalLM

                if self.llama_rep is None:
                    raise ValueError("args.llama_rep 경로를 지정하세요.")
                self.text_encoder.llm = True
                self.tokenizer = LlamaTokenizer.from_pretrained(self.llama_rep)
                self.max_length = 128

                self.text_encoder.transformer = LlamaForCausalLM.from_pretrained(
                    self.llama_rep,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                ).model.half()
                self.tokenizer._add_tokens(["<SEG>"], special_tokens=True)
                self.text_encoder.transformer.resize_token_embeddings(len(self.tokenizer) + 1)
                self.text_encoder.token_embedding = self.text_encoder.transformer.embed_tokens

                for _, p in self.text_encoder.transformer.named_parameters():
                    p.requires_grad_(False)

            feature_size_list = dims  # [C0, C1, C2, C3, C4, C5] 

            txt2vis, attntrans = [], []
            for i, C in enumerate(feature_size_list):
                txt2vis.append(nn.Linear(self.txt_embed_dim, C))
                attntrans.append(
                    TwoWayTransformer(
                        depth=2,
                        embedding_dim=C,
                        mlp_dim=C * 2, 
                        num_heads=8
                    )
                )
            self.txt2vis = nn.ModuleList(txt2vis)
            self.attn_transformer = nn.ModuleList(attntrans)
        
    def forward(self, x, report_in=None):
        skips = []
        seg_outputs = []
        features = []

        # ----- 인코더 -----
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)  # [enc0, enc1, ..., enc_{N-2}]
        x = self.conv_blocks_context[-1](x)  # bottleneck = enc_{N-1}

        # 해상도 순서에 맞춘 리스트 (skips + bottleneck)
        hidden_states_out = skips + [x]  # len == len(dims)

        # ----- 텍스트-비전 정렬 (옵션) -----
        if self.context and report_in is not None:
            hidden_states_out, _, _ = self.interactive_alignment(hidden_states_out, report_in, x_in=skips[-1] if len(skips)>0 else x)

        # if self.context and report_in is not None:
        #     hidden_states_out, _, _ = self.interactive_alignment(hidden_states_out, report_in, x_in=skips[0] if len(skips)>0 else x)


        # 업데이트된 피처로 다시 분해
        skips = hidden_states_out[:-1]
        x = hidden_states_out[-1]

        # ----- 디코더 -----
        for u in range(len(self.conv_blocks_localization)):
            x = self.upsample_layers[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))
            if self.return_features:
                features.append(x)

        if self.decoder.deep_supervision:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        elif self.return_features:
            return features[-1]
        else:
            return seg_outputs[-1]
        
    def interactive_alignment(self, hidden_states_out, report_in, x_in):
        
        tok_txt = []
        emb_txt = []
        emb_txt_t = []
            
        # prepare text tokens
        tok_txt = report_in
        with torch.no_grad():
            emb_txt = self.text_encoder(tok_txt.to(x_in.device), self.contexts)
        
        # 모델의 나머지 데이터 타입과 완벽한 호환을 위해 명시적으로 캐스팅 (half <-> float 에러 방지)
        emb_txt = emb_txt.to(dtype=x_in.dtype)

        # projection
        report_l = []
        for i in self.txt2vis._modules.keys():
            report_l.append(self.txt2vis._modules[i](emb_txt))        

        # interactive alignment
        h_offset = 0
        for j, text_vis in enumerate(zip(report_l[h_offset:], hidden_states_out[h_offset:])):
            
            txt, vis = text_vis

            if len(report_in) != len(x_in):
                txt = torch.repeat_interleave(txt, vis.shape[0], dim=0)
            
            _, hidden_states_out[j+h_offset] = self.attn_transformer[j+h_offset](vis, None, txt)

        return hidden_states_out, emb_txt, emb_txt_t
    
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
                  
    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))  
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)

class Upsample_Layer_nearest(nn.Module):
    def __init__(self, input_channels, output_channels, pool_op_kernel_size, mode='nearest'):
        super().__init__()
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode
        
    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x

def get_stunet_small(num_input_channels, 
                     num_classes, 
                     strides = [[1,1,1],[1,2,2],[1,2,2],[1,2,2],[2,2,2],[2,2,2]], 
                     enable_deep_supervision=True,
                     return_features=False,
                     context=True):
        
        kernel_sizes = [[3,3,3]] * 6
        if len(strides)>5:
            strides = strides[:5]
        while len(strides)<5:
            strides.append([1,1,1])
            
        return STUNet(num_input_channels, num_classes, depth=[1]*6, dims= [16 * x for x in [1, 2, 4, 8, 16, 16]], 
                      pool_op_kernel_sizes=strides, conv_kernel_sizes=kernel_sizes, enable_deep_supervision=enable_deep_supervision,
                      return_features=return_features, context=context)

def get_stunet_base(num_input_channels, 
                     num_classes, 
                     strides = [[1,2,2],[1,2,2],[1,2,2],[1,2,2],[2,2,2],[2,2,2]], 
                     enable_deep_supervision=True,
                     return_features=False,
                     context=True,
                     llama_rep=None):
        
        kernel_sizes = [[3,3,3]] * 6
        if len(strides)>5:
            strides = strides[:5]
        while len(strides)<5:
            strides.append([1,1,1])

        return STUNet(num_input_channels, num_classes, depth=[1]*6, dims= [32 * x for x in [1, 2, 4, 8, 16, 16]], 
                      pool_op_kernel_sizes=strides, conv_kernel_sizes=kernel_sizes, enable_deep_supervision=enable_deep_supervision,
                      return_features=return_features, context=context, llama_rep=llama_rep)


def get_stunet_large(num_input_channels, 
                     num_classes, 
                     strides = [[1,1,1],[1,2,2],[1,2,2],[1,2,2],[2,2,2],[2,2,2]], 
                     enable_deep_supervision=True,
                     return_features=False,
                     context=True):
        
        kernel_sizes = [[3,3,3]] * 6
        if len(strides)>5:
            strides = strides[:5]
        while len(strides)<5:
            strides.append([1,1,1])

        return STUNet(num_input_channels, num_classes, depth=[2]*6, dims= [64 * x for x in [1, 2, 4, 8, 16, 16]], 
                      pool_op_kernel_sizes=strides, conv_kernel_sizes=kernel_sizes, enable_deep_supervision=enable_deep_supervision,
                      return_features=return_features, context=context)
