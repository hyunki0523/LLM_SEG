import torch
import torch.nn as nn
import math

class PositionalEncodingGenerator3D(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncodingGenerator3D, self).__init__()
        self.d_model = d_model
        
        # 3D PE를 위한 충분히 큰 PE 테이블 생성
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (B, C, D, H, W)
        """
        b, c, d, h, w = x.shape
        assert c == self.d_model, f"Input channel {c} does not match d_model {self.d_model}"

        pe_d = self.pe[:d, :]  # (D, C)
        pe_h = self.pe[:h, :]  # (H, C)
        pe_w = self.pe[:w, :]  # (W, C)

        # Build a 3D positional encoding grid
        # pe_d is broadcast across H and W
        # pe_h is broadcast across D and W
        # pe_w is broadcast across D and H
        pe_3d = pe_d.unsqueeze(1).unsqueeze(2) + pe_h.unsqueeze(0).unsqueeze(2) + pe_w.unsqueeze(0).unsqueeze(1) # (D, H, W, C)
        
        # Permute to (C, D, H, W) and unsqueeze for batch dimension
        pe_3d = pe_3d.permute(3, 0, 1, 2).unsqueeze(0) # (1, C, D, H, W)

        return pe_3d.to(x.device)
