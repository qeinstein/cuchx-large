"""Modular PyTorch temporal models for multimodal wearable and skeleton streams.

Architectures:
1. TemporalConvNet (TCN with dilated 1D convolutions & residual blocks)
2. BiGRUNet (2-layer bidirectional GRU with temporal self-attention)
3. TemporalTransformerNet (Lightweight Transformer with positional encodings)
4. DualStreamTemporalNet (Modality-specific encoders with cross-attention fusion)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        return x + self.pe[:, : x.size(1), :]


class TemporalBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        out = self.drop1(self.act1(self.bn1(self.conv1(x))))
        out = self.drop2(self.act2(self.bn2(self.conv2(out))))
        return out + residual


class TemporalConvNet(nn.Module):
    """Multi-scale Dilated 1D TCN."""
    def __init__(self, in_dim: int, num_classes: int = 40, hidden_dim: int = 128, num_levels: int = 4, dropout: float = 0.2):
        super().__init__()
        layers = []
        curr_dim = in_dim
        for i in range(num_levels):
            dilation = 2**i
            layers.append(TemporalBlock1D(curr_dim, hidden_dim, kernel_size=3, dilation=dilation, dropout=dropout))
            curr_dim = hidden_dim
        self.tcn = nn.Sequential(*layers)
        self.frame_head = nn.Linear(hidden_dim, num_classes)
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, D)
        x_perm = x.permute(0, 2, 1) # (B, D, T)
        feat_map = self.tcn(x_perm) # (B, hidden_dim, T)
        feat_t = feat_map.permute(0, 2, 1) # (B, T, hidden_dim)

        frame_logits = self.frame_head(feat_t) # (B, T, num_classes)

        # Attentive temporal pooling for clip-level representation
        attn_weights = F.softmax(self.attn_pool(feat_t), dim=1) # (B, T, 1)
        clip_feat = torch.sum(feat_t * attn_weights, dim=1) # (B, hidden_dim)
        clip_logits = self.frame_head(clip_feat) # (B, num_classes)

        return frame_logits, clip_logits


class BiGRUNet(nn.Module):
    """Bidirectional GRU with temporal self-attention."""
    def __init__(self, in_dim: int, num_classes: int = 40, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.gru = nn.GRU(
            hidden_dim, hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.frame_head = nn.Linear(hidden_dim, num_classes)
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, D)
        proj = self.proj(x)
        feat_t, _ = self.gru(proj) # (B, T, hidden_dim)

        frame_logits = self.frame_head(feat_t)

        attn_weights = F.softmax(self.attn_pool(feat_t), dim=1)
        clip_feat = torch.sum(feat_t * attn_weights, dim=1)
        clip_logits = self.frame_head(clip_feat)

        return frame_logits, clip_logits


class TemporalTransformerNet(nn.Module):
    """Lightweight Temporal Transformer with sinusoidal positional encodings."""
    def __init__(self, in_dim: int, num_classes: int = 40, d_model: int = 128, nhead: int = 4, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model)
        )
        self.pos_encoder = PositionalEncoding(d_model, max_len=128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=dropout, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.frame_head = nn.Linear(d_model, num_classes)
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, D)
        h = self.proj(x)
        h = self.pos_encoder(h)
        feat_t = self.transformer(h) # (B, T, d_model)

        frame_logits = self.frame_head(feat_t)

        attn_weights = F.softmax(self.attn_pool(feat_t), dim=1)
        clip_feat = torch.sum(feat_t * attn_weights, dim=1)
        clip_logits = self.frame_head(clip_feat)

        return frame_logits, clip_logits


class DualStreamTemporalNet(nn.Module):
    """Modality-specific Skeleton + IMU encoders with cross-attention fusion."""
    def __init__(self, skel_dim: int = 166, imu_dim: int = 50, num_classes: int = 40, d_model: int = 96, dropout: float = 0.2):
        super().__init__()
        self.skel_encoder = nn.Sequential(
            TemporalBlock1D(skel_dim, d_model, kernel_size=3, dilation=1, dropout=dropout),
            TemporalBlock1D(d_model, d_model, kernel_size=3, dilation=2, dropout=dropout),
            TemporalBlock1D(d_model, d_model, kernel_size=3, dilation=4, dropout=dropout)
        )
        self.imu_encoder = nn.Sequential(
            TemporalBlock1D(imu_dim, d_model, kernel_size=3, dilation=1, dropout=dropout),
            TemporalBlock1D(d_model, d_model, kernel_size=3, dilation=2, dropout=dropout),
            TemporalBlock1D(d_model, d_model, kernel_size=3, dilation=4, dropout=dropout)
        )
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True, dropout=dropout)
        self.fuse_norm = nn.LayerNorm(d_model * 2)
        self.frame_head = nn.Linear(d_model * 2, num_classes)
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x_skel: torch.Tensor, x_imu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x_skel: (B, T, 166), x_imu: (B, T, 50)
        h_s = self.skel_encoder(x_skel.permute(0, 2, 1)).permute(0, 2, 1) # (B, T, d_model)
        h_i = self.imu_encoder(x_imu.permute(0, 2, 1)).permute(0, 2, 1)   # (B, T, d_model)

        # Cross attention: query=skel, key/value=imu
        attn_out, _ = self.cross_attn(h_s, h_i, h_i)
        fused = self.fuse_norm(torch.cat([h_s, attn_out], dim=-1)) # (B, T, 2*d_model)

        frame_logits = self.frame_head(fused)

        attn_weights = F.softmax(self.attn_pool(fused), dim=1)
        clip_feat = torch.sum(fused * attn_weights, dim=1)
        clip_logits = self.frame_head(clip_feat)

        return frame_logits, clip_logits
