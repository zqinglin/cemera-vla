"""
adapter.py

ShallowWideTransformerAdapter: A non-invasive transformer adapter that preserves the
input feature shape (B, 256, 2176) while operating at a wider internal width.

Architecture (sequential):
- Input Projection: 2176 -> 2688
- Positional Encoding: sinusoidal encoding added to token embeddings
- Transformer Encoder: stack of nn.TransformerEncoderLayer (batch_first)
- Output Projection: 2688 -> 2176

This module is designed to be dropped in front of the language model, operating purely
on the vision token sequence and returning a tensor with the same outer shape as input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sin-cos positional encoding (batch_first)."""

    def __init__(self, d_model: int, max_len: int = 256) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D)
        s = x.size(1)
        return x + self.pe[:, : s, :].to(dtype=x.dtype, device=x.device)


@dataclass
class AdapterConfig:
    num_patches: int = 256
    token_dim: int = 2176
    adapter_width: int = 2688
    nhead: int = 21
    num_layers: int = 2
    dropout: float = 0.0


class ShallowWideTransformerAdapter(nn.Module):
    """Shallow-and-wide transformer adapter preserving input shape.

    Input  shape: (B, num_patches, token_dim)
    Output shape: (B, num_patches, token_dim)
    """

    def __init__(self, config: Optional[AdapterConfig] = None) -> None:
        super().__init__()
        self.config = config or AdapterConfig()

        assert (
            self.config.adapter_width % self.config.nhead == 0
        ), "adapter_width must be divisible by nhead"

        # Input/Output projections
        self.proj_in = nn.Linear(self.config.token_dim, self.config.adapter_width, bias=True)
        self.pos_enc = SinusoidalPositionalEncoding(self.config.adapter_width, max_len=self.config.num_patches)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.adapter_width,
            nhead=self.config.nhead,
            dim_feedforward=self.config.adapter_width * 4,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.config.num_layers,
            norm=nn.LayerNorm(self.config.adapter_width),
        )

        self.proj_out = nn.Linear(self.config.adapter_width, self.config.token_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run adapter forward pass.

        Args:
            x: (B, num_patches=256, token_dim=2176)
        Returns:
            Tensor with identical shape as input.
        """
        b, s, d = x.shape
        if s != self.config.num_patches or d != self.config.token_dim:
            raise ValueError(
                f"Expected input shape (*, {self.config.num_patches}, {self.config.token_dim}), got {tuple(x.shape)}"
            )

        y = self.proj_in(x)
        y = self.pos_enc(y)
        y = self.encoder(y)
        y = self.proj_out(y)
        return y


class FeatureDiscriminator(nn.Module):
    """Per-token discriminator over (B, N, D) sequences.

    Applies a small MLP to each token independently and outputs (B, N, 1) sigmoid scores.
    """

    def __init__(self, token_dim: int = 2176) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D) -> (B*N, D)
        b, n, d = x.shape
        x_flat = x.reshape(b * n, d)
        y = self.mlp(x_flat)  # (B*N, 1)
        return y.view(b, n, 1)

if __name__ == "__main__":
    torch.manual_seed(7)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Non-negotiable hyperparameters per spec
    cfg = AdapterConfig(
        num_patches=256,
        token_dim=2176,
        adapter_width=2688,
        nhead=21,
        num_layers=2,
        dropout=0.0,
    )

    adapter = ShallowWideTransformerAdapter(cfg).to(device)
    dummy = torch.randn(1, cfg.num_patches, cfg.token_dim, device=device)
    out = adapter(dummy)
    assert out.shape == dummy.shape, f"Output shape {tuple(out.shape)} != input shape {tuple(dummy.shape)}"
    print("Adapter OK, output shape:", tuple(out.shape))


