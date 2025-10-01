"""Shared utilities for transformer-based policy components."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:  # pragma: no cover - optional dependency
    from apex.normalization.fused_layer_norm import FusedLayerNorm  # type: ignore
except ImportError:  # pragma: no cover
    FusedLayerNorm = None  # type: ignore[misc]


def make_layer_norm(d_model: int, use_fused: bool) -> nn.Module:
    """Return a LayerNorm module with optional Apex fused acceleration."""

    if use_fused and FusedLayerNorm is not None:
        return FusedLayerNorm(d_model)
    return nn.LayerNorm(d_model)


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal positional embedding compatible with arbitrary position shapes."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        inv_freq = torch.arange(0.0, d_model, 2.0) / float(d_model)
        inv_freq = torch.pow(10000.0, -inv_freq)
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, positions: torch.Tensor, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        inv_freq = self.inv_freq
        if inv_freq.device != positions.device:
            inv_freq = inv_freq.to(device=positions.device)

        sinusoid_inp = positions.to(inv_freq.dtype).unsqueeze(-1) * inv_freq
        sin, cos = sinusoid_inp.sin(), sinusoid_inp.cos()
        pos_emb = torch.cat([sin, cos], dim=-1)
        if dtype is not None and pos_emb.dtype != dtype:
            pos_emb = pos_emb.to(dtype=dtype)
        return pos_emb


class XLPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding identical to Transformer-XL."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.embedding = SinusoidalPositionEmbedding(d_model)

    def forward(self, positions: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        pos_emb = self.embedding(positions)
        if batch_size is not None:
            return pos_emb[:, None, :].expand(-1, batch_size, -1)
        return pos_emb[:, None, :]


def ensure_mask_on_device(mask: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    """Ensure attention masks live on the same device as the target tensor."""

    if mask.device != device:
        mask = mask.to(device=device)
    return mask


__all__ = [
    "FusedLayerNorm",
    "SinusoidalPositionEmbedding",
    "XLPositionalEmbedding",
    "ensure_mask_on_device",
    "make_layer_norm",
]
