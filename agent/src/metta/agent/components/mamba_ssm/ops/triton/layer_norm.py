"""PyTorch-based fallbacks for Triton layer norm kernels."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps)
        return normed * self.weight


def _ensure_dtype(x: torch.Tensor, dtype: Optional[torch.dtype]) -> torch.Tensor:
    return x if dtype is None else x.to(dtype)


def _layer_norm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    eps: float,
    residual: Optional[torch.Tensor] = None,
    x1: Optional[torch.Tensor] = None,
    weight1: Optional[torch.Tensor] = None,
    bias1: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    rowscale: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    residual_dtype: Optional[torch.dtype] = None,
    is_rms_norm: bool = False,
    return_dropout_mask: bool = False,
):
    if residual is not None:
        x = x + _ensure_dtype(residual, x.dtype)
    if x1 is not None:
        x = x + x1
    if rowscale is not None:
        x = x * rowscale[..., None]
    if dropout_p > 0:
        x = F.dropout(x, p=dropout_p)
    if is_rms_norm:
        out = RMSNorm(weight.numel(), eps)(x)
    else:
        out = F.layer_norm(x, x.shape[-1:], weight=weight, bias=bias, eps=eps)
    out = _ensure_dtype(out, out_dtype)
    return out, None, None, None, x, None, None, None


def layer_norm_fn(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    residual: Optional[torch.Tensor] = None,
    prenorm: bool = True,
    residual_in_fp32: bool = False,
    eps: float = 1e-5,
    is_rms_norm: bool = False,
    dropout_p: float = 0.0,
):
    if prenorm:
        out = hidden_states
        residual_out = hidden_states if residual is None else residual
    else:
        out = hidden_states + (residual if residual is not None else 0)
        residual_out = out
    out = F.layer_norm(out, out.shape[-1:], weight=weight, bias=bias, eps=eps)
    return out, residual_out


def rms_norm_fn(*args, **kwargs):
    return layer_norm_fn(*args, is_rms_norm=True, **kwargs)


__all__ = ["RMSNorm", "layer_norm_fn", "rms_norm_fn", "_layer_norm_fwd"]
