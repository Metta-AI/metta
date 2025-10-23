"""PyTorch reference implementation of Transformer-XL attention with segment-causal masking."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _rel_shift(x: torch.Tensor) -> torch.Tensor:
    """Relative shift operation for Transformer-XL attention."""
    B, H, T, L = x.shape
    zero_pad = x.new_zeros(B, H, T, 1)
    x_padded = torch.cat([zero_pad, x], dim=3)
    x_padded = x_padded.view(B, H, L + 1, T)
    return x_padded[:, :, 1:, :].view(B, H, T, L)


def txl_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    r: torch.Tensor,
    seg_full: torch.Tensor,
    u: torch.Tensor,
    v_bias: torch.Tensor,
    M: int,
    scale: float,
) -> torch.Tensor:
    """Compute TXL attention via PyTorch ops."""
    B, H, T, D = q.shape
    L = k.shape[2]

    q_u = q + u[None, :, None, :]
    ac = torch.einsum("bhtd,bhld->bhtl", q_u, k)

    q_v = q + v_bias[None, :, None, :]
    bd = torch.einsum("bhtd,hld->bhtl", q_v, r)
    bd = _rel_shift(bd)

    logits = (ac + bd) * scale

    key_idx = torch.arange(L, device=q.device)
    q_idx = torch.arange(T, device=q.device)
    causal = key_idx[None, :] <= (M + q_idx[:, None])

    seg_q = seg_full[:, M : M + T]
    seg_k = seg_full
    same_seg = seg_q[:, :, None] == seg_k[:, None, :]
    allow = causal[None, :, :] & same_seg
    # Expand to [B, 1, T, L] to broadcast across heads
    Bsz, Hsz = B, H
    allow4 = allow.unsqueeze(1).expand(Bsz, Hsz, T, L)
    logits = logits.masked_fill(~allow4, torch.finfo(logits.dtype).min)
    attn = F.softmax(logits.float(), dim=-1).to(q.dtype)
    ctx = torch.einsum("bhtl,bhld->bhtd", attn, v)
    return ctx


__all__ = ["txl_pytorch"]
