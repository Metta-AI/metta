from __future__ import annotations

import math
from typing import Optional

import torch


def _fwht_(x_2d: torch.Tensor) -> torch.Tensor:
    """In-place FWHT along last dim for [N, H], H must be power of two.

    Returns x_2d transformed. Uses iterative butterfly (O(H log H)).
    """
    N, H = x_2d.shape
    assert H & (H - 1) == 0, "FWHT requires H to be a power of 2"
    h = 1
    while h < H:
        x_reshaped = x_2d.view(N, -1, h * 2)
        a = x_reshaped[:, :, :h]
        b = x_reshaped[:, :, h:]
        x_reshaped[:, :, :h] = a + b
        x_reshaped[:, :, h:] = a - b
        h *= 2
    return x_2d


def srht_pytorch(
    x_btd: torch.Tensor,  # [B,T,H]
    signs_h: torch.Tensor,  # [H], +/-1 floats
    perm_h: Optional[torch.Tensor],  # [H] int64 or None
    *,
    normalize: bool = True,
) -> torch.Tensor:
    B, T, H = x_btd.shape
    device = x_btd.device
    dtype = x_btd.dtype

    if perm_h is not None:
        x = x_btd.index_select(dim=-1, index=perm_h.to(device=device))
    else:
        x = x_btd

    x = x * signs_h.view(1, 1, H).to(device=device, dtype=dtype)

    X = x.reshape(-1, H).contiguous()
    _fwht_(X)
    if normalize:
        X.mul_(1.0 / math.sqrt(H))
    return X.view(B, T, H)


__all__ = ["srht_pytorch"]
