"""Sliding-window flash-attention style memory cell."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from cortex.cells.base import MemoryCell
from cortex.cells.registry import register_cell
from cortex.config import SlidingFlashAttentionConfig
from cortex.types import MaybeState, ResetMask, Tensor


def _ensure_reset_mask(resets: Optional[ResetMask], batch: int, device: torch.device) -> Optional[torch.Tensor]:
    if resets is None or resets.numel() == 0:
        return None
    if resets.dim() == 1:
        return resets.to(device=device)
    if resets.dim() == 2:
        return resets[:, 0].to(device=device)
    raise ValueError(f"Unsupported reset mask dimension: {resets.dim()}")


def _sliding_attention_mask(length: int, window_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    idx = torch.arange(length, device=device)
    rel = idx.view(-1, 1) - idx.view(1, -1)
    allowed = (rel >= 0) & (rel < window_size)
    mask = torch.full((length, length), torch.finfo(dtype).min, device=device, dtype=dtype)
    mask = mask.masked_fill(allowed, 0.0)
    return mask


@register_cell(SlidingFlashAttentionConfig)
class SlidingFlashAttentionCell(MemoryCell):
    """Multi-head attention cell limited to a sliding temporal window."""

    def __init__(self, cfg: SlidingFlashAttentionConfig) -> None:
        if cfg.hidden_size is None:
            raise ValueError("SlidingFlashAttentionConfig.hidden_size must be specified")
        super().__init__(hidden_size=cfg.hidden_size)
        if cfg.hidden_size % cfg.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.cfg = cfg
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.window_size = cfg.window_size
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.out_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def init_state(self, batch: int, *, device: torch.device | str, dtype: torch.dtype) -> TensorDict:
        max_cache = max(self.window_size - 1, 0)
        k_cache = torch.zeros(batch, self.num_heads, max_cache, self.head_dim, device=device, dtype=dtype)
        v_cache = torch.zeros_like(k_cache)
        cache_len = torch.zeros(batch, dtype=torch.long, device=device)
        return TensorDict(
            {
                "k_cache": k_cache,
                "v_cache": v_cache,
                "cache_len": cache_len,
            },
            batch_size=[batch],
        )

    def reset_state(self, state: MaybeState, mask: ResetMask) -> MaybeState:
        if state is None:
            return state
        reset_mask = _ensure_reset_mask(mask, state.batch_size[0], state.device)
        if reset_mask is None:
            return state
        state = state.clone()
        reset_mask = reset_mask.to(dtype=torch.bool)
        if reset_mask.dim() != 1:
            raise ValueError("reset mask must be 1D for state reset")
        state["k_cache"][reset_mask] = 0.0
        state["v_cache"][reset_mask] = 0.0
        state["cache_len"][reset_mask] = 0
        return state

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.q_proj(x), self.k_proj(x), self.v_proj(x)

    def _reshape_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 3:
            B, T, _ = tensor.shape
            return tensor.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        if tensor.dim() == 2:
            B, _ = tensor.shape
            return tensor.view(B, self.num_heads, 1, self.head_dim)
        raise ValueError(f"Unexpected tensor shape {tensor.shape}")

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, NH, Tq, _ = q.shape
        Tk = k.shape[2]
        q_flat = q.reshape(B * NH, Tq, self.head_dim)
        k_flat = k.reshape(B * NH, Tk, self.head_dim)
        v_flat = v.reshape(B * NH, Tk, self.head_dim)
        attn_mask = None
        if mask is not None:
            attn_mask = mask.reshape(B * NH, Tq, Tk)
        output = F.scaled_dot_product_attention(
            q_flat,
            k_flat,
            v_flat,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        return output.reshape(B, NH, Tq, self.head_dim)

    def _sequence_forward(
        self,
        x: torch.Tensor,
        state: TensorDict,
        resets: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, TensorDict]:
        del resets  # sequence mode handles resets via mask within caller if needed
        B, T, _ = x.shape
        q, k, v = self._project(x)
        q_h = self._reshape_heads(q)
        k_h = self._reshape_heads(k)
        v_h = self._reshape_heads(v)

        mask = _sliding_attention_mask(T, self.window_size, x.device, x.dtype).unsqueeze(0).unsqueeze(0)
        mask = mask.expand(B, self.num_heads, T, T)
        context = self._attention(q_h, k_h, v_h, mask)
        context = context.transpose(1, 2).reshape(B, T, -1)
        y = self.out_proj(self.dropout(context))

        next_state = state.clone()
        max_cache = self.window_size - 1
        if max_cache > 0:
            if T >= max_cache:
                k_cache = k_h[:, :, -max_cache:, :]
                v_cache = v_h[:, :, -max_cache:, :]
                cache_len = torch.full((B,), max_cache, dtype=torch.long, device=x.device)
            else:
                pad = max_cache - T
                k_pad = torch.zeros(B, self.num_heads, pad, self.head_dim, device=x.device, dtype=x.dtype)
                v_pad = torch.zeros_like(k_pad)
                k_cache = torch.cat([k_pad, k_h], dim=2)
                v_cache = torch.cat([v_pad, v_h], dim=2)
                cache_len = torch.full((B,), T, dtype=torch.long, device=x.device)
        else:
            k_cache = torch.zeros(B, self.num_heads, 0, self.head_dim, device=x.device, dtype=x.dtype)
            v_cache = torch.zeros_like(k_cache)
            cache_len = torch.zeros(B, dtype=torch.long, device=x.device)

        next_state["k_cache"] = k_cache
        next_state["v_cache"] = v_cache
        next_state["cache_len"] = cache_len
        return y, next_state

    def _step_forward(
        self,
        x: torch.Tensor,
        state: TensorDict,
        resets: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, TensorDict]:
        B, _ = x.shape
        q, k, v = self._project(x)
        q_h = self._reshape_heads(q)
        k_h = self._reshape_heads(k)
        v_h = self._reshape_heads(v)

        k_cache = state.get("k_cache")
        v_cache = state.get("v_cache")
        cache_len = state.get("cache_len")
        max_cache = self.window_size - 1

        if resets is not None:
            resets = resets.to(dtype=torch.bool)
            if resets.dim() != 1:
                raise ValueError("Reset mask for step mode must be 1D")
            if max_cache > 0:
                k_cache[resets] = 0.0
                v_cache[resets] = 0.0
            cache_len[resets] = 0

        if max_cache > 0:
            k_window = torch.cat([k_cache, k_h], dim=2)
            v_window = torch.cat([v_cache, v_h], dim=2)
            positions = torch.arange(max_cache, device=x.device)
            valid_prev = positions.view(1, 1, -1) >= (max_cache - cache_len).view(-1, 1, 1)
            valid_prev = valid_prev.unsqueeze(2)
            attn_mask = torch.zeros(B, 1, 1, self.window_size, device=x.device, dtype=x.dtype)
            attn_mask[:, :, :, :max_cache] = attn_mask[:, :, :, :max_cache].masked_fill(
                ~valid_prev, torch.finfo(x.dtype).min
            )
        else:
            k_window = k_h
            v_window = v_h
            attn_mask = None

        mask = None if attn_mask is None else attn_mask.expand(B, self.num_heads, 1, k_window.shape[2])
        context = self._attention(q_h, k_window, v_window, mask)
        context = context.squeeze(2).reshape(B, -1)
        y = self.out_proj(self.dropout(context))

        next_state = state.clone()
        if max_cache > 0:
            next_state["k_cache"] = k_window[:, :, -max_cache:, :]
            next_state["v_cache"] = v_window[:, :, -max_cache:, :]
            next_state["cache_len"] = torch.clamp(cache_len + 1, max=max_cache)
        else:
            next_state["cache_len"] = torch.zeros_like(cache_len)
        return y, next_state

    def forward(
        self,
        x: Tensor,
        state: MaybeState,
        *,
        resets: Optional[ResetMask] = None,
    ) -> Tuple[Tensor, MaybeState]:
        if state is None or not isinstance(state, TensorDict):
            state = self.init_state(batch=x.shape[0], device=x.device, dtype=x.dtype)
        reset_mask = _ensure_reset_mask(resets, x.shape[0], x.device)
        if x.dim() == 2:
            return self._step_forward(x, state, reset_mask)
        if x.dim() != 3:
            raise ValueError("SlidingFlashAttentionCell expects inputs of shape [B, H] or [B, T, H]")
        return self._sequence_forward(x, state, reset_mask)


__all__ = ["SlidingFlashAttentionCell"]
