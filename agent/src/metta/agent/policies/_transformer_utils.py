"""Shared utilities for Metta transformer backbones."""

from __future__ import annotations

import contextlib
import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn


def sinusoidal_position_encoding(positions: torch.Tensor, d_model: int, *, scale: float = 1.0) -> torch.Tensor:
    """Generate sinusoidal positional encodings for arbitrary position tensors."""

    positions = positions.to(dtype=torch.float32)
    device = positions.device

    log_term = -math.log(10000.0) / d_model
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=torch.float32) * log_term)
    pe = torch.zeros(*positions.shape, d_model, device=device, dtype=torch.float32)
    expanded = positions.unsqueeze(-1)
    pe[..., 0::2] = torch.sin(expanded * div_term)
    pe[..., 1::2] = torch.cos(expanded * div_term)
    return pe * scale


def _record_function(name: str):
    profiler_mod = getattr(torch, "profiler", None)
    if profiler_mod is not None and hasattr(profiler_mod, "record_function"):
        return profiler_mod.record_function(name)
    return contextlib.nullcontext()


def _make_layer_norm(d_model: int, _: bool) -> nn.Module:
    return nn.LayerNorm(d_model)


def empty_memory(
    num_layers: int,
    batch_size: int,
    d_model: int,
    device: torch.device,
    dtype: torch.dtype,
) -> List[torch.Tensor]:
    """Return a list of empty memory tensors."""

    with _record_function("Transformer/empty_memory"):
        return [torch.zeros(0, batch_size, d_model, device=device, dtype=dtype) for _ in range(num_layers)]


def normalize_memory(
    memory_len: int,
    num_layers: int,
    memory: Optional[Sequence[torch.Tensor]],
    batch_size: int,
    d_model: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[List[torch.Tensor]]:
    """Normalize previously stored memory tensors to the expected shape."""

    with _record_function("Transformer/normalize_memory"):
        if memory_len <= 0:
            return None

        if memory is None or len(memory) != num_layers:
            return empty_memory(num_layers, batch_size, d_model, device, dtype)

        normalized: List[torch.Tensor] = []
        for tensor in memory:
            if tensor is None or tensor.numel() == 0:
                normalized.append(torch.zeros(0, batch_size, d_model, device=device, dtype=dtype))
                continue
            mem = tensor.to(device=device, dtype=dtype)
            if mem.size(1) != batch_size:
                mem = mem[:, :batch_size].contiguous()
            normalized.append(mem)
        return normalized


def update_memory_window(
    layer_outputs: Sequence[torch.Tensor],
    previous_memory: Optional[Sequence[torch.Tensor]],
    memory_len: int,
    ext_len: int = 0,
) -> Optional[List[torch.Tensor]]:
    """Return the updated memory window for each layer."""

    with _record_function("Transformer/update_memory_window"):
        if memory_len <= 0:
            return None

        if not layer_outputs:
            return [torch.zeros(0)] * 0

        device = layer_outputs[0].device
        dtype = layer_outputs[0].dtype
        batch_size = layer_outputs[0].size(1)
        d_model = layer_outputs[0].size(2)
        num_layers = len(layer_outputs)

        if previous_memory is None or len(previous_memory) != num_layers:
            previous_memory = empty_memory(num_layers, batch_size, d_model, device, dtype)

        with torch.no_grad():
            mlen = previous_memory[0].size(0) if previous_memory else 0
            qlen = layer_outputs[0].size(0)
            end_idx = mlen + max(0, qlen - ext_len)
            beg_idx = max(0, end_idx - memory_len)

            updated: List[torch.Tensor] = []
            for prev, output in zip(previous_memory, layer_outputs, strict=False):
                cat = torch.cat([prev, output], dim=0)
                updated.append(cat[beg_idx:end_idx].detach())
        return updated


__all__ = [
    "sinusoidal_position_encoding",
    "_record_function",
    "_make_layer_norm",
    "empty_memory",
    "normalize_memory",
    "update_memory_window",
]
