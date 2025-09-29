"""Utility wrapper providing two-buffer context for transformer modules."""

from __future__ import annotations

import contextlib
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

TwoBufferMemory = Dict[str, Optional[torch.Tensor]]


class TwoBufferTransformer(nn.Module):
    """Wraps a transformer module to provide a two-buffer sliding context.

    The wrapper maintains at most ``memory_len`` tokens from the previous buffer and
    prepends them to the current sequence before delegating to ``core``. At the end
    of the forward pass it stores the most recent tokens so the next call reuses
    them as contextual history. No per-layer KV caches are kept – the transformer is
    re-run on the concatenated context, which keeps the implementation simple and
    avoids specialised kernels.
    """

    def __init__(
        self,
        core: nn.Module,
        *,
        memory_len: int,
        token_dim: int,
        max_context: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.core = core
        self.memory_len = max(0, int(memory_len))
        self.token_dim = token_dim
        self.max_context = max_context

    def forward(
        self,
        inputs: torch.Tensor,
        memory: Optional[TwoBufferMemory] = None,
    ) -> Tuple[torch.Tensor, TwoBufferMemory]:
        with _record_function("TwoBufferTransformer/forward"):
            squeeze = False
            if inputs.dim() == 2:
                inputs = inputs.unsqueeze(0)
                squeeze = True
            if inputs.dim() != 3:
                raise ValueError(f"Expected (T, B, D) tensor, received {inputs.shape}.")

            seq_len, batch_size, hidden = inputs.shape
            if hidden != self.token_dim:
                raise ValueError(f"Expected token dim {self.token_dim}, received {hidden}.")

            with _record_function("TwoBufferTransformer/load_prev_tokens"):
                prev_tokens = None
                if memory:
                    prev_tokens = memory.get("prev_tokens")
                if prev_tokens is not None and prev_tokens.numel() == 0:
                    prev_tokens = None

            with _record_function("TwoBufferTransformer/build_context"):
                context_parts = []
                if prev_tokens is not None:
                    prev_tokens = prev_tokens.to(device=inputs.device, dtype=inputs.dtype)
                    if self.memory_len > 0 and prev_tokens.size(0) > self.memory_len:
                        prev_tokens = prev_tokens[-self.memory_len :]
                    context_parts.append(prev_tokens)
                context_parts.append(inputs)
                context = torch.cat(context_parts, dim=0) if len(context_parts) > 1 else inputs

                if self.max_context is not None and context.size(0) > self.max_context:
                    raise ValueError(f"Context length {context.size(0)} exceeds configured max {self.max_context}.")

            with _record_function("TwoBufferTransformer/core_forward"):
                core_out, _ = self.core(context, None)
                current_out = core_out[-seq_len:]

            next_memory: Optional[torch.Tensor] = None
            if self.memory_len > 0:
                with _record_function("TwoBufferTransformer/update_memory"):
                    context_detached = context.detach()
                    next_memory = context_detached[-self.memory_len :]

            result = current_out.squeeze(0) if squeeze else current_out
            return result, {"prev_tokens": next_memory}

    def initialize_memory(self, batch_size: int) -> TwoBufferMemory:
        if self.memory_len <= 0:
            return {"prev_tokens": None}
        return {"prev_tokens": torch.zeros(0, batch_size, self.token_dim)}


__all__ = ["TwoBufferTransformer", "TwoBufferMemory"]


def _record_function(name: str):
    profiler_mod = getattr(torch, "profiler", None)
    if profiler_mod is not None and hasattr(profiler_mod, "record_function"):
        return profiler_mod.record_function(name)
    return contextlib.nullcontext()
