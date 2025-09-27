from __future__ import annotations

import torch
import torch.nn as nn


class Mamba2(nn.Module):
    """Minimal CPU-friendly stub for the Mamba2 module."""

    def __init__(self, d_model: int, **kwargs):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, hidden_states, inference_params=None, **kwargs):
        return self.proj(hidden_states)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return None
