"""Router modules for Column experts."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from cortex.config import RouterConfig
from cortex.types import Tensor


class BaseRouter(nn.Module):
    """Abstract router that returns a global expert gate via forward()."""

    def forward(self, expert_outputs: list[Tensor]) -> Tensor:  # shape: [K]
        raise NotImplementedError


class GlobalContextDotRouter(BaseRouter):
    """Global context · expert keys with softmax gating."""

    def __init__(self, d_hidden: int, num_experts: int, cfg: RouterConfig) -> None:
        super().__init__()
        d_key = int(cfg.d_key or d_hidden)
        self.d_key = d_key
        self.temperature = float(cfg.temperature)
        self.top_k = int(cfg.top_k) if cfg.top_k is not None else None
        self.use_sqrt_scale = bool(cfg.use_sqrt_scale)

        self.context = nn.Parameter(torch.zeros(d_hidden))
        self.keys = nn.Parameter(torch.zeros(num_experts, d_key))
        self.Wq = nn.Linear(d_hidden, d_key, bias=False)
        self.Wk = nn.Linear(d_key, d_key, bias=False)

        nn.init.uniform_(self.context, a=-1e-3, b=1e-3)
        nn.init.uniform_(self.keys, a=-1e-3, b=1e-3)
        nn.init.uniform_(self.Wq.weight, a=-cfg.init_scale_wq, b=cfg.init_scale_wq)
        nn.init.uniform_(self.Wk.weight, a=-cfg.init_scale_wk, b=cfg.init_scale_wk)

    @torch.no_grad()
    def _topk_mask(self, scores: Tensor) -> Tensor:
        if self.top_k is None or self.top_k >= scores.numel():
            return scores
        k = self.top_k
        topk_vals, topk_idx = torch.topk(scores, k)
        masked = torch.full_like(scores, float("-inf"))
        masked[topk_idx] = topk_vals
        return masked

    def forward(self, expert_outputs: list[Tensor]) -> Tensor:
        q = self.Wq(self.context)
        k_proj = self.Wk(self.keys)
        scale = 1.0 / math.sqrt(self.d_key) if self.use_sqrt_scale else 1.0 / float(self.d_key)
        scores = torch.einsum("kd,d->k", k_proj, q) * scale
        if self.top_k is not None:
            with torch.no_grad():
                scores = self._topk_mask(scores)
        logits = scores / max(self.temperature, 1e-6)
        gate = torch.softmax(logits, dim=-1)
        return gate


__all__ = ["BaseRouter", "GlobalContextDotRouter"]
