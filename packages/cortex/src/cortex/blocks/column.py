"""Column block with a global router over experts."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict

from cortex.blocks.base import BaseBlock
from cortex.blocks.registry import build_block, register_block
from cortex.cells import build_cell
from cortex.cells.base import MemoryCell
from cortex.config import BlockConfig, ColumnBlockConfig
from cortex.types import MaybeState, ResetMask, Tensor


@register_block(ColumnBlockConfig)
class ColumnBlock(BaseBlock):
    def __init__(self, config: ColumnBlockConfig, d_hidden: int, cell: MemoryCell | None = None) -> None:
        super().__init__(d_hidden=d_hidden, cell=self._make_placeholder_cell(d_hidden))

        self.config = config
        self.d_hidden = d_hidden

        self.experts = nn.ModuleList(self._build_experts(config.experts, d_hidden))

        d_key = int(config.router.d_key or d_hidden)
        self.d_key = d_key
        self.temperature = float(config.router.temperature)
        self.top_k = int(config.router.top_k) if config.router.top_k is not None else None
        self.use_sqrt_scale = bool(config.router.use_sqrt_scale)

        self.context = nn.Parameter(torch.zeros(d_hidden))
        self.keys = nn.Parameter(torch.zeros(len(self.experts), d_key))

        self.Wq = nn.Linear(d_hidden, d_key, bias=False)
        self.Wk = nn.Linear(d_key, d_key, bias=False)

        nn.init.uniform_(self.context, a=-1e-3, b=1e-3)
        nn.init.uniform_(self.keys, a=-1e-3, b=1e-3)
        nn.init.uniform_(self.Wq.weight, a=-config.router.init_scale_wq, b=config.router.init_scale_wq)
        nn.init.uniform_(self.Wk.weight, a=-config.router.init_scale_wk, b=config.router.init_scale_wk)

    @staticmethod
    def _make_placeholder_cell(hidden_size: int) -> MemoryCell:
        """Minimal placeholder MemoryCell for BaseBlock."""

        class _NoOpCell(MemoryCell):
            def init_state(self, batch: int, *, device: torch.device | str, dtype: torch.dtype) -> TensorDict:
                return TensorDict({}, batch_size=[batch], device=torch.device(device))

            def forward(self, x: Tensor, state: MaybeState, *, resets: Optional[ResetMask] = None):
                return x, state

            def reset_state(self, state: MaybeState, mask: ResetMask) -> MaybeState:
                return state

        return _NoOpCell(hidden_size)

    def _build_experts(self, expert_cfgs: list[BlockConfig], d_hidden: int) -> list[BaseBlock]:
        experts: list[BaseBlock] = []
        for cfg in expert_cfgs:
            if cfg.cell is None:
                block = build_block(config=cfg, d_hidden=d_hidden, cell=None)  # type: ignore[arg-type]
            else:
                hs = cfg.get_cell_hidden_size(d_hidden)
                dumped = cfg.cell.model_dump()
                dumped["hidden_size"] = hs
                cell_config = type(cfg.cell)(**dumped)
                cell = build_cell(cell_config)
                block = build_block(config=cfg, d_hidden=d_hidden, cell=cell)
            experts.append(block)
        return experts

    def init_state(self, batch: int, *, device: torch.device | str, dtype: torch.dtype) -> TensorDict:
        td = TensorDict({}, batch_size=[batch], device=torch.device(device))
        for i, expert in enumerate(self.experts):
            key = self._expert_state_key(i, expert)
            td[key] = expert.init_state(batch=batch, device=device, dtype=dtype)
        return td

    def reset_state(self, state: MaybeState, mask: ResetMask) -> MaybeState:
        if state is None or not isinstance(state, TensorDict):
            return state
        batch_size = state.batch_size[0] if state.batch_size else mask.shape[0]
        new_state = TensorDict({}, batch_size=[batch_size], device=state.device)
        for i, expert in enumerate(self.experts):
            key = self._expert_state_key(i, expert)
            cur = state.get(key)
            new_state[key] = expert.reset_state(cur, mask)
        return new_state

    def forward(
        self,
        x: Tensor,
        state: MaybeState,
        *,
        resets: Optional[ResetMask] = None,
    ) -> Tuple[Tensor, MaybeState]:
        gate = self._compute_gate(x)
        is_step = x.dim() == 2
        B = x.shape[0]
        expert_outs = []
        next_state = TensorDict({}, batch_size=[B], device=x.device)

        for i, expert in enumerate(self.experts):
            key = self._expert_state_key(i, expert)
            expert_state = state.get(key) if isinstance(state, TensorDict) else None
            y_i, s_i = expert(x, expert_state, resets=resets)
            expert_outs.append(y_i)
            next_state[key] = s_i if isinstance(s_i, TensorDict) else TensorDict({}, batch_size=[B])

        if is_step:
            Y = torch.stack(expert_outs, dim=0)
            mixed = torch.einsum("k,kbh->bh", gate, Y)
        else:
            Y = torch.stack(expert_outs, dim=0)
            mixed = torch.einsum("k,kbth->bth", gate, Y)
        y = x + mixed
        return y, next_state

    @torch.no_grad()
    def _topk_mask_(self, scores: Tensor) -> Tensor:
        if self.top_k is None or self.top_k >= scores.numel():
            return scores
        k = self.top_k
        topk_vals, topk_idx = torch.topk(scores, k)
        mask = torch.full_like(scores, float("-inf"))
        mask[topk_idx] = topk_vals
        return mask

    def _compute_gate(self, x: Tensor) -> Tensor:
        q = self.Wq(self.context)
        k_proj = self.Wk(self.keys)
        if self.use_sqrt_scale:
            scale = 1.0 / math.sqrt(self.d_key)
        else:
            scale = 1.0 / float(self.d_key)

        scores = torch.einsum("kd,d->k", k_proj, q)
        scores = scores * scale
        if self.top_k is not None:
            with torch.no_grad():
                scores = self._topk_mask_(scores)
        logits = scores / max(self.temperature, 1e-6)
        gate = torch.softmax(logits, dim=-1)
        return gate

    @staticmethod
    def _expert_state_key(i: int, expert: nn.Module) -> str:
        cls = expert.__class__.__name__
        return f"expert_{cls}_{i}"


__all__ = ["ColumnBlock"]
