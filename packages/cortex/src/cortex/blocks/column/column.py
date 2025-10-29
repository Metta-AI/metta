"""Column block with pluggable router modules."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch._dynamo import disable

from cortex.blocks.base import BaseBlock
from cortex.blocks.column.routers import GlobalContextDotRouter
from cortex.blocks.registry import build_block, register_block
from cortex.cells import build_cell
from cortex.cells.base import MemoryCell
from cortex.config import BlockConfig, ColumnBlockConfig
from cortex.types import MaybeState, ResetMask, Tensor


@register_block(ColumnBlockConfig)
class ColumnBlock(BaseBlock):
    """Mix multiple experts using a router that outputs a global gate."""

    def __init__(self, config: ColumnBlockConfig, d_hidden: int, cell: MemoryCell | None = None) -> None:
        super().__init__(d_hidden=d_hidden, cell=self._make_placeholder_cell(d_hidden))
        self.config = config
        self.d_hidden = d_hidden
        self.experts = nn.ModuleList(self._build_experts(config.experts, d_hidden))
        self.router = GlobalContextDotRouter(d_hidden, len(self.experts), config.router)
        # Precompute stable per‑expert state keys once.
        self._expert_keys: list[str] = [self._expert_state_key(i, expert) for i, expert in enumerate(self.experts)]
        self._compiled_experts: list | None = None

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
        # Build in one shot to avoid per‑item __setitem__ during graph capture.
        state_map = {
            key: expert.init_state(batch=batch, device=device, dtype=dtype)
            for key, expert in zip(self._expert_keys, self.experts, strict=False)
        }
        return TensorDict(state_map, batch_size=[batch], device=torch.device(device))

    def reset_state(self, state: MaybeState, mask: ResetMask) -> MaybeState:
        if state is None or not isinstance(state, TensorDict):
            return state
        batch_size = state.batch_size[0] if state.batch_size else mask.shape[0]
        # Build mapping in Python first, then wrap.
        state_map = {}
        for key, expert in zip(self._expert_keys, self.experts, strict=False):
            cur = state.get(key)
            state_map[key] = expert.reset_state(cur, mask)
        return TensorDict(state_map, batch_size=[batch_size], device=state.device)

    def forward(
        self,
        x: Tensor,
        state: MaybeState,
        *,
        resets: Optional[ResetMask] = None,
    ) -> Tuple[Tensor, MaybeState]:
        is_step = x.dim() == 2
        B = x.shape[0]
        expert_outs: list[Tensor] = []
        state_map: dict[str, TensorDict] = {}
        td_empty = _empty_td(B, x.device)
        use_compiled = self._compiled_experts is not None and torch.is_grad_enabled()
        expert_call_list = self._compiled_experts if use_compiled else list(self.experts)

        for key, expert in zip(self._expert_keys, expert_call_list, strict=False):
            expert_state = state.get(key) if isinstance(state, TensorDict) else td_empty
            if expert_state is None:
                expert_state = td_empty
            y_i, s_i = expert(x, expert_state, resets=resets)
            expert_outs.append(y_i)
            state_map[key] = s_i if isinstance(s_i, TensorDict) else _empty_td(B, x.device)
        next_state = _make_td(state_map, B, x.device)

        if len(expert_outs) == 1:
            return expert_outs[0], next_state

        gate = self.router(expert_outs)
        if is_step:
            Y = torch.stack(expert_outs, dim=0)  # [K, B, H]
            y = torch.einsum("k,kbh->bh", gate, Y)
        else:
            Y = torch.stack(expert_outs, dim=0)  # [K, B, T, H]
            y = torch.einsum("k,kbth->bth", gate, Y)
        return y, next_state

    @staticmethod
    def _expert_state_key(i: int, expert: nn.Module) -> str:
        cls = expert.__class__.__name__
        return f"expert_{cls}_{i}"


__all__ = ["ColumnBlock"]


@disable
def _make_td(state_map: dict[str, TensorDict], batch_size: int, device: torch.device | str) -> TensorDict:
    return TensorDict(state_map, batch_size=[batch_size], device=torch.device(device))


@disable
def _empty_td(batch_size: int, device: torch.device | str) -> TensorDict:
    return TensorDict({}, batch_size=[batch_size], device=torch.device(device))
