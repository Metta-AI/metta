"""Sequential stack with optional per-block torch.compile."""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from tensordict import TensorDict

from cortex.blocks import ColumnBlock, build_block
from cortex.blocks.base import BaseBlock
from cortex.cells import build_cell
from cortex.config import CortexStackConfig
from cortex.types import MaybeState, ResetMask, Tensor

logger = logging.getLogger(__name__)


class CortexStack(nn.Module):
    """Stack of blocks that preserves external hidden size."""

    def __init__(self, cfg: CortexStackConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList(self._build_blocks(cfg))
        self.norm = nn.LayerNorm(cfg.d_hidden) if cfg.post_norm else nn.Identity()
        self._compiled_blocks: list | None = None

        compile_requested = bool(getattr(cfg, "compile_blocks", False))
        if compile_requested and not torch.cuda.is_available():
            logger.warning("Disabling block compilation for CortexStack: running on CPU.")
            compile_requested = False

        if compile_requested and hasattr(torch, "compile"):
            compiled: list[nn.Module] = []
            for b in self.blocks:
                if isinstance(b, ColumnBlock):
                    b._compiled_experts = [torch.compile(e) for e in b.experts]  # type: ignore[attr-defined]
                    compiled.append(b)
                else:
                    compiled.append(torch.compile(b))
            self._compiled_blocks = compiled

    def _build_blocks(self, cfg: CortexStackConfig) -> list[BaseBlock]:
        blocks: list[BaseBlock] = []
        d_hidden = cfg.d_hidden

        for _idx, block_cfg in enumerate(cfg.blocks):
            # For adapter blocks, cell comes from base_block, so skip cell building
            if block_cfg.cell is None:
                # Build block without cell (adapters handle this internally)
                block = build_block(config=block_cfg, d_hidden=d_hidden, cell=None)
            else:
                # Get the appropriate hidden size for the cell
                cell_hidden_size = block_cfg.get_cell_hidden_size(d_hidden)

                # Build cell with the appropriate hidden size using generic builder.
                # We intentionally overwrite any provided hidden_size (including None)
                # so that blocks can infer and enforce their dimensionality.
                dumped = block_cfg.cell.model_dump()
                dumped["hidden_size"] = cell_hidden_size
                cell_config = type(block_cfg.cell)(**dumped)
                cell = build_cell(cell_config)

                # Use generic block builder
                block = build_block(config=block_cfg, d_hidden=d_hidden, cell=cell)

            blocks.append(block)

        return blocks

    def init_state(self, batch: int, *, device: torch.device | str, dtype: torch.dtype) -> TensorDict:
        state = TensorDict({}, batch_size=[batch], device=torch.device(device))
        for i, block in enumerate(self.blocks):
            block_key = f"{block.__class__.__name__}_{i}"
            state[block_key] = block.init_state(batch=batch, device=device, dtype=dtype)
        return state

    def forward(
        self,
        x: Tensor,
        state: MaybeState,
        *,
        resets: Optional[ResetMask] = None,
    ) -> tuple[Tensor, MaybeState]:
        # Always expect batch-first input: [B, T, H] or [B, H]
        y = x
        batch_size = x.shape[0]
        next_state = TensorDict({}, batch_size=[batch_size])
        for i, block in enumerate(self.blocks):
            block_key = f"{block.__class__.__name__}_{i}"
            if isinstance(state, TensorDict):
                block_state = state.get(block_key)
                if block_state is None:
                    block_state = TensorDict({}, batch_size=[batch_size], device=y.device)
            else:
                block_state = TensorDict({}, batch_size=[batch_size], device=y.device)
            if self._compiled_blocks is not None and torch.is_grad_enabled():
                call = self._compiled_blocks[i]
            else:
                call = block
            y, block_next_state = call(y, block_state, resets=resets)
            next_state[block_key] = (
                block_next_state
                if isinstance(block_next_state, TensorDict)
                else TensorDict({}, batch_size=[batch_size])
            )
        y = self.norm(y)
        return y, next_state

    def step(
        self,
        x: Tensor,
        state: MaybeState = None,
        **kwargs,
    ) -> tuple[Tensor, MaybeState]:
        """Single timestep forward pass."""
        return self.forward(x, state, **kwargs)

    def reset_state(self, state: MaybeState, mask: ResetMask) -> MaybeState:
        if state is None or not isinstance(state, TensorDict):
            return state
        batch_size = state.batch_size[0] if state.batch_size else mask.shape[0]
        new_state = TensorDict({}, batch_size=[batch_size], device=state.device)
        for i, block in enumerate(self.blocks):
            block_key = f"{block.__class__.__name__}_{i}"
            new_state[block_key] = block.reset_state(state.get(block_key), mask)
        return new_state


__all__ = ["CortexStack"]
