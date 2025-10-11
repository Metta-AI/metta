"""Template Cortex stack builders for the synthetic evaluation harness.

This module centralizes a handful of small, readable "stack recipes" that are
useful for quick comparisons on synthetic tasks. They showcase how to compose
cells and blocks via the configuration layer, and how to expose higher‑level
architectures like xLSTM behind a simple callable.

Add new templates by:
1) Writing a `build_*` function that returns a `CortexStack` (see examples).
2) Registering it in `STACKS` with a `StackSpec` so it becomes available via
   the `--stack` CLI flag in `run.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from cortex.config import (
    CortexStackConfig,
    PostUpBlockConfig,
    PreUpBlockConfig,
    mLSTMCellConfig,
    sLSTMCellConfig,
)
from cortex.factory import build_cortex
from cortex.stacks import CortexStack
from cortex.stacks.xlstm import build_xlstm_stack


@dataclass
class StackSpec:
    name: str
    builder: Callable[[], CortexStack]
    d_hidden: int


def build_slstm_postup(*, d_hidden: int = 128, proj_factor: float = 1.5, num_heads: int = 4) -> CortexStack:
    """sLSTM cell in a PostUp block; cell size equals external hidden size."""
    cfg = CortexStackConfig(
        d_hidden=d_hidden,
        post_norm=True,
        blocks=[
            PostUpBlockConfig(
                proj_factor=proj_factor,
                # hidden_size may be None here; the stack builder sets it to d_hidden for PostUp
                cell=sLSTMCellConfig(hidden_size=None, num_heads=num_heads, conv1d_kernel_size=4, dropout=0.0),
            )
        ],
    )
    return build_cortex(cfg)


def build_mlstm_preup(*, d_hidden: int = 128, proj_factor: float = 2.0, num_heads: int = 4) -> CortexStack:
    """mLSTM cell in a PreUp block; cell runs at inner dim = proj_factor*d_hidden."""
    cfg = CortexStackConfig(
        d_hidden=d_hidden,
        post_norm=True,
        blocks=[
            PreUpBlockConfig(
                proj_factor=proj_factor,
                # hidden_size may be None here; the stack builder sets it to int(proj_factor * d_hidden) for PreUp
                cell=mLSTMCellConfig(hidden_size=None, num_heads=num_heads, chunk_size=256, conv1d_kernel_size=4),
            )
        ],
    )
    return build_cortex(cfg)


# Registry of available stacks for the evaluation harness
STACKS: Dict[str, StackSpec] = {
    # Single‑block templates
    "slstm_postup": StackSpec(name="slstm_postup", builder=lambda: build_slstm_postup(), d_hidden=128),
    "mlstm_preup": StackSpec(name="mlstm_preup", builder=lambda: build_mlstm_preup(), d_hidden=128),
    # Composite templates
    # xLSTM: alternates mLSTM (PreUp) and sLSTM (PostUp)
    "xlstm": StackSpec(name="xlstm", builder=lambda: build_xlstm_stack(d_hidden=128, num_blocks=3), d_hidden=128),
    # Small and deeper variants for quick sweeps
    "xlstm_tiny": StackSpec(
        name="xlstm_tiny", builder=lambda: build_xlstm_stack(d_hidden=128, num_blocks=2), d_hidden=128
    ),
    "xlstm_deep": StackSpec(
        name="xlstm_deep", builder=lambda: build_xlstm_stack(d_hidden=128, num_blocks=6), d_hidden=128
    ),
}


__all__ = [
    "StackSpec",
    "STACKS",
    "build_slstm_postup",
    "build_mlstm_preup",
]
