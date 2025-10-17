"""Cortex auto stack: mixes Axon, mLSTM, and sLSTM blocks.

Each layer is one of:
- 'A' → Axon cell in a PreUp block
- 'M' → mLSTM cell in a PreUp block
- 'S' → sLSTM cell in a PostUp block

Default pattern repeats "AMS" to reach ``num_layers``.
"""

from __future__ import annotations

from cortex.config import (
    AxonConfig,
    CortexStackConfig,
    PostUpBlockConfig,
    PreUpBlockConfig,
    mLSTMCellConfig,
    sLSTMCellConfig,
)
from cortex.stacks.base import CortexStack


def build_cortex_auto_stack(
    *,
    d_hidden: int,
    num_layers: int = 3,
    block_pattern: str | None = None,
    axon_preup: PreUpBlockConfig | None = None,
    mlstm_preup: PreUpBlockConfig | None = None,
    slstm_postup: PostUpBlockConfig | None = None,
    use_axonlayers: bool = False,
    post_norm: bool = True,
) -> CortexStack:
    """Build a mixed stack of Axon/mLSTM/sLSTM blocks with config-driven components.

    - Axon and mLSTM are wrapped with PreUp blocks; sLSTM with PostUp.
    - Provide per-block configs (including their internal cell configs) via
      ``axon_preup``, ``mlstm_preup``, and ``slstm_postup``. Defaults are used
      when a config is not supplied.
    - ``block_pattern`` remains; it selects which block to use per layer using
      characters 'A' (Axon PreUp), 'M' (mLSTM PreUp), and 'S' (sLSTM PostUp).
    """

    # Resolve pattern over {A, M, S}
    if block_pattern is None:
        base = "AMS"
        block_pattern = (base * ((num_layers + len(base) - 1) // len(base)))[:num_layers]
    else:
        if len(block_pattern) != num_layers:
            raise AssertionError(f"Pattern length {len(block_pattern)} != num_layers {num_layers}")
        allowed = {"A", "M", "S"}
        if not set(block_pattern).issubset(allowed):
            raise AssertionError("Pattern must use only 'A', 'M', 'S'")

    # Establish defaults when block configs are not provided
    if axon_preup is None:
        # Use defaults from config classes; do not override here
        axon_preup = PreUpBlockConfig(cell=AxonConfig())
    if mlstm_preup is None:
        mlstm_preup = PreUpBlockConfig(cell=mLSTMCellConfig())
    if slstm_postup is None:
        slstm_postup = PostUpBlockConfig(cell=sLSTMCellConfig())

    # Helper to clone a pydantic model (preserve subclass types on nested models)
    def _clone(cfg):
        if hasattr(cfg, "model_copy"):
            return cfg.model_copy(deep=True)  # pydantic v2
        return cfg.copy(deep=True)  # pydantic v1

    blocks: list[PreUpBlockConfig | PostUpBlockConfig] = []
    for ch in block_pattern:
        if ch == "A":
            blk = _clone(axon_preup)
        elif ch == "M":
            blk = _clone(mlstm_preup)
            if use_axonlayers and isinstance(blk.cell, mLSTMCellConfig):
                dumped = blk.cell.model_dump()
                dumped["use_axon_layer"] = True
                dumped["use_axon_qkv"] = True
                blk.cell = mLSTMCellConfig(**dumped)
        else:  # 'S'
            blk = _clone(slstm_postup)
            if use_axonlayers and isinstance(blk.cell, sLSTMCellConfig):
                dumped = blk.cell.model_dump()
                dumped["use_axon_layer"] = True
                blk.cell = sLSTMCellConfig(**dumped)

        blocks.append(blk)

    cfg = CortexStackConfig(blocks=blocks, d_hidden=d_hidden, post_norm=post_norm)
    return CortexStack(cfg)


__all__ = ["build_cortex_auto_stack"]
