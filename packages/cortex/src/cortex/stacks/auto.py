"""Cortex auto stack: mixes Axon, XL, mLSTM, and sLSTM blocks.

Each layer is one of:
- 'A' → Axon cell with PostUp block by default; if the FIRST layer is 'A' and no
  explicit Axon block is supplied, it defaults to a PassThrough block.
- 'X' → XL cell in a PostUp block
- 'M' → mLSTM cell in a PreUp block
- 'S' → sLSTM cell in a PostUp block

Default pattern repeats "AXMS" to reach ``num_layers``.
"""

from __future__ import annotations

from cortex.config import (
    AxonConfig,
    BlockConfig,
    CortexStackConfig,
    PassThroughBlockConfig,
    PostUpBlockConfig,
    PreUpBlockConfig,
    XLCellConfig,
    mLSTMCellConfig,
    sLSTMCellConfig,
)
from cortex.stacks.base import CortexStack


def build_cortex_auto_config(
    *,
    d_hidden: int,
    num_layers: int = 4,
    block_pattern: str | None = None,
    # Axon generally defaults to a PostUp block. However, when the FIRST
    # layer is Axon and no explicit Axon block is provided, we default that
    # first Axon layer to a PassThrough block to preserve identity behavior
    # and avoid redundant projections at the input boundary.
    axon_postup: PostUpBlockConfig | PreUpBlockConfig | None = None,
    mlstm_preup: PreUpBlockConfig | None = None,
    xl_postup: PostUpBlockConfig | None = None,
    slstm_postup: PostUpBlockConfig | None = None,
    use_axonlayers: bool = False,
    post_norm: bool = True,
) -> CortexStackConfig:
    """Return a CortexStackConfig for a mixed Axon/XL/mLSTM/sLSTM stack.

    - Axon and mLSTM are wrapped with PreUp blocks; XL and sLSTM with PostUp.
    - Provide per-block configs (including their internal cell configs) via
      ``axon_postup``, ``mlstm_preup``, ``xl_postup`` and ``slstm_postup``. Defaults are used
      when a config is not supplied.
    - ``block_pattern`` selects which block to use per layer using
      characters 'A' (Axon PostUp), 'X' (XL PostUp), 'M' (mLSTM PreUp), and 'S' (sLSTM PostUp).
    """

    # Resolve pattern over {A, X, M, S}
    if block_pattern is None:
        base = "AXMS"
        block_pattern = (base * ((num_layers + len(base) - 1) // len(base)))[:num_layers]
    else:
        if len(block_pattern) != num_layers:
            raise AssertionError(f"Pattern length {len(block_pattern)} != num_layers {num_layers}")
        allowed = {"A", "X", "M", "S"}
        if not set(block_pattern).issubset(allowed):
            raise AssertionError("Pattern must use only 'A', 'X', 'M', 'S'")

    # Establish defaults when block configs are not provided
    if axon_postup is None:
        # Axon defaults to a PostUp block configuration
        axon_postup = PostUpBlockConfig(cell=AxonConfig())
    if mlstm_preup is None:
        mlstm_preup = PreUpBlockConfig(cell=mLSTMCellConfig())
    if slstm_postup is None:
        slstm_postup = PostUpBlockConfig(cell=sLSTMCellConfig())
    if xl_postup is None:
        xl_postup = PostUpBlockConfig(cell=XLCellConfig())

    # Helper to clone a pydantic model (preserve subclass types on nested models)
    def _clone(cfg):
        if hasattr(cfg, "model_copy"):
            return cfg.model_copy(deep=True)  # pydantic v2
        return cfg.copy(deep=True)  # pydantic v1

    blocks: list[BlockConfig] = []
    for i, ch in enumerate(block_pattern):
        if ch == "A":
            # Special-case: if this is the very first layer and the caller
            # did not supply a custom Axon block, default to PassThrough.
            if i == 0:
                blk = PassThroughBlockConfig(cell=AxonConfig())
            else:
                blk = _clone(axon_postup)
        elif ch == "X":
            blk = _clone(xl_postup)
            if use_axonlayers and isinstance(blk.cell, XLCellConfig):
                dumped = blk.cell.model_dump()
                dumped["use_axon_qkv"] = True
                blk.cell = XLCellConfig(**dumped)
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
    return cfg


def build_cortex_auto_stack(
    *,
    d_hidden: int,
    num_layers: int = 3,
    block_pattern: str | None = None,
    # Accept either PreUp or PostUp; default is PostUp when None.
    axon_postup: PostUpBlockConfig | PreUpBlockConfig | None = None,
    mlstm_preup: PreUpBlockConfig | None = None,
    slstm_postup: PostUpBlockConfig | None = None,
    use_axonlayers: bool = False,
    post_norm: bool = True,
) -> CortexStack:
    """Build a CortexStack using the configuration returned by build_cortex_auto_config."""
    cfg = build_cortex_auto_config(
        d_hidden=d_hidden,
        num_layers=num_layers,
        block_pattern=block_pattern,
        axon_postup=axon_postup,
        mlstm_preup=mlstm_preup,
        slstm_postup=slstm_postup,
        use_axonlayers=use_axonlayers,
        post_norm=post_norm,
    )
    return CortexStack(cfg)


__all__ = ["build_cortex_auto_config", "build_cortex_auto_stack"]
