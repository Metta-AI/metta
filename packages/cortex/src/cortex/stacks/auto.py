"""Cortex auto stack builder mixing Axon, XL, mLSTM, and sLSTM blocks with configurable patterns."""

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
    axon_postup: PostUpBlockConfig | None = None,
    mlstm_preup: PreUpBlockConfig | None = None,
    xl_postup: PostUpBlockConfig | None = None,
    slstm_postup: PostUpBlockConfig | None = None,
    use_axonlayers: bool = False,
    post_norm: bool = True,
) -> CortexStackConfig:
    """Build CortexStackConfig for mixed Axon/XL/mLSTM/sLSTM stack with customizable block configs and patterns."""

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
