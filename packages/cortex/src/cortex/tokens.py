"""Built-in token definitions registered via decorators."""

from __future__ import annotations

from cortex.config import (
    AGaLiTeCellConfig,
    AxonConfig,
    BlockConfig,
    CausalConv1dConfig,
    LSTMCellConfig,
    PassThroughBlockConfig,
    PostUpBlockConfig,
    PostUpGatedBlockConfig,
    PreUpBlockConfig,
    XLCellConfig,
    mLSTMCellConfig,
    sLSTMCellConfig,
)
from cortex.registry import register_token


@register_token("A")
def _build_A() -> BlockConfig:
    # Axon (post-up) expert; caret is irrelevant here.
    return PostUpBlockConfig(cell=AxonConfig())


@register_token("X")
def _build_X() -> BlockConfig:
    cell = XLCellConfig()
    return PostUpGatedBlockConfig(cell=cell)


@register_token("X^")
def _build_X_axon() -> BlockConfig:
    dumped = XLCellConfig().model_dump()
    dumped["use_axon_qkv"] = True
    return PostUpGatedBlockConfig(cell=XLCellConfig(**dumped))


@register_token("M")
def _build_M() -> BlockConfig:
    cell = mLSTMCellConfig()
    return PreUpBlockConfig(cell=cell)


@register_token("M^")
def _build_M_axon() -> BlockConfig:
    dumped = mLSTMCellConfig().model_dump()
    dumped["use_axon_layer"] = True
    dumped["use_axon_qkv"] = True
    return PreUpBlockConfig(cell=mLSTMCellConfig(**dumped))


@register_token("S")
def _build_S() -> BlockConfig:
    cell = sLSTMCellConfig()
    return PostUpBlockConfig(cell=cell)


@register_token("S^")
def _build_S_axon() -> BlockConfig:
    dumped = sLSTMCellConfig().model_dump()
    dumped["use_axon_layer"] = True
    return PostUpBlockConfig(cell=sLSTMCellConfig(**dumped))


@register_token("L")
def _build_L() -> BlockConfig:
    # Standard LSTM expert, passthrough block to keep projections external.
    return PassThroughBlockConfig(cell=LSTMCellConfig())


@register_token("C")
def _build_C() -> BlockConfig:
    # Causal convolution expert, passthrough block.
    return PassThroughBlockConfig(cell=CausalConv1dConfig())


__all__: list[str] = []


@register_token("Ag")
def _build_Ag() -> BlockConfig:
    # AGaLiTe attention expert; use gated post block for GTrXL-like gating.
    return PostUpGatedBlockConfig(cell=AGaLiTeCellConfig())
