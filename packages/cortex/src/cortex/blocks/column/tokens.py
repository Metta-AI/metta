"""Built‑in Column expert symbols and helpers."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List

from cortex.config import (
    AxonConfig,
    BlockConfig,
    PostUpBlockConfig,
    PreUpBlockConfig,
    XLCellConfig,
    mLSTMCellConfig,
    sLSTMCellConfig,
)

# A builder returns a configured BlockConfig for a symbol; the boolean is whether
# the symbol carries a "^" suffix (request to enable Axon-backed projections if supported).
BuilderFn = Callable[[bool], BlockConfig]


def _build_A(ax: bool) -> BlockConfig:
    # Axon (post-up) expert; caret is irrelevant here.
    return PostUpBlockConfig(cell=AxonConfig())


def _build_X(ax: bool) -> BlockConfig:
    cell = XLCellConfig()
    if ax:
        dumped = cell.model_dump()
        dumped["use_axon_qkv"] = True
        cell = XLCellConfig(**dumped)
    return PostUpBlockConfig(cell=cell)


def _build_M(ax: bool) -> BlockConfig:
    cell = mLSTMCellConfig()
    if ax:
        dumped = cell.model_dump()
        dumped["use_axon_layer"] = True
        dumped["use_axon_qkv"] = True
        cell = mLSTMCellConfig(**dumped)
    return PreUpBlockConfig(cell=cell)


def _build_S(ax: bool) -> BlockConfig:
    cell = sLSTMCellConfig()
    if ax:
        dumped = cell.model_dump()
        dumped["use_axon_layer"] = True
        cell = sLSTMCellConfig(**dumped)
    return PostUpBlockConfig(cell=cell)


# Registry of built‑in symbols.
BUILTIN_TOKENS: Dict[str, BuilderFn] = {
    "A": _build_A,
    "X": _build_X,
    "M": _build_M,
    "S": _build_S,
}

# Single‑character built‑ins allowed in concatenated patterns.
SINGLE_CHAR_BUILTINS: List[str] = [k for k in BUILTIN_TOKENS.keys() if len(k) == 1]

# Symbols for which the "^" suffix is supported.
CARET_ALLOWED_BASES = frozenset({"M", "X", "S"})


def builtin_block_for_token(token: str) -> BlockConfig | None:
    """Return a BlockConfig for a built‑in token, or None if unknown.

    The token may optionally end with "^" to request Axon-backed projections
    where applicable.
    """

    base = token.rstrip("^")
    ax = token.endswith("^") and (base in CARET_ALLOWED_BASES)
    builder = BUILTIN_TOKENS.get(base)
    return builder(ax) if builder else None


def get_single_char_builtin_symbols() -> Iterable[str]:
    return tuple(SINGLE_CHAR_BUILTINS)


def can_use_caret(base: str) -> bool:
    return base in CARET_ALLOWED_BASES


__all__ = [
    "BUILTIN_TOKENS",
    "builtin_block_for_token",
    "get_single_char_builtin_symbols",
    "can_use_caret",
    "CARET_ALLOWED_BASES",
]
