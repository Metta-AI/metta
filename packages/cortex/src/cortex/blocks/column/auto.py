"""Auto-build Column from a simple AXMS pattern with optional overrides."""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel

from cortex.blocks.column import ColumnBlock
from cortex.blocks.registry import build_block
from cortex.config import (
    AxonConfig,
    BlockConfig,
    ColumnBlockConfig,
    PostUpBlockConfig,
    PreUpBlockConfig,
    RouterConfig,
    XLCellConfig,
    mLSTMCellConfig,
    sLSTMCellConfig,
)


def _clone_model(model: BaseModel) -> BaseModel:
    if hasattr(model, "model_copy"):
        return model.model_copy(deep=True)  # pydantic v2
    return model.copy(deep=True)  # pydantic v1


def _builtin_for_token(token: str) -> BlockConfig | None:
    base = token.rstrip("^")
    ax = token.endswith("^")

    if base == "A":
        return PostUpBlockConfig(cell=AxonConfig())
    if base == "X":
        cell = XLCellConfig()
        if ax:
            dumped = cell.model_dump()
            dumped["use_axon_qkv"] = True
            cell = XLCellConfig(**dumped)
        return PostUpBlockConfig(cell=cell)
    if base == "M":
        cell = mLSTMCellConfig()
        if ax:
            dumped = cell.model_dump()
            dumped["use_axon_layer"] = True
            dumped["use_axon_qkv"] = True
            cell = mLSTMCellConfig(**dumped)
        return PreUpBlockConfig(cell=cell)
    if base == "S":
        cell = sLSTMCellConfig()
        if ax:
            dumped = cell.model_dump()
            dumped["use_axon_layer"] = True
            cell = sLSTMCellConfig(**dumped)
        return PostUpBlockConfig(cell=cell)
    return None


def _parse_tokens(pattern: str, custom_map: Dict[str, BlockConfig] | None) -> List[str]:
    s = pattern.replace(",", " ").strip()
    if not s:
        return []
    parts = s.split()
    if len(parts) > 1:
        return parts
    # Single concatenated run: scan for built-ins only
    tokens: List[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch not in ("A", "X", "M", "S"):
            raise ValueError(
                f"Unknown token at position {i}: '{ch}'. Use separators for custom symbols or allowed built-ins."
            )
        if i + 1 < len(s) and s[i + 1] == "^" and ch in ("M", "X", "S"):
            tokens.append(ch + "^")
            i += 2
        else:
            tokens.append(ch)
            i += 1
    return tokens


def build_column_auto_config(
    *,
    d_hidden: int,
    pattern: str | None = None,
    router: RouterConfig | None = None,
    custom_map: Dict[str, BlockConfig] | None = None,
) -> ColumnBlockConfig:
    pattern = pattern or "AXMS"
    tokens = _parse_tokens(pattern, custom_map)
    if not tokens:
        raise ValueError("Pattern produced no experts")

    experts: List[BlockConfig] = []
    for tok in tokens:
        cfg: BlockConfig | None = None
        if custom_map and tok in custom_map:
            cfg = custom_map[tok]
        else:
            # Consider base override with axonify toggle layered on
            base = tok.rstrip("^")
            ax = tok.endswith("^")
            if custom_map and base in custom_map:
                cfg = _clone_model(custom_map[base])  # type: ignore[assignment]
                cell = getattr(cfg, "cell", None)
                if ax and cell is not None:
                    if isinstance(cell, mLSTMCellConfig):
                        dumped = cell.model_dump()
                        dumped["use_axon_layer"] = True
                        dumped["use_axon_qkv"] = True
                        cfg.cell = mLSTMCellConfig(**dumped)
                    elif isinstance(cell, XLCellConfig):
                        dumped = cell.model_dump()
                        dumped["use_axon_qkv"] = True
                        cfg.cell = XLCellConfig(**dumped)
                    elif isinstance(cell, sLSTMCellConfig):
                        dumped = cell.model_dump()
                        dumped["use_axon_layer"] = True
                        cfg.cell = sLSTMCellConfig(**dumped)
            if cfg is None:
                cfg = _builtin_for_token(tok)

        if cfg is None:
            raise ValueError(f"Unknown token '{tok}'. Use A|X|M|S|M^|X^|S^ or provide a custom_map entry.")

        experts.append(_clone_model(cfg))  # new instance per expert

    col_cfg = ColumnBlockConfig(experts=experts, router=(router or RouterConfig()))
    return col_cfg


def build_column_auto_block(
    *,
    d_hidden: int,
    pattern: str | None = None,
    router: RouterConfig | None = None,
    custom_map: Dict[str, BlockConfig] | None = None,
) -> ColumnBlock:
    cfg = build_column_auto_config(d_hidden=d_hidden, pattern=pattern, router=router, custom_map=custom_map)
    block = build_block(config=cfg, d_hidden=d_hidden, cell=None)  # type: ignore[arg-type]
    assert isinstance(block, ColumnBlock)
    return block


__all__ = ["build_column_auto_config", "build_column_auto_block"]
