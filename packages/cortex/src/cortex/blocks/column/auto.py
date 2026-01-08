"""Auto-build Column from AXMS-like patterns; built-ins live in cortex.tokens."""

from __future__ import annotations

import re
from typing import Dict, List

from pydantic import BaseModel

# Ensure built-in tokens are registered via decorators.
import cortex.tokens  # noqa: F401
from cortex.blocks.column import ColumnBlock
from cortex.blocks.registry import build_block
from cortex.config import (
    BlockConfig,
    ColumnBlockConfig,
    RouterConfig,
    XLCellConfig,
    mLSTMCellConfig,
    sLSTMCellConfig,
)
from cortex.registry import block_config_for_token, get_registered_tokens


def _clone_model(model: BaseModel) -> BaseModel:
    if hasattr(model, "model_copy"):
        return model.model_copy(deep=True)  # pydantic v2
    return model.copy(deep=True)  # pydantic v1


def _builtin_for_token(token: str) -> BlockConfig | None:
    return block_config_for_token(token)


def _parse_tokens(pattern: str, custom_map: Dict[str, BlockConfig] | None) -> List[str]:
    s = pattern.replace(",", " ").strip()
    if not s:
        return []
    parts = s.split()
    if len(parts) > 1:
        return parts
    # Single concatenated run: greedy regex over registered tokens only
    registered = sorted(get_registered_tokens(), key=len, reverse=True)
    if not registered:
        return []
    pattern_re = re.compile("|".join(re.escape(tok) for tok in registered))
    tokens: List[str] = []
    pos = 0
    n = len(s)
    while pos < n:
        m = pattern_re.match(s, pos)
        if not m:
            raise ValueError(f"Unknown token at position {pos}: '{s[pos]}'")
        tokens.append(m.group(0))
        pos = m.end()
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
            raise ValueError(
                f"Unknown token '{tok}'. Use A|C|L|M|S|X (with ^ for M/X/S) or provide a custom_map entry."
            )

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
