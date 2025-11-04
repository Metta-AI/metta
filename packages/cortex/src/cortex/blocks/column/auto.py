"""Auto-build Column from AXMS-like patterns; built-ins live in tokens.py."""

from __future__ import annotations

from typing import Dict, List

from cortex.blocks.column import ColumnBlock
from cortex.blocks.column.tokens import (
    builtin_block_for_token,
    can_use_caret,
    get_single_char_builtin_symbols,
)
from cortex.blocks.registry import build_block
from cortex.config import BlockConfig, CellConfig, ColumnBlockConfig, RouterConfig


def _enable_axon_if_supported(cell: CellConfig) -> CellConfig:
    """Enable Axon-backed projections when the cell exposes the relevant flags."""

    dumped = cell.model_dump()
    updated = False
    for field in ("use_axon_layer", "use_axon_qkv"):
        if field in dumped:
            dumped[field] = True
            updated = True
    if not updated:
        return cell
    return type(cell)(**dumped)


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
    allowed = set(get_single_char_builtin_symbols())
    while i < len(s):
        ch = s[i]
        if ch not in allowed:
            raise ValueError(
                f"Unknown token at position {i}: '{ch}'. Use separators for custom symbols or allowed built-ins."
            )
        if i + 1 < len(s) and s[i + 1] == "^" and can_use_caret(ch):
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
                base_cfg = custom_map[base]
                cfg = base_cfg.model_copy(deep=True) if hasattr(base_cfg, "model_copy") else base_cfg.copy(deep=True)  # type: ignore[assignment]
                cell = getattr(cfg, "cell", None)
                if ax and cell is not None:
                    cfg.cell = _enable_axon_if_supported(cell)
            if cfg is None:
                cfg = builtin_block_for_token(tok)

        if cfg is None:
            raise ValueError(f"Unknown token '{tok}'. Use A|X|M|S|M^|X^|S^ or provide a custom_map entry.")

        clone = cfg.model_copy(deep=True) if hasattr(cfg, "model_copy") else cfg.copy(deep=True)
        experts.append(clone)  # new instance per expert

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
