"""Auto builder: stacks of Column layers built from AXMS patterns."""

from __future__ import annotations

from cortex.blocks.column.auto import build_column_auto_config
from cortex.config import BlockConfig, CortexStackConfig, RouterConfig
from cortex.stacks.base import CortexStack


def build_cortex_auto_config(
    *,
    d_hidden: int,
    num_layers: int = 2,
    pattern: str | list[str] | None = "AXMS",
    custom_map: dict[str, BlockConfig] | None = None,
    router: RouterConfig | None = None,
    post_norm: bool = True,
    compile_blocks: bool = True,
) -> CortexStackConfig:
    """Build a CortexStackConfig with Column layers from AXMS patterns."""

    if pattern is None:
        patterns: list[str] = ["AXMS"] * num_layers
    elif isinstance(pattern, str):
        patterns = [pattern] * num_layers
    else:
        if len(pattern) != num_layers:
            raise ValueError(f"pattern list length {len(pattern)} != num_layers {num_layers}")
        patterns = list(pattern)

    blocks: list[BlockConfig] = []
    for pat in patterns:
        col_cfg = build_column_auto_config(d_hidden=d_hidden, pattern=pat, router=router, custom_map=custom_map)
        blocks.append(col_cfg)

    return CortexStackConfig(
        blocks=blocks,
        d_hidden=d_hidden,
        post_norm=post_norm,
        compile_blocks=bool(compile_blocks),
    )


def build_cortex_auto_stack(
    *,
    d_hidden: int,
    num_layers: int = 4,
    pattern: str | list[str] | None = "AXMS",
    custom_map: dict[str, BlockConfig] | None = None,
    router: RouterConfig | None = None,
    post_norm: bool = True,
    compile_blocks: bool = True,
) -> CortexStack:
    """Build a Column-based CortexStack with per-layer patterns."""
    cfg = build_cortex_auto_config(
        d_hidden=d_hidden,
        num_layers=num_layers,
        pattern=pattern,
        custom_map=custom_map,
        router=router,
        post_norm=post_norm,
        compile_blocks=compile_blocks,
    )
    return CortexStack(cfg)
