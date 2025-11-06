"""Auto builder: stacks of Column layers built from AXMS patterns."""


import cortex.blocks.column.auto
import cortex.config
import cortex.stacks.base


def build_cortex_auto_config(
    *,
    d_hidden: int,
    num_layers: int = 2,
    pattern: str | list[str] | None = "AXMS",
    custom_map: dict[str, cortex.config.BlockConfig] | None = None,
    router: cortex.config.RouterConfig | None = None,
    post_norm: bool = True,
    compile_blocks: bool = True,
) -> cortex.config.CortexStackConfig:
    """Build a CortexStackConfig with Column layers from AXMS patterns."""

    if pattern is None:
        patterns: list[str] = ["AXMS"] * num_layers
    elif isinstance(pattern, str):
        patterns = [pattern] * num_layers
    else:
        if len(pattern) != num_layers:
            raise ValueError(f"pattern list length {len(pattern)} != num_layers {num_layers}")
        patterns = list(pattern)

    blocks: list[cortex.config.BlockConfig] = []
    for pat in patterns:
        col_cfg = cortex.blocks.column.auto.build_column_auto_config(
            d_hidden=d_hidden, pattern=pat, router=router, custom_map=custom_map
        )
        blocks.append(col_cfg)

    return cortex.config.CortexStackConfig(
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
    custom_map: dict[str, cortex.config.BlockConfig] | None = None,
    router: cortex.config.RouterConfig | None = None,
    post_norm: bool = True,
    compile_blocks: bool = True,
) -> cortex.stacks.base.CortexStack:
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
    return cortex.stacks.base.CortexStack(cfg)
