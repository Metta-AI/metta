"""Auto builder: stacks of Column layers built from AXMS patterns."""

from __future__ import annotations

from typing import Iterable, List, cast

from pydantic import BaseModel

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
    override_global_configs: Iterable[BaseModel] | None = None,
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

    # Optionally apply global overrides by type (e.g., XLCellConfig(mem_len=64)).
    if override_global_configs:
        blocks = [cast(BlockConfig, _apply_overrides_model(b, override_global_configs)) for b in blocks]

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
    override_global_configs: Iterable[BaseModel] | None = None,
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
        override_global_configs=override_global_configs,
    )
    return CortexStack(cfg)


def _clone_model(model: BaseModel) -> BaseModel:
    if hasattr(model, "model_copy"):
        return model.model_copy(deep=True)  # pydantic v2
    return model.copy(deep=True)  # pydantic v1


def _merge_model(model: BaseModel, update: BaseModel) -> BaseModel:
    """Return a new model with explicitly set fields from update overriding model."""
    fields_set = getattr(update, "model_fields_set", None) or getattr(update, "__fields_set__", None)
    # Pydantic v2 path
    if hasattr(model, "model_copy") and hasattr(update, "model_dump"):
        if fields_set:
            dump_all = update.model_dump()
            upd = {k: dump_all[k] for k in fields_set if k in dump_all}
        else:
            upd = update.model_dump(exclude_unset=True)
        return model.model_copy(update=upd)  # type: ignore[attr-defined]
    # Pydantic v1 fallback (no try/except)
    upd_data = update.dict()
    if fields_set:
        upd = {k: upd_data[k] for k in fields_set if k in upd_data}
    else:
        upd = update.dict(exclude_unset=True)
    data = model.dict()
    data.update(upd)
    return type(model)(**data)


def _apply_overrides_model(model: BaseModel, overrides: Iterable[BaseModel]) -> BaseModel:
    """Recursively apply overrides by matching model types.

    Any submodel whose type matches one of the override models' types is
    reconstructed with the override's fields merged on top of the original.
    """
    # Direct match: merge and return
    for ov in overrides:
        if isinstance(model, type(ov)):
            return _merge_model(model, ov)

    # Otherwise, recurse into fields
    # Build a shallow clone to avoid mutating the input instance
    cloned = _clone_model(model)
    fields = getattr(cloned, "model_fields", None) or getattr(cloned, "__fields__", {})
    for name in fields.keys():
        value = getattr(cloned, name, None)
        new_value = _apply_overrides_value(value, overrides)
        if new_value is not value:
            setattr(cloned, name, new_value)
    return cloned


def _apply_overrides_value(value, overrides: Iterable[BaseModel]):
    if isinstance(value, BaseModel):
        return _apply_overrides_model(value, overrides)
    if isinstance(value, list):
        changed = False
        out: List = []
        for item in value:
            new_item = _apply_overrides_value(item, overrides)
            changed = changed or (new_item is not item)
            out.append(new_item)
        return out if changed else value
    return value
