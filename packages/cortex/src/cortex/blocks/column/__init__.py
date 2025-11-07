from __future__ import annotations

import importlib

__all__ = ["auto", "column", "routers", "tokens", "ColumnBlock"]

_SUBMODULES = {name: f"{__name__}.{name}" for name in ["auto", "column", "routers", "tokens"]}
_CLASS_EXPORTS = {
    "ColumnBlock": ("cortex.blocks.column.column", "ColumnBlock"),
}


def __getattr__(name: str):
    module_path = _SUBMODULES.get(name)
    if module_path is not None:
        module = importlib.import_module(module_path)
        globals()[name] = module
        return module

    export = _CLASS_EXPORTS.get(name)
    if export is not None:
        module_path, attr = export
        module = importlib.import_module(module_path)
        value = getattr(module, attr)
        globals()[name] = value
        return value

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
