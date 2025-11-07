"""Configuration builders for Metta environments."""

from __future__ import annotations

import importlib

__all__ = ["building", "envs", "empty_assemblers"]

_SUBMODULES = {name: f"{__name__}.{name}" for name in __all__}


def __getattr__(name: str):
    target = _SUBMODULES.get(name)
    if target is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    module = importlib.import_module(target)
    globals()[name] = module
    return module
