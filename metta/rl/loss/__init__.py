"""Lazy-loading facade for loss components.

This avoids import-time circular dependencies between loss modules and the
loss package by deferring heavy submodule imports until the symbols are
actually needed.
"""

import importlib
from typing import TYPE_CHECKING, Any

# Type imports only happen during type checking, not at runtime
if TYPE_CHECKING:
    from metta.rl.loss.loss import Loss
    from metta.rl.loss.loss_config import LossConfig
    from metta.rl.loss.ppo import PPOConfig

_EXPORTS: dict[str, tuple[str, str | None]] = {
    "LossConfig": ("metta.rl.loss.loss_config", "LossConfig"),
    "Loss": ("metta.rl.loss.loss", "Loss"),
    "PPOConfig": ("metta.rl.loss.ppo", "PPOConfig"),
}

# Explicitly define __all__ to help type checkers
__all__ = ["LossConfig", "Loss", "PPOConfig"]


def __getattr__(name: str) -> Any:
    """Dynamically import loss submodules on attribute access."""
    if name not in _EXPORTS:
        raise AttributeError(f"module 'metta.rl.loss' has no attribute '{name}'")

    module_path, attr_name = _EXPORTS[name]
    module = importlib.import_module(module_path)
    attr = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = attr
    return attr


def __dir__() -> list[str]:
    return sorted(__all__)
