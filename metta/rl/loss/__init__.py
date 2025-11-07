from __future__ import annotations

import importlib

__all__ = [
    "loss_config",
    "Loss",
]

_loss_module = importlib.import_module("metta.rl.loss.loss")

Loss = _loss_module.Loss


def __getattr__(name: str):
    try:
        module = importlib.import_module(f"{__name__}.{name}")
    except ModuleNotFoundError as exc:  # pragma: no cover - dynamic import
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from exc
    globals()[name] = module
    return module
