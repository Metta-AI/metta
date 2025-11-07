from __future__ import annotations

import importlib

__all__ = [
    "loss_config",
    "Loss",
]

_loss_module = importlib.import_module("metta.rl.loss.loss")

Loss = _loss_module.Loss
