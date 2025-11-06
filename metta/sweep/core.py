"""Sweep parameter helpers for Ray-based sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ray import tune


@dataclass(frozen=True)
class ParameterSpec:
    """A single sweep parameter: dotted path + Ray Tune domain."""
    path: str
    space: Any


class SweepParameters:
    """Common parameter presets for sweeps."""

    # Optimizer parameters
    LEARNING_RATE = ParameterSpec(
        path="trainer.optimizer.learning_rate",
        space=tune.loguniform(1e-5, 1e-2),
    )

    ADAM_EPS = ParameterSpec(
        path="trainer.optimizer.eps",
        space=tune.loguniform(1e-8, 1e-4),
    )

    # PPO loss parameters
    PPO_CLIP_COEF = ParameterSpec(
        path="trainer.losses.loss_configs.ppo.clip_coef",
        space=tune.uniform(0.05, 0.3),
    )

    PPO_ENT_COEF = ParameterSpec(
        path="trainer.losses.loss_configs.ppo.ent_coef",
        space=tune.loguniform(1e-4, 3e-2),
    )

    PPO_GAE_LAMBDA = ParameterSpec(
        path="trainer.losses.loss_configs.ppo.gae_lambda",
        space=tune.uniform(0.8, 0.99),
    )

    PPO_VF_COEF = ParameterSpec(
        path="trainer.losses.loss_configs.ppo.vf_coef",
        space=tune.uniform(0.1, 1.0),
    )


# Convenience alias for cleaner imports
SP = SweepParameters

__all__ = [
    "ParameterSpec",
    "SweepParameters",
    "SP",
]