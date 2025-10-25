"""Latent-variable dynamics model recipe for ABES (Arena Basic Easy Shaped).

Based on "Learning Dynamics Model in Reinforcement Learning by Incorporating
the Long Term Future" (Ke et al., ICLR 2019).

Usage:
    uv run ./tools/run.py experiments.recipes.abes.latent_dynamics.train \\
        run=my_latent_dynamics_run

    uv run ./tools/run.py experiments.recipes.abes.latent_dynamics.train \\
        run=my_tiny_run \\
        policy_architecture.class_path=metta.agent.policies.latent_dynamics.LatentDynamicsTinyConfig
"""

import logging
from typing import TYPE_CHECKING, Optional

from experiments.recipes.arena_basic_easy_shaped import (
    evaluate,
    evaluate_in_sweep,
    make_curriculum,
    mettagrid,
    play,
    replay,
    simulations,
    sweep_async_progressive,
    train as base_train,
)
from metta.agent.policy import PolicyArchitecture
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.rl.trainer_config import TorchProfilerConfig
from metta.tools.train import TrainTool

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    pass


DEFAULT_LEARNING_RATE = 8e-4
DEFAULT_BATCH_SIZE = 131_072
DEFAULT_MINIBATCH_SIZE = 4_096
DEFAULT_FORWARD_PASS_MINIBATCH_TARGET_SIZE = 1_024


def _apply_overrides(
    tool: TrainTool,
    *,
    learning_rate: float,
    batch_size: int,
    minibatch_size: int,
    forward_pass_minibatch_target_size: int,
) -> None:
    """Apply training hyperparameter overrides."""
    trainer = tool.trainer
    trainer.optimizer.learning_rate = learning_rate
    trainer.batch_size = batch_size
    trainer.minibatch_size = minibatch_size

    tool.training_env.forward_pass_minibatch_target_size = (
        forward_pass_minibatch_target_size
    )
    tool.torch_profiler = TorchProfilerConfig(interval_epochs=0)


def train(
    *,
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
    policy_architecture: PolicyArchitecture | None = None,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    minibatch_size: int = DEFAULT_MINIBATCH_SIZE,
    forward_pass_minibatch_target_size: int = DEFAULT_FORWARD_PASS_MINIBATCH_TARGET_SIZE,
) -> TrainTool:
    """Train with latent-variable dynamics model.

    Args:
        curriculum: Training curriculum (default: arena_basic_easy_shaped)
        enable_detailed_slice_logging: Enable detailed curriculum logging
        policy_architecture: Policy architecture to use (default: LatentDynamicsPolicyConfig)
        learning_rate: Optimizer learning rate
        batch_size: Training batch size
        minibatch_size: Minibatch size for gradient updates
        forward_pass_minibatch_target_size: Target size for forward pass minibatches

    Returns:
        TrainTool configured for latent dynamics training
    """
    from metta.agent.policies.latent_dynamics import LatentDynamicsPolicyConfig

    policy = policy_architecture or LatentDynamicsPolicyConfig()

    tool = base_train(
        curriculum=curriculum,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
    )

    # Override policy architecture
    tool.policy_architecture = policy

    _apply_overrides(
        tool,
        learning_rate=learning_rate,
        batch_size=batch_size,
        minibatch_size=minibatch_size,
        forward_pass_minibatch_target_size=forward_pass_minibatch_target_size,
    )

    return tool


__all__ = [
    "mettagrid",
    "make_curriculum",
    "simulations",
    "play",
    "replay",
    "evaluate",
    "evaluate_in_sweep",
    "sweep_async_progressive",
    "train",
]
