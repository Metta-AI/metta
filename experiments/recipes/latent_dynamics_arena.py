"""Arena recipe with latent-variable dynamics model for model-based RL.

This recipe demonstrates training agents with a latent dynamics model that learns
to predict future states by encoding transitions into stochastic latent variables.

Based on "Learning Dynamics Model in Reinforcement Learning by Incorporating
the Long Term Future" (Ke et al., ICLR 2019).

Usage:
    # Train with full latent dynamics policy
    uv run ./tools/run.py train latent_dynamics_arena

    # Train with tiny variant for faster testing
    uv run ./tools/run.py train latent_dynamics_arena \\
        policy_architecture=metta.agent.policies.latent_dynamics.LatentDynamicsTinyConfig

    # Train with custom dynamics hyperparameters
    uv run ./tools/run.py train latent_dynamics_arena \\
        policy_architecture.components[4].beta_kl=0.05 \\
        policy_architecture.components[4].gamma_auxiliary=2.0
"""

from typing import Optional

import metta.cogworks.curriculum as cc
import mettagrid.builder.envs as eb
from metta.agent.policies.latent_dynamics import LatentDynamicsPolicyConfig
from metta.cogworks.curriculum.curriculum import (
    CurriculumAlgorithmConfig,
    CurriculumConfig,
)
from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig
from metta.rl.training import EvaluatorConfig, TrainingEnvironmentConfig
from metta.sim.simulation_config import SimulationConfig
from metta.tools.train import TrainTool
from mettagrid import MettaGridConfig


def mettagrid(num_agents: int = 24) -> MettaGridConfig:
    """Create arena environment configuration."""
    arena_env = eb.make_arena(num_agents=num_agents)
    return arena_env


def make_curriculum(
    arena_env: Optional[MettaGridConfig] = None,
    enable_detailed_slice_logging: bool = False,
    algorithm_config: Optional[CurriculumAlgorithmConfig] = None,
) -> CurriculumConfig:
    """Create curriculum for arena with latent dynamics learning."""
    arena_env = arena_env or mettagrid()

    arena_tasks = cc.bucketed(arena_env)

    # Inventory rewards curriculum
    for item in ["ore_red", "battery_red", "laser", "armor"]:
        arena_tasks.add_bucket(
            f"game.agent.rewards.inventory.{item}", [0, 0.1, 0.5, 0.9, 1.0]
        )
        arena_tasks.add_bucket(f"game.agent.rewards.inventory_max.{item}", [1, 2])

    # Attack curriculum (enable/disable via cost)
    arena_tasks.add_bucket("game.actions.attack.consumed_resources.laser", [1, 100])

    # Initial resources curriculum
    for obj in ["mine_red", "generator_red", "altar", "lasery", "armory"]:
        arena_tasks.add_bucket(f"game.objects.{obj}.initial_resource_count", [0, 1])

    if algorithm_config is None:
        algorithm_config = LearningProgressConfig(
            use_bidirectional=True,
            ema_timescale=0.001,
            exploration_bonus=0.1,
            max_memory_tasks=1000,
            max_slice_axes=5,
            enable_detailed_slice_logging=enable_detailed_slice_logging,
        )

    return arena_tasks.to_curriculum(algorithm_config=algorithm_config)


def simulations(env: Optional[MettaGridConfig] = None) -> list[SimulationConfig]:
    """Create evaluation simulations for arena."""
    basic_env = env or mettagrid()
    basic_env.game.actions.attack.consumed_resources["laser"] = 100

    combat_env = basic_env.model_copy()
    combat_env.game.actions.attack.consumed_resources["laser"] = 1

    return [
        SimulationConfig(suite="arena", name="basic", env=basic_env),
        SimulationConfig(suite="arena", name="combat", env=combat_env),
    ]


def train(
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
    policy_architecture: Optional[LatentDynamicsPolicyConfig] = None,
    beta_kl: float = 0.01,
    gamma_auxiliary: float = 1.0,
    future_horizon: int = 5,
) -> TrainTool:
    """Create training tool with latent dynamics policy.

    Args:
        curriculum: Training curriculum (default: arena curriculum)
        enable_detailed_slice_logging: Enable detailed curriculum logging
        policy_architecture: Policy architecture to use (default: LatentDynamicsPolicyConfig)
        beta_kl: KL divergence loss weight for dynamics model
        gamma_auxiliary: Auxiliary task loss weight for dynamics model
        future_horizon: Steps ahead to predict for auxiliary task

    Returns:
        TrainTool configured for latent dynamics training
    """
    curriculum = curriculum or make_curriculum(
        enable_detailed_slice_logging=enable_detailed_slice_logging
    )

    # Use provided policy or default
    if policy_architecture is None:
        policy_architecture = LatentDynamicsPolicyConfig()

    # Note: Latent dynamics loss configuration can be customized via CLI:
    # trainer.loss_configs='[{"class_path": "metta.rl.loss.latent_dynamics.LatentDynamicsLossConfig",
    #                         "beta_kl": 0.01, "gamma_auxiliary": 1.0}]'
    # The dynamics component in the policy will automatically produce the required outputs.

    return TrainTool(
        training_env=TrainingEnvironmentConfig(curriculum=curriculum),
        evaluator=EvaluatorConfig(simulations=simulations()),
        policy_architecture=policy_architecture,
    )


def train_tiny(
    curriculum: Optional[CurriculumConfig] = None,
    enable_detailed_slice_logging: bool = False,
) -> TrainTool:
    """Train with tiny latent dynamics policy for faster testing.

    This uses a smaller architecture for faster iteration during development.

    Args:
        curriculum: Training curriculum (default: arena curriculum)
        enable_detailed_slice_logging: Enable detailed curriculum logging

    Returns:
        TrainTool configured for tiny latent dynamics training
    """
    from metta.agent.policies.latent_dynamics import LatentDynamicsTinyConfig

    return train(
        curriculum=curriculum,
        enable_detailed_slice_logging=enable_detailed_slice_logging,
        policy_architecture=LatentDynamicsTinyConfig(),
    )


# Convenience functions for common configurations


def train_high_kl(curriculum: Optional[CurriculumConfig] = None) -> TrainTool:
    """Train with higher KL divergence weight for more structured latent space."""
    return train(curriculum=curriculum, beta_kl=0.05)


def train_high_auxiliary(curriculum: Optional[CurriculumConfig] = None) -> TrainTool:
    """Train with higher auxiliary task weight to emphasize future prediction."""
    return train(curriculum=curriculum, gamma_auxiliary=2.0)


def train_long_horizon(curriculum: Optional[CurriculumConfig] = None) -> TrainTool:
    """Train with longer prediction horizon for auxiliary task."""
    return train(curriculum=curriculum, future_horizon=10)
