"""Config type conversion utilities for the recipe/tool system.

These functions convert MettaGridConfig and other basic configs into the
specific config types needed by different tools.
"""

from typing import Sequence

from metta.mettagrid.mettagrid_config import MettaGridConfig
from metta.sim.simulation_config import SimulationConfig


def mettagrid_to_simulation(env: MettaGridConfig, name: str | None = None) -> SimulationConfig:
    """Convert a MettaGridConfig to a SimulationConfig for play/replay tools."""
    return SimulationConfig(
        name=name or "simulation",
        env=env,
        num_episodes=1,
        max_time_s=120,
    )


def mettagrid_to_simulations(env: MettaGridConfig, name: str | None = None) -> Sequence[SimulationConfig]:
    """Convert a MettaGridConfig to a list of SimulationConfigs for sim/evaluate tools."""
    return [mettagrid_to_simulation(env, name)]


def simulation_to_simulations(sim: SimulationConfig) -> Sequence[SimulationConfig]:
    """Wrap a single SimulationConfig in a list for sim/evaluate tools."""
    return [sim]


def mettagrid_to_curriculum(env: MettaGridConfig):
    """Convert a MettaGridConfig to a simple env-only CurriculumConfig."""
    import metta.cogworks.curriculum as cc

    return cc.env_curriculum(env)


def mettagrid_to_trainer(env: MettaGridConfig):
    """Convert a MettaGridConfig to a simple TrainerConfig for training."""
    from metta.rl.loss.loss_config import LossConfig
    from metta.rl.trainer_config import TrainerConfig

    curriculum = mettagrid_to_curriculum(env)
    return TrainerConfig(
        curriculum=curriculum,
        losses=LossConfig(),
        # Reasonable defaults for simple training
        total_timesteps=10_000_000,
        batch_size=32768,
        bptt_horizon=256,
    )


def curriculum_to_trainer(curriculum):
    """Convert a CurriculumConfig to a TrainerConfig with reasonable defaults."""
    from metta.rl.loss.loss_config import LossConfig
    from metta.rl.trainer_config import TrainerConfig

    return TrainerConfig(
        curriculum=curriculum,
        losses=LossConfig(),
        # Reasonable defaults
        total_timesteps=50_000_000,
        batch_size=32768,
        bptt_horizon=256,
    )
