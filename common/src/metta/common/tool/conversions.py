"""Config type conversion utilities and helpers for the recipe/tool system."""

from typing import Callable, Literal, Sequence

import metta.cogworks.curriculum as cc
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.rl.trainer_config import TrainerConfig
from metta.sim.simulation_config import SimulationConfig
from mettagrid.config.mettagrid_config import MettaGridConfig


def mg_to_simulation(mg: MettaGridConfig) -> SimulationConfig:
    """Convert MettaGrid config to simulation config using the env's label."""
    return SimulationConfig(name=mg.label or "simulation", env=mg)


def mg_to_simulations(mg: MettaGridConfig) -> Sequence[SimulationConfig]:
    """Convert MettaGrid config to list of simulation configs."""
    return [mg_to_simulation(mg)]


def mg_to_curriculum(mg: MettaGridConfig) -> CurriculumConfig:
    """Convert MettaGrid config to curriculum config."""
    return cc.single_task_curriculum(mg)


def mg_to_trainer(mg: MettaGridConfig) -> TrainerConfig:
    """Convert MettaGrid config to trainer config with env-only curriculum."""
    return TrainerConfig(curriculum=mg_to_curriculum(mg))


def recipe_tool(
    tool: Literal["train", "sim", "play", "replay", "analyze"],
) -> Callable[[Callable[..., object]], Callable[..., object]]:
    """Decorator to mark a recipe function's default tool when no verb is provided.

    Example:
        @recipe_tool("play")
        def my_custom_env() -> MettaGridConfig: ...

    This lets `./tools/run.py my.module.my_custom_env` default to PlayTool,
    while `./tools/run.py sim my.module.my_custom_env policy_uri=...` still forces SimTool.
    """

    def _wrap(fn: Callable[..., object]) -> Callable[..., object]:
        setattr(fn, "_default_tool", tool)
        return fn

    return _wrap
