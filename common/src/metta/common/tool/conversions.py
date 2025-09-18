"""Config type conversion utilities for the recipe/tool system."""

from typing import Sequence

import metta.cogworks.curriculum as cc
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.mettagrid.mettagrid_config import MettaGridConfig
from metta.rl.trainer_config import TrainerConfig
from metta.sim.simulation_config import SimulationConfig


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
