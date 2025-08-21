"""Configuration for sweep execution."""

from typing import Sequence

from pydantic import Field

from metta.common.config import Config
from metta.sim.simulation_config import SimulationConfig
from metta.sweep.protein_config import ProteinConfig


class SweepConfig(Config):
    """Configuration for hyperparameter sweep execution."""

    # Number of trials to run
    num_trials: int = Field(default=10, description="Number of sweep trials to execute")

    # Protein optimizer configuration
    protein: ProteinConfig = Field(description="Configuration for the Protein optimizer")

    # Optional evaluation simulations
    evaluation_simulations: Sequence[SimulationConfig] = Field(
        default_factory=list, description="Simulations to run for evaluating each trial"
    )

    # Maximum observations to load from previous runs
    max_observations_to_load: int = Field(
        default=100, description="Maximum number of previous observations to load from WandB"
    )
