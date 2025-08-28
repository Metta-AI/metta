"""Configuration for sweep execution."""

from typing import Optional

from pydantic import Field

from metta.app_backend.clients.stats_client import StatsClient
from metta.common.config import Config
from metta.sweep.protein_config import ProteinConfig


class SweepConfig(Config):
    """Configuration for hyperparameter sweep execution using tAXIOM."""

    # Number of trials to run
    num_trials: int = Field(default=10, description="Number of sweep trials to execute")

    # Protein optimizer configuration
    protein: ProteinConfig = Field(description="Configuration for the Protein optimizer")

    # Service configuration
    # Loading configuration
    max_observations_to_load: int = Field(
        default=250, description="Maximum number of previous observations to load from WandB"
    )

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True  # Allow Callable and other complex types
