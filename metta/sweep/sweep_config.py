"""Configuration for sweep execution."""

from typing import TYPE_CHECKING, Callable, Optional

from pydantic import Field

from metta.app_backend.clients.stats_client import StatsClient
from metta.common.config import Config
from metta.common.wandb.wandb_context import WandbConfig
from metta.sweep.protein_config import ProteinConfig
from metta.tools.train import TrainTool

if TYPE_CHECKING:
    from metta.tools.sim import SimTool


class SweepConfig(Config):
    """Configuration for hyperparameter sweep execution using tAXIOM."""

    # Sweep identification
    sweep_name: str = Field(description="Unique identifier for this sweep")

    # Number of trials to run
    num_trials: int = Field(default=10, description="Number of sweep trials to execute")

    # Protein optimizer configuration
    protein: ProteinConfig = Field(description="Configuration for the Protein optimizer")

    # WandB configuration
    wandb: WandbConfig = Field(description="Weights & Biases configuration for experiment tracking")

    # Training configuration
    train_tool_factory: Callable[[str], TrainTool] = Field(
        description="Factory function that creates TrainTool instances",
        exclude=True,  # Exclude from serialization
    )

    # Evaluation configuration
    eval_tool_factory: Callable[[str, TrainTool], "SimTool"] = Field(
        description="Factory function that creates SimTool instances for evaluation",
        exclude=True,  # Exclude from serialization
    )

    # Service configuration
    sweep_server_uri: str = Field(
        default="https://api.observatory.softmax-research.net", description="Cogweb server URI for sweep coordination"
    )

    # Loading configuration
    max_observations_to_load: int = Field(
        default=250, description="Maximum number of previous observations to load from WandB"
    )

    # Optional stats client
    stats_client: Optional[StatsClient] = Field(
        default=None,
        description="Optional stats client for remote monitoring",
        exclude=True,  # Exclude from serialization
    )

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True  # Allow Callable and other complex types
