"""SweepTool for hyperparameter optimization using Protein."""

import logging
from typing import Callable, Optional, Sequence

from pydantic import ConfigDict

from metta.app_backend.clients.stats_client import StatsClient
from metta.common.config.tool import Tool
from metta.common.wandb.wandb_context import WandbConfig
from metta.sim.simulation_config import SimulationConfig
from metta.sweep.sweep_config import SweepConfig
from metta.sweep.sweep import sweep
from metta.tools.sim import SimTool
from metta.tools.train import TrainTool
from metta.tools.utils.auto_config import auto_stats_server_uri, auto_wandb_config

logger = logging.getLogger(__name__)


class SweepTool(Tool):
    """Tool for running hyperparameter sweeps with Protein optimization."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Core sweep configuration
    sweep: SweepConfig

    # Sweep identity
    sweep_name: str
    sweep_dir: Optional[str] = None

    # Factory for creating TrainTool instances
    # This is a callable that takes a run name and returns a configured TrainTool
    train_tool_factory: Callable[[str], TrainTool]

    simulations: Sequence[SimulationConfig]

    # Infrastructure configuration
    wandb: WandbConfig = WandbConfig.Unconfigured()
    stats_server_uri: Optional[str] = auto_stats_server_uri()

    # Track consumed arguments from the function signature
    consumed_args: list[str] = ["sweep_name", "num_trials", "run"]

    def model_post_init(self, __context):
        """Post-initialization setup."""
        # Set sweep_dir based on sweep name if not explicitly set
        if self.sweep_dir is None and self.sweep_name:
            self.sweep_dir = f"{self.system.data_dir}/sweeps/{self.sweep_name}"

        # Auto-configure wandb if not set
        if self.wandb == WandbConfig.Unconfigured() and self.sweep_name:
            self.wandb = auto_wandb_config(self.sweep_name)

    def model_dump(self, **kwargs):
        """Override to exclude factory functions from serialization."""
        # Don't pass 'indent' to parent model_dump as it doesn't accept it
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop("indent", None)
        data = super().model_dump(**kwargs_copy)
        # Exclude factory functions as they can't be serialized
        data.pop("train_tool_factory", None)
        data.pop("eval_tool_factory", None)
        return data

    def model_dump_json(self, **kwargs):
        """Override to exclude factory functions from JSON serialization."""
        # Extract indent if present
        indent = kwargs.pop("indent", None)
        # Get the dict without factories
        data = self.model_dump(**kwargs)
        # Serialize to JSON
        import json

        return json.dumps(data, indent=indent)

    def invoke(self, args: dict[str, str] | None = None, overrides: list[str] | None = None) -> int:
        """Execute the sweep."""
        # Handle mutable defaults
        if args is None:
            args = {}
        if overrides is None:
            overrides = []

        # Handle runtime arguments
        if "sweep_name" in args:
            if self.sweep_name is None or self.sweep_name == "":
                self.sweep_name = args["sweep_name"]
            elif self.sweep_name != args["sweep_name"]:
                raise ValueError(
                    f"sweep_name conflict: configured as '{self.sweep_name}' but args has '{args['sweep_name']}'"
                )

        if "num_trials" in args:
            # Override num_trials from args if provided
            self.sweep.num_trials = int(args["num_trials"])

        # Ensure we have required fields
        if not self.sweep_name:
            raise ValueError("sweep_name is required")

        # Re-run post_init in case sweep_name was set from args
        if self.sweep_dir is None:
            self.sweep_dir = f"{self.system.data_dir}/sweeps/{self.sweep_name}"
        if self.wandb == WandbConfig.Unconfigured():
            self.wandb = auto_wandb_config(self.sweep_name)

        logger.info(f"Starting sweep '{self.sweep_name}' with {self.sweep.num_trials} trials")

        # Create stats client if URI provided
        stats_client = None
        if self.stats_server_uri:
            stats_client = StatsClient.create(self.stats_server_uri)

        # Create the SequentialSweepPipeline instance with wrapped factories
        sweep(
            sweep_name=self.sweep_name,
            protein_config=self.sweep.protein,
            train_tool_factory=self.train_tool_factory,
            wandb_cfg=self.wandb,
            evaluation_simulations=self.simulations,
            num_trials=self.sweep.num_trials,
            sweep_server_uri=self.stats_server_uri or "https://api.observatory.softmax-research.net",
            max_observations_to_load=self.sweep.max_observations_to_load,
            stats_client=stats_client,
        )

        # Initialize services once at the start
        logger.info(f"Sweep '{self.sweep_name}' completed")
        return 0
