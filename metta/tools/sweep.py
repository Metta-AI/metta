"""SweepTool for hyperparameter optimization using Protein."""

import logging
from typing import Callable, Optional

from metta.app_backend.clients.stats_client import StatsClient
from metta.common.config.tool import Tool
from metta.common.wandb.wandb_context import WandbConfig
from metta.sweep.sweep import sweep as run_sweep
from metta.sweep.sweep_config import SweepConfig
from metta.tools.train import TrainTool
from metta.tools.utils.auto_config import auto_stats_server_uri, auto_wandb_config

logger = logging.getLogger(__name__)


class SweepTool(Tool):
    """Tool for running hyperparameter sweeps with Protein optimization."""

    # Core sweep configuration
    sweep: SweepConfig

    # Sweep identity
    sweep_name: str
    sweep_dir: Optional[str] = None

    # Factory for creating TrainTool instances
    # This is a callable that takes a run name and returns a configured TrainTool
    train_tool_factory: Callable[[str], TrainTool]

    # Infrastructure configuration
    wandb: WandbConfig = WandbConfig.Unconfigured()
    stats_server_uri: Optional[str] = auto_stats_server_uri()

    # Track consumed arguments from the function signature
    consumed_args: list[str] = ["sweep_name", "num_trials"]

    def model_post_init(self, __context):
        """Post-initialization setup."""
        # Set sweep_dir based on sweep name if not explicitly set
        if self.sweep_dir is None:
            self.sweep_dir = f"{self.system.data_dir}/sweeps/{self.sweep_name}"

        # Auto-configure wandb if not set
        if self.wandb == WandbConfig.Unconfigured():
            self.wandb = auto_wandb_config(self.sweep_name)

    def invoke(self, args: dict[str, str] | None = None, overrides: list[str] | None = None) -> int:
        """Execute the sweep."""
        # Handle mutable defaults
        if args is None:
            args = {}
        if overrides is None:
            overrides = []

        logger.info(f"Starting sweep '{self.sweep_name}'")

        # Create stats client if URI provided
        stats_client = None
        if self.stats_server_uri:
            stats_client = StatsClient.create(self.stats_server_uri)

        # Run the sweep
        run_sweep(
            sweep_name=self.sweep_name,
            protein_config=self.sweep.protein,
            train_tool_factory=self.train_tool_factory,
            wandb_cfg=self.wandb,
            num_trials=self.sweep.num_trials,
            stats_client=stats_client,
            evaluation_simulations=self.sweep.evaluation_simulations,
        )

        logger.info(f"Sweep '{self.sweep_name}' completed")
        return 0
