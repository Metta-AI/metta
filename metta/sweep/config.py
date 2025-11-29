"""Canonical configuration models for sweep orchestration and tools."""

from enum import StrEnum
from typing import Any, Dict, Optional

from pydantic import Field

from metta.common.util.constants import PROD_STATS_SERVER_URI
from metta.common.wandb.context import WandbConfig
from mettagrid.base_config import Config


class SweepOrchestratorConfig(Config):
    """Configuration for base sweep orchestration loop."""

    max_parallel: int = Field(default=4, ge=1, description="Maximum number of parallel trials")
    poll_interval: float = Field(default=10.0, gt=0, description="Seconds between state syncs")
    initial_wait: float = Field(default=5.0, ge=0, description="Initial delay before first sync")
    metric_key: str = Field(default="score", description="Metric key to optimize")
    cost_key: Optional[str] = Field(default=None, description="Optional cost metric key")
    skip_evaluation: bool = Field(default=False, description="If true, skip separate evaluation phase")
    stop_on_error: bool = Field(default=False, description="Stop the sweep on any error in the main loop")
    resume: bool = Field(default=False, description="Recover state from store on startup")


class DispatcherType(StrEnum):
    """Available dispatcher types for job execution."""

    LOCAL = "local"
    SKYPILOT = "skypilot"
    REMOTE_QUEUE = "remote_queue"


class SweepToolConfig(Config):
    """Top-level configuration for running a sweep via SweepTool."""

    # Identity and IO
    sweep_name: Optional[str] = None
    sweep_dir: Optional[str] = None
    sweep_server_uri: str | None = Field(default=PROD_STATS_SERVER_URI)

    # Core sweep configuration
    max_trials: int = Field(default=10, ge=1)
    batch_size: int = Field(default=4, ge=1)
    recipe_module: str = Field(default="experiments.recipes.arena")
    train_entrypoint: str = Field(default="train")
    eval_entrypoint: str = Field(default="evaluate")

    # Resource settings
    gpus: int = Field(default=1, ge=1)
    nodes: int = Field(default=1, ge=1)
    max_parallel: int = Field(default=6, ge=1, description="Maximum parallel trials")
    poll_interval: float = Field(default=60.0, gt=0)
    initial_wait: float = Field(default=5.0, ge=0)

    # Sweep behavior
    local_test: bool = False
    force_eval: bool = False

    # Overrides
    train_overrides: Dict[str, Any] = Field(default_factory=dict)
    eval_overrides: Dict[str, Any] = Field(default_factory=dict)

    # Infrastructure configuration
    wandb: WandbConfig = Field(default_factory=WandbConfig.Unconfigured)
    stats_server_uri: str | None = Field(default=None)
    dispatcher_type: DispatcherType = Field(default=DispatcherType.SKYPILOT)
    capture_output: bool = Field(default=True, description="Capture and stream subprocess output (local only)")
    db_url: Optional[str] = None
    cost_key: Optional[str] = None
