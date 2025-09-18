"""Simplified tool for adaptive experiments.

This tool wires the adaptive controller to run different experiment schedulers
with minimal configuration. It follows the same conventions as other tools:
- Fields are serializable (Pydantic) with clear defaults and descriptions
- Relevant CLI args are consumed and applied to configuration
- A JSON config is written to a run directory for reproducibility
"""

import logging
import time
from enum import StrEnum
from typing import Any, Optional

from pydantic import Field

from metta.adaptive.adaptive_config import AdaptiveConfig
from metta.adaptive.adaptive_controller import AdaptiveController
from metta.adaptive.dispatcher import LocalDispatcher
from metta.adaptive.dispatcher.skypilot import SkypilotDispatcher
from metta.adaptive.protocols import ExperimentState
from metta.adaptive.schedulers.train_and_eval import TrainAndEvalConfig, TrainAndEvalScheduler
from metta.sweep.schedulers.batched_synced import (
    BatchedSyncedOptimizingScheduler,
    BatchedSyncedSchedulerConfig,
)
from metta.adaptive.stores import WandbStore
from metta.common.tool import Tool
from metta.common.util.log_config import init_logging
from metta.common.wandb.wandb_context import WandbConfig
from metta.tools.utils.auto_config import auto_stats_server_uri, auto_wandb_config

logger = logging.getLogger(__name__)


class SchedulerType(StrEnum):
    """Available scheduler types for adaptive experiments."""

    TRAIN_AND_EVAL = "train_and_eval"
    BATCHED_SYNCED = "batched_synced"


class DispatcherType(StrEnum):
    """Available dispatcher types for job execution."""

    LOCAL = "local"  # All jobs run locally
    SKYPILOT = "skypilot"  # All jobs run on Skypilot


class AdaptiveTool(Tool):
    """Simple tool for running adaptive experiments."""

    # Core components
    scheduler_type: SchedulerType = Field(description="Adaptive scheduler type (e.g., train_and_eval, batched_synced)")
    scheduler_config: Any = Field(
        default_factory=dict,
        description="Typed scheduler config object (e.g., TrainAndEvalConfig or BatchedSyncedSchedulerConfig)",
    )
    config: AdaptiveConfig = Field(default_factory=AdaptiveConfig, description="Adaptive controller configuration")

    # Infrastructure configuration
    dispatcher_type: DispatcherType = Field(
        default=DispatcherType.SKYPILOT, description="Dispatcher backend (local or skypilot)"
    )
    capture_output: bool = Field(default=True, description="Capture and stream subprocess output (local only)")

    # Standard tool configuration
    wandb: WandbConfig = Field(default_factory=auto_wandb_config, description="W&B configuration")
    stats_server_uri: Optional[str] = Field(
        default_factory=auto_stats_server_uri, description="Stats server for remote evaluations"
    )
    experiment_id: Optional[str] = Field(default=None, description="Experiment identifier (used as W&B group)")
    run_dir: Optional[str] = Field(default=None, description="Directory where configs/logs are written")
    scheduler_state: Any | None = Field(default=None, description="Typed ExperimentState instance (optional)")

    # Hook configuration
    on_eval_completed: Optional[Any] = Field(default=None, description="Hook called when evaluation completes")
    on_job_dispatch: Optional[Any] = Field(default=None, description="Hook called after job dispatch")

    def invoke(self, args):
        """Run the adaptive experiment."""

        # Apply CLI args to configuration (minimal, common ones)
        if "experiment_id" in args:
            # With the new argument system, experiment_id might be set both ways
            if self.experiment_id is None:
                self.experiment_id = args["experiment_id"]
            # If already set and different, that's an error
            elif self.experiment_id != args["experiment_id"]:
                raise ValueError(
                    f"experiment_id mismatch: config has '{self.experiment_id}', args has '{args['experiment_id']}'"
                )

        # Optional scheduler/dispatcher type overrides via args
        if "scheduler_type" in args:
            self.scheduler_type = SchedulerType(args["scheduler_type"])  # type: ignore[arg-type]
        if "dispatcher_type" in args:
            self.dispatcher_type = DispatcherType(args["dispatcher_type"])  # type: ignore[arg-type]

        # Apply minimal CLI overrides directly to typed scheduler config

        if self.scheduler_type == SchedulerType.TRAIN_AND_EVAL:
            if not isinstance(self.scheduler_config, TrainAndEvalConfig):
                raise ValueError("scheduler_config must be a TrainAndEvalConfig instance for TRAIN_AND_EVAL")
            if "max_trials" in args:
                self.scheduler_config.max_trials = int(args["max_trials"])  # type: ignore[reportAttributeAccessIssue]
            if "gpus" in args:
                self.scheduler_config.gpus = int(args["gpus"])  # type: ignore[reportAttributeAccessIssue]
            if "recipe_module" in args:
                self.scheduler_config.recipe_module = str(args["recipe_module"])  # type: ignore[reportAttributeAccessIssue]
            if "train_entrypoint" in args:
                self.scheduler_config.train_entrypoint = str(args["train_entrypoint"])  # type: ignore[reportAttributeAccessIssue]
            if "eval_entrypoint" in args:
                self.scheduler_config.eval_entrypoint = str(args["eval_entrypoint"])  # type: ignore[reportAttributeAccessIssue]

        if self.scheduler_type == SchedulerType.BATCHED_SYNCED:
            if not isinstance(self.scheduler_config, BatchedSyncedSchedulerConfig):
                raise ValueError("scheduler_config must be a BatchedSyncedSchedulerConfig instance for BATCHED_SYNCED")
            if "max_trials" in args:
                self.scheduler_config.max_trials = int(args["max_trials"])  # type: ignore[reportAttributeAccessIssue]
            if "batch_size" in args:
                self.scheduler_config.batch_size = int(args["batch_size"])  # type: ignore[reportAttributeAccessIssue]
            if "gpus" in args:
                self.scheduler_config.gpus = int(args["gpus"])  # type: ignore[reportAttributeAccessIssue]
            if "recipe_module" in args:
                self.scheduler_config.recipe_module = str(args["recipe_module"])  # type: ignore[reportAttributeAccessIssue]
            if "train_entrypoint" in args:
                self.scheduler_config.train_entrypoint = str(args["train_entrypoint"])  # type: ignore[reportAttributeAccessIssue]
            if "eval_entrypoint" in args:
                self.scheduler_config.eval_entrypoint = str(args["eval_entrypoint"])  # type: ignore[reportAttributeAccessIssue]

        # Set up experiment ID
        experiment_id = self.experiment_id or f"adaptive_{int(time.time())}"

        # Configure run_dir and logging
        if self.run_dir is None:
            self.run_dir = f"{self.system.data_dir}/adaptive/{experiment_id}"
        init_logging(run_dir=self.run_dir)

        # Create scheduler based on type
        scheduler = self._create_scheduler()

        # Configure components
        store = WandbStore(entity=self.wandb.entity, project=self.wandb.project)

        # Create dispatcher based on type
        dispatcher = self._create_dispatcher()

        # Create and run controller
        controller = AdaptiveController(
            experiment_id=experiment_id,
            scheduler=scheduler,
            dispatcher=dispatcher,
            store=store,
            config=self.config,
            on_eval_completed=self.on_eval_completed,
            on_job_dispatch=self.on_job_dispatch,
        )

        # Persist configuration
        try:
            import os

            os.makedirs(self.run_dir, exist_ok=True)
            with open(f"{self.run_dir}/adaptive_config.json", "w") as f:
                f.write(self.model_dump_json(indent=2))
        except Exception:
            logger.warning("[AdaptiveTool] Failed to write adaptive_config.json", exc_info=True)

        controller.run()

    def _create_scheduler(self):
        """Create scheduler instance based on scheduler_type."""
        # Optional: validate scheduler_state implements ExperimentState protocol
        if self.scheduler_state is not None and not isinstance(self.scheduler_state, ExperimentState):
            raise ValueError("scheduler_state must implement ExperimentState protocol (model_dump/model_validate)")

        if self.scheduler_type == SchedulerType.TRAIN_AND_EVAL:
            if not isinstance(self.scheduler_config, TrainAndEvalConfig):
                raise ValueError("scheduler_config must be a TrainAndEvalConfig instance for TRAIN_AND_EVAL")

            # Inject stats_server_uri from AdaptiveTool into scheduler config
            if hasattr(self.scheduler_config, "stats_server_uri"):
                self.scheduler_config.stats_server_uri = self.stats_server_uri

            return TrainAndEvalScheduler(self.scheduler_config, state=self.scheduler_state)

        if self.scheduler_type == SchedulerType.BATCHED_SYNCED:
            if not isinstance(self.scheduler_config, BatchedSyncedSchedulerConfig):
                raise ValueError("scheduler_config must be a BatchedSyncedSchedulerConfig instance for BATCHED_SYNCED")

            # Inject stats_server_uri from AdaptiveTool into scheduler config
            if hasattr(self.scheduler_config, "stats_server_uri"):
                self.scheduler_config.stats_server_uri = self.stats_server_uri

            return BatchedSyncedOptimizingScheduler(self.scheduler_config, state=self.scheduler_state)

        else:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")

    def _create_dispatcher(self):
        """Create dispatcher instance based on dispatcher_type."""
        if self.dispatcher_type == DispatcherType.LOCAL:
            return LocalDispatcher(capture_output=self.capture_output)

        elif self.dispatcher_type == DispatcherType.SKYPILOT:
            # SkypilotDispatcher handles both train jobs (via launch.py) and eval jobs (via ./tools/run.py)
            dispatcher = SkypilotDispatcher()
            logger.info("[AdaptiveTool] Using Skypilot for both training and evaluation")
            return dispatcher

        else:
            raise ValueError(f"Unsupported dispatcher type: {self.dispatcher_type}")
