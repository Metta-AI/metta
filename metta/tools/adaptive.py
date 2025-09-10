"""Simplified tool for adaptive experiments."""

import time
from enum import StrEnum
from typing import Any, Optional

from metta.adaptive.adaptive_config import AdaptiveConfig
from metta.adaptive.adaptive_controller import AdaptiveController
from metta.common.tool import Tool
from metta.common.wandb.wandb_context import WandbConfig
from metta.tools.utils.auto_config import auto_wandb_config


class SchedulerType(StrEnum):
    """Available scheduler types for adaptive experiments."""
    TRAIN_AND_EVAL = "train_and_eval"


class AdaptiveTool(Tool):
    """Simple tool for running adaptive experiments."""

    # Core components
    scheduler_type: SchedulerType
    scheduler_config: dict[str, Any] = {}
    config: AdaptiveConfig = AdaptiveConfig()

    # Standard tool configuration
    wandb: WandbConfig = auto_wandb_config()
    experiment_id: Optional[str] = None

    def invoke(self, args, overrides):
        """Run the adaptive experiment"""
        from metta.adaptive.dispatcher import LocalDispatcher, RoutingDispatcher
        from metta.adaptive.stores import WandbStore

        # Set up experiment ID
        experiment_id = self.experiment_id or f"adaptive_{int(time.time())}"

        # Create scheduler based on type
        scheduler = self._create_scheduler()

        # Configure components
        store = WandbStore(
            entity=self.wandb.entity,
            project=self.wandb.project
        )
        dispatcher = RoutingDispatcher(
            routes={},  # Use defaults
            default_dispatcher=LocalDispatcher()
        )

        # Create and run controller
        controller = AdaptiveController(
            experiment_id=experiment_id,
            scheduler=scheduler,
            dispatcher=dispatcher,
            store=store,
            config=self.config,
        )

        controller.run()

    def _create_scheduler(self):
        """Create scheduler instance based on scheduler_type."""
        if self.scheduler_type == SchedulerType.TRAIN_AND_EVAL:
            from metta.adaptive.schedulers.train_and_eval import TrainAndEvalConfig, TrainAndEvalScheduler

            config = TrainAndEvalConfig(**self.scheduler_config)
            return TrainAndEvalScheduler(config)

        else:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")
