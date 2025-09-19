"""Hyperparameter scheduling management."""

import logging

from metta.rl.hyperparameter_scheduler import step_hyperparameters
from metta.rl.training.component import TrainerComponent
from mettagrid.config import Config

logger = logging.getLogger(__name__)


class HyperparameterConfig(Config):
    """Configuration for hyperparameter scheduling."""

    interval: int = 1
    """How often to update hyperparameters (in epochs)"""


class HyperparameterComponent(TrainerComponent):
    """Manages hyperparameter scheduling."""

    def __init__(self, config: HyperparameterConfig):
        """Initialize hyperparameter component.

        Args:
            config: Hyperparameter configuration
        """
        super().__init__(epoch_interval=config.interval)

    def on_epoch_end(self, epoch: int) -> None:
        """Update hyperparameters for the current training epoch."""
        context = self.context
        trainer_cfg = context.config

        if not getattr(trainer_cfg.hyperparameter_scheduler, "enabled", False):
            return

        # Update learning rate and other hyperparameters across ranks
        step_hyperparameters(
            trainer_cfg,
            context.optimizer,
            context.agent_step,
            trainer_cfg.total_timesteps,
            logger,
        )
