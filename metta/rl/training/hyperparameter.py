"""Hyperparameter scheduling management."""

import logging

from metta.mettagrid.config import Config
from metta.rl.hyperparameter_scheduler import step_hyperparameters
from metta.rl.training.component import TrainerComponent

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
        super().__init__(config)

    def on_epoch_end(self, epoch: int) -> None:
        """Update hyperparameters for the current training epoch."""
        context = self.context
        # Update learning rate and other hyperparameters
        step_hyperparameters(
            context.cfg,
            context.optimizer,
            context.trainer_state.agent_step,
            context.cfg.total_timesteps,
            logger,
        )
