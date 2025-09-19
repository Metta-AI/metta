"""Hyperparameter scheduling management."""

import logging
from typing import TYPE_CHECKING

from metta.rl.hyperparameter_scheduler import step_hyperparameters
from metta.rl.training.component import TrainerComponent
from mettagrid.config import Config

if TYPE_CHECKING:
    from metta.rl.trainer import Trainer

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

    def on_epoch_end(self, trainer: "Trainer", epoch: int) -> None:
        """Update hyperparameters.

        Args:
            trainer: The trainer instance
        """
        # Update learning rate and other hyperparameters
        step_hyperparameters(
            trainer._cfg,
            trainer.optimizer,
            trainer.trainer_state.agent_step,
            trainer._cfg.total_timesteps,
            logger,
        )
