"""Hyperparameter scheduling management."""

import logging
from typing import TYPE_CHECKING

from metta.rl.hyperparameter_scheduler import step_hyperparameters
from metta.rl.training.component import ComponentConfig, MasterComponent

if TYPE_CHECKING:
    from metta.rl.trainer_v2 import Trainer

logger = logging.getLogger(__name__)


class HyperparameterConfig(ComponentConfig):
    """Configuration for hyperparameter scheduling."""

    interval: int = 1
    """How often to update hyperparameters (in epochs)"""


class HyperparameterComponent(MasterComponent):
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
            trainer.trainer_cfg,
            trainer.optimizer,
            trainer.trainer_state.agent_step,
            trainer.trainer_cfg.total_timesteps,
            logger,
        )
