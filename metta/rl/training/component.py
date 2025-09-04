"""Base training component infrastructure."""

import logging
from typing import TYPE_CHECKING, Any, Dict

from metta.mettagrid.config import Config

if TYPE_CHECKING:
    from metta.rl.trainer_v2 import Trainer

logger = logging.getLogger(__name__)


class TrainerComponentConfig(Config):
    """Base configuration for training components."""

    interval: int = 1
    """How often to trigger component callbacks (in epochs)"""


class TrainerComponent:
    """Base class for training components."""

    def __init__(self, config: TrainerComponentConfig):
        """Initialize component.

        Args:
            config: Component configuration
        """
        self.config = config
        self.interval = config.interval

    def register(self, trainer: "Trainer") -> None:
        """Register this component with the trainer.

        Args:
            trainer: The trainer to register with
        """
        trainer.register_component(self)

    def on_step(self, trainer: "Trainer", infos: Dict[str, Any]) -> None:
        """Called after each environment step.

        Args:
            trainer: The trainer instance
            infos: Step information from environment
        """
        pass

    def on_epoch_end(self, trainer: "Trainer", epoch: int) -> None:
        """Called at the end of an epoch.

        Args:
            trainer: The trainer instance
            epoch: The current epoch number
        """
        pass

    def on_training_complete(self, trainer: "Trainer") -> None:
        """Called when training completes successfully.

        Args:
            trainer: The trainer instance
        """
        pass

    def on_failure(self, trainer: "Trainer") -> None:
        """Called when training fails.

        Args:
            trainer: The trainer instance
        """
        pass


# Backward compatibility aliases
ComponentConfig = TrainerComponentConfig
TrainingComponent = TrainerComponent


class MasterComponent(TrainerComponent):
    """Base class for training components that should only run on the master process.

    These components automatically check if they're on the master process before
    registering and running callbacks.
    """

    def register(self, trainer: "Trainer") -> None:
        """Register this component with the trainer only if on master.

        Args:
            trainer: The trainer instance to register with
        """
        # Only register if we're on the master process
        if hasattr(trainer, "distributed_helper") and trainer.distributed_helper.is_master():
            super().register(trainer)
        # If distributed_helper doesn't exist yet, assume we should register
        elif not hasattr(trainer, "distributed_helper"):
            super().register(trainer)
