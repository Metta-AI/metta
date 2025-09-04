"""Heartbeat writer for training monitoring."""

import logging
from typing import TYPE_CHECKING

from metta.common.util.heartbeat import record_heartbeat
from metta.rl.training.component import ComponentConfig, MasterComponent

if TYPE_CHECKING:
    from metta.rl.trainer_v2 import Trainer

logger = logging.getLogger(__name__)


class HeartbeatConfig(ComponentConfig):
    """Configuration for heartbeat monitoring."""

    interval: int = 1
    """How often to write heartbeat (in epochs)"""


class HeartbeatWriter(MasterComponent):
    """Writes heartbeat signals for monitoring training progress."""

    def __init__(self, config: HeartbeatConfig):
        """Initialize heartbeat writer.

        Args:
            config: Heartbeat configuration
        """
        super().__init__(config)

    def on_epoch_end(self, trainer: "Trainer", epoch: int) -> None:
        """Called at the end of each epoch.

        Args:
            trainer: The trainer instance
        """
        epoch = trainer.trainer_state.epoch
        record_heartbeat()
        logger.debug(f"Heartbeat recorded at epoch {epoch}")
