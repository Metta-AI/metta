"""Heartbeat writer for training monitoring."""

import logging

from metta.common.util.heartbeat import record_heartbeat
from mettagrid.config import Config
from softmax.training.rl.training import TrainerComponent

logger = logging.getLogger(__name__)


class HeartbeatConfig(Config):
    """Configuration for heartbeat monitoring."""

    epoch_interval: int = 1
    """How often to write heartbeat (in epochs)."""


class Heartbeat(TrainerComponent):
    """Writes heartbeat signals for monitoring training progress."""

    def on_epoch_end(self, epoch: int) -> None:
        record_heartbeat()
        logger.debug(f"Heartbeat recorded at epoch {epoch}")
