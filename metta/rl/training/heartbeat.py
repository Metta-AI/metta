"""Heartbeat writer for training monitoring."""

import logging
from typing import TYPE_CHECKING

from metta.common.util.heartbeat import record_heartbeat
from metta.mettagrid.config import Config
from metta.rl.training.component import TrainerComponent

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class HeartbeatConfig(Config):
    """Configuration for heartbeat monitoring."""

    epoch_interval: int = 1
    """How often to write heartbeat (in epochs)"""


class HeartbeatWriter(TrainerComponent):
    """Writes heartbeat signals for monitoring training progress."""

    def on_epoch_end(self, epoch: int) -> None:
        record_heartbeat()
        logger.debug(f"Heartbeat recorded at epoch {epoch}")
