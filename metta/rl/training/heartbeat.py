"""Heartbeat writer for training monitoring."""

import logging
from typing import TYPE_CHECKING

from metta.common.util.heartbeat import record_heartbeat
from metta.rl.training.component import TrainerComponent, TrainerComponentConfig

if TYPE_CHECKING:
    from metta.rl.trainer_v2 import Trainer

logger = logging.getLogger(__name__)


class HeartbeatConfig(TrainerComponentConfig):
    """Configuration for heartbeat monitoring."""

    interval: int = 1
    """How often to write heartbeat (in epochs)"""


class HeartbeatWriter(TrainerComponent):
    """Writes heartbeat signals for monitoring training progress."""

    def on_epoch_end(self, trainer: "Trainer", epoch: int) -> None:
        record_heartbeat()
        logger.debug(f"Heartbeat recorded at epoch {epoch}")
