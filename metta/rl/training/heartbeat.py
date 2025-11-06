"""Heartbeat writer for training monitoring."""

import logging

import metta.common.util.heartbeat
import metta.rl.training
import mettagrid.base_config

logger = logging.getLogger(__name__)


class HeartbeatConfig(mettagrid.base_config.Config):
    """Configuration for heartbeat monitoring."""

    epoch_interval: int = 1
    """How often to write heartbeat (in epochs)."""


class Heartbeat(metta.rl.training.TrainerComponent):
    """Writes heartbeat signals for monitoring training progress."""

    def on_epoch_end(self, epoch: int) -> None:
        metta.common.util.heartbeat.record_heartbeat()
        logger.debug(f"Heartbeat recorded at epoch {epoch}")
