"""Trainer component that respects wandb abort tags."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from metta.common.wandb.context import WandbRun
from metta.common.wandb.utils import abort_requested
from metta.rl.training.component import TrainerComponent

if TYPE_CHECKING:  # pragma: no cover - circular import guard
    from metta.rl.trainer import Trainer

logger = logging.getLogger(__name__)


class WandbAbortComponent(TrainerComponent):
    """Polls wandb for abort tags and stops training when detected."""

    def __init__(self, wandb_run: WandbRun | None, epoch_interval: int = 5) -> None:
        super().__init__(epoch_interval=epoch_interval)
        self._wandb_run = wandb_run

    def on_epoch_end(self, epoch: int) -> None:  # noqa: D401 - documented in base class
        trainer: Trainer = self._trainer
        distributed_helper = trainer._distributed_helper

        target_timesteps: int | None = None

        if distributed_helper.is_master() and self._wandb_run:
            if abort_requested(self._wandb_run):
                target_timesteps = int(trainer._agent_step)
                trainer._cfg.total_timesteps = target_timesteps
                logger.info("Abort tag detected. Stopping training at agent_step=%s", target_timesteps)

                try:
                    self._wandb_run.config.update({"trainer.total_timesteps": target_timesteps}, allow_val_change=True)
                except Exception as exc:  # noqa: BLE001 - we only log the failure
                    logger.warning("Failed to update wandb config with abort timesteps: %s", exc, exc_info=True)

        if distributed_helper.is_distributed():
            target_timesteps = distributed_helper.broadcast_from_master(target_timesteps)

        if target_timesteps is not None:
            trainer._cfg.total_timesteps = int(target_timesteps)
