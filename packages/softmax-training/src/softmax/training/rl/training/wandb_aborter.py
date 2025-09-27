"""Trainer component that respects wandb abort tags."""

from __future__ import annotations

import logging

from metta.common.wandb.context import WandbRun
from metta.common.wandb.utils import abort_requested
from mettagrid.config import Config
from softmax.training.rl.training import ComponentContext, TrainerComponent

logger = logging.getLogger(__name__)


class WandbAborterConfig(Config):
    """Configuration for wandb abort polling."""

    epoch_interval: int = 5
    """How often to poll wandb for abort tags (in epochs)."""


class WandbAborter(TrainerComponent):
    """Polls wandb for abort tags and stops training when detected."""

    def __init__(
        self,
        *,
        wandb_run: WandbRun | None,
        config: WandbAborterConfig | None = None,
    ) -> None:
        cfg = config or WandbAborterConfig()
        super().__init__(epoch_interval=cfg.epoch_interval)
        self._wandb_run = wandb_run
        self._config = cfg

    def on_epoch_end(self, epoch: int) -> None:  # noqa: D401 - documented in base class
        context: ComponentContext = self.context
        distributed_helper = context.distributed

        target_timesteps: int | None = None

        if distributed_helper.is_master() and self._wandb_run:
            if abort_requested(self._wandb_run):
                target_timesteps = int(context.agent_step)
                context.config.total_timesteps = target_timesteps
                logger.info("Abort tag detected. Stopping training at agent_step=%s", target_timesteps)

                try:
                    self._wandb_run.config.update({"trainer.total_timesteps": target_timesteps}, allow_val_change=True)
                except Exception as exc:  # noqa: BLE001 - we only log the failure
                    logger.warning("Failed to update wandb config with abort timesteps: %s", exc, exc_info=True)

        if distributed_helper.is_distributed:
            target_timesteps = distributed_helper.broadcast_from_master(target_timesteps)

        if target_timesteps is not None:
            context.config.total_timesteps = int(target_timesteps)
