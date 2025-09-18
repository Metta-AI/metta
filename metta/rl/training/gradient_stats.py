"""Gradient statistics computation callback."""

import logging
from typing import TYPE_CHECKING

import torch
from pydantic import Field

from metta.mettagrid.config import Config
from metta.rl.training.component import TrainerComponent

if TYPE_CHECKING:
    from metta.rl.trainer import Trainer

logger = logging.getLogger(__name__)


class GradientStatsConfig(Config):
    """Configuration for gradient statistics computation."""

    epoch_interval: int = Field(default=0, ge=0)  # 0 to disable
    """How often to compute gradient statistics (in epochs)"""


class GradientStatsComponent(TrainerComponent):
    """Computes gradient statistics for monitoring."""

    def __init__(self, config: GradientStatsConfig):
        """Initialize gradient stats component."""
        interval = max(1, config.epoch_interval) if config.epoch_interval else 0
        super().__init__(epoch_interval=interval)
        self._config = config
        self._master_only = True

    def on_epoch_end(self, trainer: "Trainer", epoch: int) -> None:
        """Compute gradient statistics and stash on trainer."""
        if self._config.epoch_interval == 0:
            return
        if epoch % self._config.epoch_interval != 0:
            return

        policy = trainer.policy
        gradients = [param.grad.view(-1) for param in policy.parameters() if param.grad is not None]
        if not gradients:
            return

        grad_tensor = torch.cat(gradients).to(torch.float32)
        grad_stats = {
            "grad/mean": grad_tensor.mean().item(),
            "grad/variance": grad_tensor.var().item(),
            "grad/norm": grad_tensor.norm(2).item(),
        }

        trainer.latest_grad_stats = grad_stats
        stats_reporter = getattr(trainer, "stats_reporter", None)
        if stats_reporter and hasattr(stats_reporter, "update_grad_stats"):
            stats_reporter.update_grad_stats(grad_stats)
