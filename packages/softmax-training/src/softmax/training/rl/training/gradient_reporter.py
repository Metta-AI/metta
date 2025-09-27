"""Gradient statistics computation callback."""

import logging

import torch
from pydantic import Field

from mettagrid.config import Config
from softmax.training.rl.training import TrainerComponent

logger = logging.getLogger(__name__)


class GradientReporterConfig(Config):
    """Configuration for gradient statistics computation."""

    epoch_interval: int = Field(default=0, ge=0)
    """How often to compute gradient statistics (in epochs)."""


class GradientReporter(TrainerComponent):
    """Computes gradient statistics for monitoring."""

    def __init__(self, config: GradientReporterConfig):
        """Initialize gradient reporter component."""
        enabled = config.epoch_interval > 0
        super().__init__(epoch_interval=config.epoch_interval if enabled else 0)
        self._master_only = True
        self._enabled = enabled

    def on_epoch_end(self, epoch: int) -> None:
        """Compute gradient statistics and stash on trainer."""
        if not self._enabled:
            return

        context = self.context
        policy = context.policy
        gradients = [param.grad.view(-1) for param in policy.parameters() if param.grad is not None]
        if not gradients:
            return

        grad_tensor = torch.cat(gradients).to(torch.float32)
        grad_stats = {
            "grad/mean": grad_tensor.mean().item(),
            "grad/variance": grad_tensor.var().item(),
            "grad/norm": grad_tensor.norm(2).item(),
        }

        self.context.update_gradient_stats(grad_stats)

        stats_reporter = context.stats_reporter
        if stats_reporter and hasattr(stats_reporter, "update_grad_stats"):
            stats_reporter.update_grad_stats(grad_stats)
