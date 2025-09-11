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
        """Initialize gradient stats component.

        Args:
            config: Gradient stats configuration
        """
        super().__init__(config)

    def on_epoch_end(self, trainer: "Trainer", epoch: int) -> None:
        """Compute gradient statistics.

        Args:
            trainer: The trainer instance
        """
        policy = trainer._policy

        if not policy.parameters():
            return {}

        all_gradients = []
        for param in policy.parameters():
            if param.grad is not None:
                all_gradients.append(param.grad.view(-1))

        if len(all_gradients) == 0:
            return {}

        all_gradients_tensor = torch.cat(all_gradients).to(torch.float32)

        grad_mean = all_gradients_tensor.mean()
        grad_variance = all_gradients_tensor.var()
        grad_norm = all_gradients_tensor.norm(2)

        grad_stats = {
            "grad/mean": grad_mean.item(),
            "grad/variance": grad_variance.item(),
            "grad/norm": grad_norm.item(),
        }

        return grad_stats
