"""Gradient statistics computation callback."""

import logging
from typing import TYPE_CHECKING

from metta.rl.optimization import compute_gradient_stats
from metta.rl.training.component import ComponentConfig, MasterComponent

if TYPE_CHECKING:
    from metta.rl.trainer_v2 import Trainer

logger = logging.getLogger(__name__)


class GradientStatsConfig(ComponentConfig):
    """Configuration for gradient statistics computation."""

    interval: int = 50
    """How often to compute gradient statistics (in epochs)"""


class GradientStatsComponent(MasterComponent):
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
        with trainer.timer("grad_stats"):
            grad_stats = compute_gradient_stats(trainer.policy)
            # Store grad stats on trainer so other components can access them
            trainer.latest_grad_stats = grad_stats
            logger.debug(f"Gradient stats computed at epoch {trainer.trainer_state.epoch}")
