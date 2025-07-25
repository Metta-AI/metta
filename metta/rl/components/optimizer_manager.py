"""Manages optimizer creation and state."""

import logging
from typing import Any, Optional

import torch
from heavyball import ForeachMuon

from metta.rl.trainer_checkpoint import TrainerCheckpoint
from metta.rl.trainer_config import OptimizerConfig

logger = logging.getLogger(__name__)


class OptimizerManager:
    """Manages optimizer creation, configuration, and state persistence."""

    def __init__(
        self,
        optimizer_config: OptimizerConfig,
        device: torch.device,
    ):
        """Initialize optimizer manager.

        Args:
            optimizer_config: Optimizer configuration
            device: Device to run computations on
        """
        self.optimizer_config = optimizer_config
        self.device = device
        self._optimizer = None

    def create_optimizer(self, agent: Any) -> Any:
        """Create optimizer for the agent.

        Args:
            agent: The policy/agent whose parameters to optimize

        Returns:
            Configured optimizer instance
        """
        optimizer_type = self.optimizer_config.type

        # Select optimizer class
        if optimizer_type == "adam":
            opt_cls = torch.optim.Adam
            # Adam expects float weight_decay
            weight_decay = float(self.optimizer_config.weight_decay)
        else:
            opt_cls = ForeachMuon
            # ForeachMuon expects int weight_decay
            weight_decay = int(self.optimizer_config.weight_decay)

        # Create optimizer
        self._optimizer = opt_cls(
            agent.parameters(),
            lr=self.optimizer_config.learning_rate,
            betas=(self.optimizer_config.beta1, self.optimizer_config.beta2),
            eps=self.optimizer_config.eps,
            weight_decay=weight_decay,
        )

        logger.info(
            f"Created {optimizer_type} optimizer with lr={self.optimizer_config.learning_rate}, "
            f"weight_decay={weight_decay}"
        )

        return self._optimizer

    def load_state_from_checkpoint(
        self,
        optimizer: Any,
        checkpoint: Optional[TrainerCheckpoint],
    ) -> bool:
        """Load optimizer state from checkpoint if available.

        Args:
            optimizer: The optimizer to load state into
            checkpoint: Optional checkpoint containing optimizer state

        Returns:
            True if state was loaded successfully, False otherwise
        """
        if checkpoint and checkpoint.optimizer_state_dict:
            try:
                optimizer.load_state_dict(checkpoint.optimizer_state_dict)
                logger.info("Successfully loaded optimizer state from checkpoint")
                return True
            except ValueError as e:
                logger.warning(f"Optimizer state dict doesn't match: {e}. Starting with fresh optimizer state.")
                return False
        return False

    def get_current_lr(self, optimizer: Optional[Any] = None) -> float:
        """Get current learning rate from optimizer.

        Args:
            optimizer: Optional optimizer instance, uses stored optimizer if not provided

        Returns:
            Current learning rate
        """
        opt = optimizer or self._optimizer
        if opt is None:
            return self.optimizer_config.learning_rate
        return opt.param_groups[0]["lr"]

    def update_lr(self, new_lr: float, optimizer: Optional[Any] = None) -> None:
        """Update learning rate.

        Args:
            new_lr: New learning rate value
            optimizer: Optional optimizer instance, uses stored optimizer if not provided
        """
        opt = optimizer or self._optimizer
        if opt is not None:
            for param_group in opt.param_groups:
                param_group["lr"] = new_lr
            logger.info(f"Updated learning rate to {new_lr}")
