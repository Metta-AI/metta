import logging
from typing import Tuple

import torch
from heavyball import ForeachMuon

from metta.rl.trainer_checkpoint import TrainerCheckpoint
from metta.rl.trainer_config import OptimizerConfig

__all__ = ["Optimizer", "create_optimizer"]

logger = logging.getLogger(__name__)


class Optimizer:
    """Unified wrapper for Adam and Heavyball-Muon optimizers with state management."""

    def __init__(
        self,
        optimizer_type: str,
        policy: torch.nn.Module,
        learning_rate: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.optimizer_type = optimizer_type
        self.initial_lr = learning_rate

        opt_cls = torch.optim.Adam if optimizer_type == "adam" else ForeachMuon
        if optimizer_type != "adam":
            weight_decay = int(weight_decay)

        self.optimizer = opt_cls(
            policy.parameters(),
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.max_grad_norm = max_grad_norm
        self.param_groups = self.optimizer.param_groups

    def step(self, loss: torch.Tensor, epoch: int, accumulate_minibatches: int = 1) -> None:
        """Back-propagate *loss* and update parameters with optional gradient accumulation."""
        self.optimizer.zero_grad()
        loss.backward()

        if (epoch + 1) % accumulate_minibatches == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for group in self.optimizer.param_groups for p in group["params"]],
                self.max_grad_norm,
            )
            self.optimizer.step()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

    def load_from_checkpoint(self, checkpoint: TrainerCheckpoint | None) -> bool:
        """Load optimizer state from checkpoint if available.

        Args:
            checkpoint: Optional checkpoint containing optimizer state

        Returns:
            True if state was loaded successfully, False otherwise
        """
        if checkpoint and checkpoint.optimizer_state_dict:
            try:
                self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
                logger.info("Successfully loaded optimizer state from checkpoint")
                return True
            except ValueError as e:
                logger.warning(f"Optimizer state dict doesn't match: {e}. Starting with fresh optimizer state.")
                return False
        return False

    @property
    def current_lr(self) -> float:
        """Get current learning rate."""
        return self.param_groups[0]["lr"]

    def update_lr(self, new_lr: float) -> None:
        """Update learning rate."""
        for param_group in self.param_groups:
            param_group["lr"] = new_lr
        logger.info(f"Updated learning rate to {new_lr}")

    def zero_grad(self) -> None:
        """Zero out gradients."""
        self.optimizer.zero_grad()

    # Thin wrappers to mimic torch.optim.Optimizer API
    def state_dict(self):  # noqa: D401  # "Returns" docstring skipped for brevity
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


def create_optimizer(
    policy: torch.nn.Module,
    optimizer_config: OptimizerConfig,
    checkpoint: TrainerCheckpoint | None = None,
) -> Optimizer:
    """Create an optimizer from configuration and optionally load checkpoint state.

    Args:
        policy: The policy/agent whose parameters to optimize
        optimizer_config: Optimizer configuration
        checkpoint: Optional checkpoint to load state from

    Returns:
        Configured optimizer instance
    """
    # Create optimizer
    optimizer = Optimizer(
        optimizer_type=optimizer_config.type,
        policy=policy,
        learning_rate=optimizer_config.learning_rate,
        betas=(optimizer_config.beta1, optimizer_config.beta2),
        eps=optimizer_config.eps,
        weight_decay=optimizer_config.weight_decay,
        max_grad_norm=1.0,  # This should come from PPO config if needed
    )

    # Load checkpoint if available
    optimizer.load_from_checkpoint(checkpoint)

    logger.info(
        f"Created {optimizer_config.type} optimizer with lr={optimizer_config.learning_rate}, "
        f"weight_decay={optimizer_config.weight_decay}"
    )

    return optimizer
