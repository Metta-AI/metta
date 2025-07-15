"""Optimizer wrapper for Metta."""

import logging
from typing import Any, Dict, Optional, Tuple

import torch

from metta.agent.metta_agent import MettaAgent

logger = logging.getLogger(__name__)


class Optimizer:
    """Wrapper for PyTorch optimizers with gradient accumulation and clipping support.

    This provides a clean interface for the optimization step, handling:
    - Gradient accumulation across minibatches
    - Gradient clipping
    - Optional weight clipping on the policy
    """

    def __init__(
        self,
        optimizer_type: str = "adam",
        policy: Optional[MettaAgent] = None,
        learning_rate: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        max_grad_norm: float = 0.5,
    ):
        """Initialize optimizer wrapper.

        Args:
            optimizer_type: Type of optimizer ("adam" or "muon")
            policy: Policy to optimize
            learning_rate: Learning rate
            betas: Beta parameters for Adam/Muon
            eps: Epsilon for numerical stability
            weight_decay: Weight decay coefficient
            max_grad_norm: Maximum gradient norm for clipping
        """
        if policy is None:
            raise ValueError("Policy must be provided to Optimizer")
        logger.info(f"Creating optimizer... Using {optimizer_type.capitalize()} optimizer with lr={learning_rate}")

        self.policy = policy
        self.max_grad_norm = max_grad_norm

        if optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(
                policy.parameters(),
                lr=learning_rate,
                betas=betas,
                eps=eps,
                weight_decay=float(weight_decay),  # type: ignore - PyTorch accepts float
            )
        elif optimizer_type == "muon":
            from heavyball import ForeachMuon

            self.optimizer = ForeachMuon(
                policy.parameters(),
                lr=learning_rate,
                betas=betas,
                eps=eps,
                weight_decay=float(weight_decay),  # type: ignore - PyTorch accepts float
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}. Choose 'adam' or 'muon'")

    def step(self, loss: torch.Tensor, epoch: int, accumulate_steps: int = 1):
        """Perform optimization step with gradient accumulation.

        Args:
            loss: Loss tensor to backpropagate
            epoch: Current epoch (for accumulation check)
            accumulate_steps: Number of steps to accumulate gradients
        """
        self.optimizer.zero_grad()
        loss.backward()

        if (epoch + 1) % accumulate_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Optional weight clipping
            if hasattr(self.policy, "clip_weights"):
                self.policy.clip_weights()

    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state dict."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state dict."""
        self.optimizer.load_state_dict(state_dict)

    @property
    def param_groups(self):
        """Access to optimizer param groups (for learning rate etc)."""
        return self.optimizer.param_groups
