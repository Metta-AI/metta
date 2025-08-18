import abc
from typing import Any, Optional

import torch
from tensordict import TensorDict
from torch import nn
from torchrl.data import Composite


class MettaAgentInterface(nn.Module, metaclass=abc.ABCMeta):
    """
    Abstract base class for all Metta agents.
    Enforces a consistent API for policies, memory management,
    and environment initialization.
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(
        self,
        td: dict[str, torch.Tensor],
        state: Optional[Any] = None,
        action: Optional[torch.Tensor] = None,
    ) -> TensorDict:
        """Forward pass through the policy. Must return a TensorDict."""
        raise NotImplementedError

    def reset_memory(self) -> None:
        """Reset recurrent memory if agent supports it."""
        return None

    def get_memory(self) -> dict:
        """Return current recurrent memory state (if any)."""
        return {}

    @abc.abstractmethod
    def initialize_to_environment(
        self,
        features: dict[str, dict],
        action_names: list[str],
        action_max_params: list[int],
        device: torch.device,
        is_training: bool = True,
    ):
        """Initialize the agent given environment-provided features and actions."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_agent_experience_spec(self) -> Composite:
        """Return the expected experience spec for training (obs/action/other)."""
        raise NotImplementedError

    def l2_init_loss(self) -> torch.Tensor:
        """Optional regularization loss used during training."""
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def clip_weights(self) -> None:
        """Optional weight clipping hook for stability."""
        return None

    def update_l2_init_weight_copy(self):
        """Optional update for weight-copy state (e.g., L2 regularization)."""
        return None

    def compute_weight_metrics(self, delta: float = 0.01) -> list[dict]:
        """Optional hook to compute weight-related metrics."""
        return []
