"""Lightweight abstract Policy base class to avoid circular imports."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.data import Composite, UnboundedDiscrete


class Policy(ABC, nn.Module):
    """Abstract base class defining the policy interface."""

    @abstractmethod
    def forward(self, td: TensorDict, action: torch.Tensor | None = None) -> TensorDict:
        """Forward pass used by losses / rollout."""
        raise NotImplementedError

    def get_agent_experience_spec(self) -> Composite:
        """Default experience spec; concrete policies may override."""
        return Composite(
            env_obs=UnboundedDiscrete(shape=torch.Size([200, 3]), dtype=torch.uint8),
            dones=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
            truncateds=UnboundedDiscrete(shape=torch.Size([]), dtype=torch.float32),
        )

    def initialize_to_environment(self, env_metadata, device: torch.device) -> None:
        """Hook for env-specific initialization (optional)."""
        return None

    @property
    @abstractmethod
    def device(self) -> torch.device:
        raise NotImplementedError

    @property
    @abstractmethod
    def total_params(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def reset_memory(self) -> None:
        raise NotImplementedError
