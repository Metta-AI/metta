"""Policy interfaces and implementations for CoGames."""

from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn


class Policy(ABC):
    """Abstract base class for policies."""

    @abstractmethod
    def step(self, observations: list[Any]) -> list[Any]:
        """Get actions for given observations for all agents.

        Args:
            observations: The current observations from the environment

        Returns:
            The actions to take
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the policy state."""
        pass


class TrainablePolicy(Policy):
    """Abstract base class for trainable policies."""

    @abstractmethod
    def network(self) -> nn.Module: ...

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str) -> None: ...
