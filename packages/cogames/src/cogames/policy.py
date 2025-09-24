"""Policy interfaces and implementations for CoGames."""

from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn


class Policy(ABC):
    """Abstract base class for policies."""

    @abstractmethod
    def step(self, agent_id: int, agent_obs: Any) -> Any:
        """Get action for a single agent given its observation.

        Args:
            agent_id: The ID of the agent
            agent_obs: The observation for this specific agent

        Returns:
            The action for this agent to take
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
