"""Base policy classes and interfaces."""

from abc import abstractmethod
from pathlib import Path
from typing import Generic, Optional, Tuple, TypeVar

import torch.nn as nn
from pydantic import BaseModel

from mettagrid import MettaGridAction, MettaGridObservation

# Type variable for agent state - can be any type
StateType = TypeVar("StateType")


class AgentPolicy:
    """Base class for per-agent policies.

    AgentPolicy represents the interface for controlling a single agent.
    It provides the step() method that produces actions from observations.
    This is what play.py and evaluation code use directly.
    """

    def step(self, obs: MettaGridObservation) -> MettaGridAction:
        """Get action given an observation."""
        raise NotImplementedError("Subclasses must implement step()")

    def reset(self) -> None:
        """Reset the policy state. Default implementation does nothing."""
        return None


class Policy:
    """Abstract base class for multi-agent policies."""

    @abstractmethod
    def agent_policy(self, agent_id: int) -> AgentPolicy:
        """Get an AgentPolicy instance for a specific agent."""
        ...

    def load_policy_data(self, policy_data_path: str) -> None:
        """Load policy data from a file (no-op for stateless policies)."""
        return None

    def save_policy_data(self, policy_data_path: str) -> None:
        """Persist policy data to a file (no-op for stateless policies)."""
        return None


class StatefulAgentPolicy(AgentPolicy, Generic[StateType]):
    """AgentPolicy wrapper that manages internal state (e.g., for RNNs)."""

    def __init__(self, base_policy: "StatefulPolicyImpl[StateType]", agent_id: int) -> None:
        self._base_policy = base_policy
        self._agent_id = agent_id
        self._state: Optional[StateType] = self._base_policy.agent_state()

    def step(self, obs: MettaGridObservation) -> MettaGridAction:
        action, self._state = self._base_policy.step_with_state(obs, self._state)
        return action

    def reset(self) -> None:
        self._state = self._base_policy.agent_state()


class StatefulPolicyImpl(Generic[StateType]):
    """Base class for stateful policy implementations."""

    @abstractmethod
    def agent_state(self) -> Optional[StateType]:
        """Return the initial state for a new agent, or None if stateless."""
        ...

    def step_with_state(
        self,
        obs: MettaGridObservation,
        state: Optional[StateType],
    ) -> Tuple[MettaGridAction, Optional[StateType]]:
        """Return action and updated state for the provided observation/state."""
        raise NotImplementedError


class TrainablePolicy(Policy):
    """Abstract base class for trainable policies."""

    @abstractmethod
    def network(self) -> nn.Module:
        """Get the underlying neural network for training."""
        ...

    @abstractmethod
    def agent_policy(self, agent_id: int) -> AgentPolicy:
        """Return the per-agent policy for the requested agent id."""
        ...

    def load_policy_data(self, policy_data_path: str) -> None:
        import torch

        self.network().load_state_dict(torch.load(policy_data_path, map_location="cpu"))

    def save_policy_data(self, policy_data_path: str) -> None:
        import torch

        torch.save(self.network().state_dict(), policy_data_path)

    def is_recurrent(self) -> bool:
        """Return whether the policy expects recurrent (RNN-style) training."""
        return False


class PolicySpec(BaseModel):
    """Specification for a policy used during evaluation."""

    policy_class_path: str
    policy_data_path: Optional[str]
    proportion: float

    @property
    def name(self) -> str:
        parts = [self.policy_class_path.split(".")[-1]]
        if self.policy_data_path:
            parts.append(Path(self.policy_data_path).name)
        return "-".join(parts)
