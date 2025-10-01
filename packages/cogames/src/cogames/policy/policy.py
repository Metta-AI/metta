"""Base policy classes and interfaces."""

from abc import abstractmethod
from typing import Generic, Optional, Tuple, TypeVar

import torch.nn as nn

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
        """Return the action for the provided observation."""
        raise NotImplementedError("Subclasses must implement step()")

    def reset(self) -> None:
        """Reset any per-agent state (no-op by default)."""
        # Default: no-op for stateless policies
        return None


class Policy:
    """Abstract base class for multi-agent policies.

    A Policy manages creating AgentPolicy instances for multiple agents.
    This is the class users instantiate and pass to training/play functions.
    Training uses the Policy directly, while play calls agent_policy() to
    get per-agent instances.
    """

    @abstractmethod
    def agent_policy(self, agent_id: int) -> AgentPolicy:
        """Return the per-agent policy for the requested agent id."""
        ...

    def load_policy_data(self, policy_data_path: str) -> None:
        """Load policy parameters from ``policy_data_path``."""
        pass  # Default: no-op for policies without learnable parameters

    def save_policy_data(self, policy_data_path: str) -> None:
        """Persist policy parameters to ``policy_data_path``."""
        pass  # Default: no-op for policies without learnable parameters


class StatefulAgentPolicy(AgentPolicy, Generic[StateType]):
    """AgentPolicy wrapper that manages internal state (e.g., for RNNs).

    This wraps a stateful policy implementation and maintains the state
    across step() calls, providing a stateless AgentPolicy interface.

    StateType can be any type representing the agent's internal state.
    For example, Tuple[torch.Tensor, torch.Tensor] for LSTM hidden states.
    """

    def __init__(self, base_policy: "StatefulPolicyImpl[StateType]", agent_id: int):
        """Wrap a stateful policy implementation for a specific agent id."""
        self._base_policy = base_policy
        self._agent_id = agent_id
        # Initialize state using the base policy's agent_state() method
        self._state: Optional[StateType] = self._base_policy.agent_state()

    def step(self, obs: MettaGridObservation) -> MettaGridAction:
        """Get action and update hidden state."""
        action, self._state = self._base_policy.step_with_state(obs, self._state)
        return action

    def reset(self) -> None:
        """Reset the hidden state to initial state."""
        self._state = self._base_policy.agent_state()


class StatefulPolicyImpl(Generic[StateType]):
    """Base class for stateful policy implementations.

    This is used internally by policies that need to manage state.
    It provides step_with_state() which returns both action and new state,
    and agent_state() which returns the initial state for a new agent.
    """

    @abstractmethod
    def agent_state(self) -> Optional[StateType]:
        """Return the initial state for a new agent, or ``None`` for stateless policies."""
        ...

    def step_with_state(
        self, obs: MettaGridObservation, state: Optional[StateType]
    ) -> Tuple[MettaGridAction, Optional[StateType]]:
        """Return the action and updated state for the provided observation/state."""
        raise NotImplementedError


class TrainablePolicy(Policy):
    """Abstract base class for trainable policies.

    TrainablePolicy extends Policy and manages a neural network that can be trained.
    It creates per-agent AgentPolicy instances that share the same network.
    """

    @abstractmethod
    def network(self) -> nn.Module:
        """Get the underlying neural network for training."""
        ...

    @abstractmethod
    def agent_policy(self, agent_id: int) -> AgentPolicy:
        """Return the per-agent policy for the requested agent id."""
        ...

    def load_policy_data(self, policy_data_path: str) -> None:
        """Load network weights from ``policy_data_path`` using ``torch.load``."""
        import torch

        self.network().load_state_dict(torch.load(policy_data_path, map_location="cpu"))

    def save_policy_data(self, policy_data_path: str) -> None:
        """Save network weights to ``policy_data_path`` using ``torch.save``."""
        import torch

        torch.save(self.network().state_dict(), policy_data_path)
