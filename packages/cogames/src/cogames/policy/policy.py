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
        """Get action given an observation.

        Args:
            obs: The observation for this agent

        Returns:
            The action to take
        """
        raise NotImplementedError("Subclasses must implement step()")

    def reset(self) -> None:
        """Reset the policy state. Default implementation does nothing."""
        pass


class Policy:
    """Abstract base class for multi-agent policies.

    A Policy manages creating AgentPolicy instances for multiple agents.
    This is the class users instantiate and pass to training/play functions.
    Training uses the Policy directly, while play calls agent_policy() to
    get per-agent instances.
    """

    @abstractmethod
    def agent_policy(self, agent_id: int) -> AgentPolicy:
        """Get an AgentPolicy instance for a specific agent.

        Args:
            agent_id: The ID of the agent

        Returns:
            An AgentPolicy instance for this agent
        """
        ...

    def load_policy_data(self, policy_data_path: str) -> None:
        """Load policy data from a file.

        Args:
            policy_data_path: Path to the policy data file

        Default implementation does nothing. Override to load weights/parameters.
        """
        pass  # Default: no-op for policies without learnable parameters

    def save_policy_data(self, policy_data_path: str) -> None:
        """Save policy data to a file.

        Args:
            policy_data_path: Path to save the policy data

        Default implementation does nothing. Override to save weights/parameters.
        """
        pass  # Default: no-op for policies without learnable parameters


class StatefulAgentPolicy(AgentPolicy, Generic[StateType]):
    """AgentPolicy wrapper that manages internal state (e.g., for RNNs).

    This wraps a stateful policy implementation and maintains the state
    across step() calls, providing a stateless AgentPolicy interface.

    StateType can be any type representing the agent's internal state.
    For example, Tuple[torch.Tensor, torch.Tensor] for LSTM hidden states.
    """

    def __init__(self, base_policy: "StatefulPolicyImpl[StateType]", agent_id: int):
        """Initialize stateful wrapper.

        Args:
            base_policy: The underlying stateful policy implementation
            agent_id: The ID of the agent this policy is for
        """
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
        """Get the initial state for a new agent.

        Returns:
            Initial state for the agent, or None if no initial state needed.
            For LSTMs, this typically returns None (states are initialized by the network).
        """
        ...

    def step_with_state(
        self, obs: MettaGridObservation, state: Optional[StateType]
    ) -> Tuple[MettaGridAction, Optional[StateType]]:
        """Get action and potentially update state.

        Args:
            obs: The observation
            state: Current hidden state (e.g., RNN hidden state)

        Returns:
            Tuple of (action, new_state)
        """
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
        """Get an AgentPolicy instance for a specific agent.

        This must be overridden by trainable policies to return
        per-agent policy instances.

        Args:
            agent_id: The ID of the agent

        Returns:
            An AgentPolicy instance for this agent
        """
        ...

    def load_policy_data(self, policy_data_path: str) -> None:
        """Load network weights from file.

        Default implementation loads PyTorch state dict.
        """
        import torch

        self.network().load_state_dict(torch.load(policy_data_path, map_location="cpu"))

    def save_policy_data(self, policy_data_path: str) -> None:
        """Save network weights to file.

        Default implementation uses torch.save.
        """
        import torch

        torch.save(self.network().state_dict(), policy_data_path)


class PolicySpec(BaseModel):
    """Specification for a policy used during evaluation."""

    # Path to policy class, or shorthand
    policy_class_path: str

    # Path to policy weights, if applicable
    policy_data_path: Optional[str]

    # Proportion of total agents to assign to this policy
    proportion: float = 1.0

    @property
    def name(self) -> str:
        """Get the name of the policy."""
        parts = [
            self.policy_class_path.split(".")[-1],
        ]
        if self.policy_data_path:
            parts.append(Path(self.policy_data_path).name)
        return "-".join(parts)
