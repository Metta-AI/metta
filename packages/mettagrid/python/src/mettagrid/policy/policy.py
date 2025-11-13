"""Base policy classes and interfaces."""

from abc import abstractmethod
from pathlib import Path
from typing import Generic, Optional, Tuple, TypeVar

import torch.nn as nn
from pydantic import BaseModel

from mettagrid.mettagrid_c import dtype_actions
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.policy.policy_registry import PolicyRegistryMeta
from mettagrid.simulator import Action, AgentObservation, Simulation

# Type variable for agent state - can be any type
StateType = TypeVar("StateType")


class AgentPolicy:
    """Base class for per-agent policies.

    AgentPolicy represents the interface for controlling a single agent.
    It provides the step() method that produces actions from observations.
    This is what play.py and evaluation code use directly.
    """

    def __init__(self, policy_env_info: PolicyEnvInterface):
        self._policy_env_info = policy_env_info

    @property
    def policy_env_info(self) -> PolicyEnvInterface:
        return self._policy_env_info

    def step(self, obs: AgentObservation) -> Action:
        """Get action given an observation.

        Args:
            obs: The observation for this agent

        Returns:
            The action to take
        """
        raise NotImplementedError("Subclasses must implement step()")

    def reset(self, simulation: Optional[Simulation] = None) -> None:
        """Reset the policy state. Default implementation does nothing."""
        pass

    def step_batch(self, _raw_observations, raw_actions) -> None:
        """Optional fast-path for policies that consume raw buffers.

        Policies that support raw NumPy pointers should override this method.
        The default implementation raises so callers get a clear error if a
        policy without batch support is used in a context that requires it."""

        raise NotImplementedError(f"{self.__class__.__name__} does not implement step_batch.")


class MultiAgentPolicy(metaclass=PolicyRegistryMeta):
    """Abstract base class for multi-agent policies.

    A Policy manages creating AgentPolicy instances for multiple agents.
    This is the class users instantiate and pass to training/play functions.
    Training uses the Policy directly, while play calls agent_policy() to
    get per-agent instances.

    Subclasses can register themselves by defining:
    - short_names: list[str] = ["name1", "name2"] for one or more aliases
    """

    short_names: list[str] | None = None

    def __init__(self, policy_env_info: PolicyEnvInterface):
        self._policy_env_info = policy_env_info
        self._actions = policy_env_info.actions

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

    @property
    def policy_env_info(self) -> PolicyEnvInterface:
        return self._policy_env_info


class StatefulAgentPolicy(AgentPolicy, Generic[StateType]):
    """AgentPolicy wrapper that manages internal state (e.g., for RNNs).

    This wraps a stateful policy implementation and maintains the state
    across step() calls, providing a stateless AgentPolicy interface.

    StateType can be any type representing the agent's internal state.
    For example, Tuple[torch.Tensor, torch.Tensor] for LSTM hidden states.
    """

    def __init__(
        self,
        base_policy: "StatefulPolicyImpl[StateType]",
        policy_env_info: PolicyEnvInterface,
        agent_id: Optional[int] = None,
    ):
        """Initialize stateful wrapper.

        Args:
            base_policy: The underlying stateful policy implementation
            policy_env_info: The policy environment information
        """
        super().__init__(policy_env_info)
        self._base_policy = base_policy
        self._state: Optional[StateType] = None
        self._agent_id = agent_id
        self._agent_states: dict[int, StateType] = {}
        self._action_name_to_index = {name: idx for idx, name in enumerate(policy_env_info.action_names)}
        self._simulation: Simulation | None = None

    def step(self, obs: AgentObservation) -> Action:
        """Get action and update hidden state."""
        assert self._state is not None, "reset() must be called before step()"
        action, self._state = self._base_policy.step_with_state(obs, self._state)
        if self._agent_id is not None:
            self._agent_states[self._agent_id] = self._state
        return action

    def reset(self, simulation: Optional[Simulation] = None) -> None:
        """Reset the hidden state to initial state."""
        self._base_policy.reset(simulation)
        self._simulation = simulation
        self._state = self._base_policy.initial_agent_state()
        self._agent_states.clear()
        if self._agent_id is not None:
            self._agent_states[self._agent_id] = self._state

    def step_batch(self, _raw_observations, raw_actions) -> None:
        sim = self._simulation
        assert sim is not None, "reset() must be called before step_batch()"

        for agent_idx, obs in enumerate(sim.observations()):
            state = self._agent_states.get(agent_idx) or self._base_policy.initial_agent_state()
            action, new_state = self._base_policy.step_with_state(obs, state)
            self._agent_states[agent_idx] = new_state
            assert isinstance(action, Action), "Policies must return mettagrid.simulator.Action instances"
            raw_actions[agent_idx] = dtype_actions.type(self._action_name_to_index[action.name])

        if self._agent_id is not None and self._agent_id in self._agent_states:
            self._state = self._agent_states[self._agent_id]


class StatefulPolicyImpl(Generic[StateType]):
    """Base class for stateful policy implementations.

    This is used internally by policies that need to manage state.
    It provides step_with_state() which returns both action and new state,
    and initial_agent_state() which returns the initial state for a new agent.
    """

    def reset(self, simulation: Optional[Simulation]) -> None:
        """Reset the policy."""
        pass

    @abstractmethod
    def initial_agent_state(self) -> StateType:
        """Get the initial state for a new agent.

        Returns:
            Initial state for the agent. For LSTMs, this returns zero-initialized hidden/cell states.
        """
        ...

    def step_with_state(self, obs: AgentObservation, state: StateType) -> Tuple[Action, StateType]:
        """Get action and potentially update state.

        Args:
            obs: The observation
            state: Current hidden state (e.g., RNN hidden state)

        Returns:
            Tuple of (action, new_state)
        """
        raise NotImplementedError


class TrainablePolicy(MultiAgentPolicy):
    """Abstract base class for trainable policies.

    TrainablePolicy extends Policy and manages a neural network that can be trained.
    It creates per-agent AgentPolicy instances that share the same network.
    """

    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)

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
