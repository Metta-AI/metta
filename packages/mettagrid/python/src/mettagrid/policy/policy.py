"""Base policy classes and interfaces."""

import ctypes
from abc import abstractmethod
from pathlib import Path
from typing import Generic, Optional, Sequence, Tuple, TypeVar

import numpy as np
import torch.nn as nn
from pydantic import BaseModel

from mettagrid.mettagrid_c import dtype_observations
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.policy.policy_registry import PolicyRegistryMeta
from mettagrid.simulator import Action, AgentObservation

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

    def reset(self) -> None:
        """Reset the policy state. Default implementation does nothing."""
        pass


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

    def reset(self) -> None:
        """Reset any policy state; default no-op."""
        pass

    def step_batch(self, raw_observations: np.ndarray, raw_actions: np.ndarray) -> None:
        """Optional fast-path for policies that consume raw buffers.

        Policies that support raw NumPy pointers should override this method.
        The default implementation raises so callers get a clear error if a
        policy without batch support is used in a context that requires it."""

        raise NotImplementedError(f"{self.__class__.__name__} does not implement step_batch.")


class NimMultiAgentPolicy(MultiAgentPolicy):
    """Base class for Nim-backed multi-agent policies."""

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        handle_ctor,
        step_batch_name: str,
        step_single_name: str,
        agent_ids: Sequence[int] | None = None,
        reset_name: str | None = None,
    ) -> None:
        super().__init__(policy_env_info)
        self._handle = handle_ctor(policy_env_info.to_json())
        self._step_batch = getattr(self._handle, step_batch_name)
        self._handle_reset = getattr(self._handle, reset_name) if reset_name else None
        self._num_agents = policy_env_info.num_agents
        self._action_names = policy_env_info.action_names
        self._num_actions = len(self._action_names)
        obs_shape = policy_env_info.observation_space.shape
        self._num_tokens = obs_shape[0]
        self._token_dim = obs_shape[1]
        ids = list(agent_ids) if agent_ids is not None else list(range(self._num_agents))
        if not ids:
            raise ValueError("agent_ids must not be empty")
        self._agent_ids = set(ids)
        subset = np.array(ids, dtype=np.int32)
        self._default_subset = subset
        self._default_subset_ptr = subset.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)) if subset.size > 0 else None
        self._default_subset_len = subset.size
        self._step_single = getattr(self._handle, step_single_name)
        self._single_obs = np.empty(obs_shape, dtype=dtype_observations)
        self._single_action = np.zeros(1, dtype=np.int32)

    def step_batch(self, raw_observations: np.ndarray, raw_actions: np.ndarray) -> None:
        self._step_batch(
            self._default_subset_ptr,
            self._default_subset_len,
            self._num_agents,
            self._num_tokens,
            self._token_dim,
            raw_observations.ctypes.data,
            self._num_actions,
            raw_actions.ctypes.data,
        )

    def step_single(self, agent_id: int, obs: AgentObservation) -> int:
        if agent_id not in self._agent_ids:
            raise ValueError(f"Agent id {agent_id} not handled by {self.__class__.__name__}")
        target = self._single_obs
        target.fill(255)
        for idx, token in enumerate(obs.tokens):
            if idx >= self._num_tokens:
                break
            token_values = token.raw_token
            target[idx, : len(token_values)] = token_values
        self._single_action.fill(0)
        self._step_single(
            agent_id,
            self._num_agents,
            self._num_tokens,
            self._token_dim,
            target.ctypes.data,
            self._num_actions,
            self._single_action.ctypes.data,
        )
        return int(self._single_action[0])

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        if agent_id not in self._agent_ids:
            raise ValueError(f"Agent id {agent_id} not handled by {self.__class__.__name__}")
        return _NimAgentPolicy(self, agent_id)

    def reset(self) -> None:
        if self._handle_reset is not None:
            self._handle_reset()


class _NimAgentPolicy(AgentPolicy):
    """Lightweight proxy that delegates to the shared Nim multi-policy."""

    def __init__(self, parent: NimMultiAgentPolicy, agent_id: int):
        super().__init__(parent.policy_env_info)
        self._parent = parent
        self._agent_id = agent_id

    def step(self, obs: AgentObservation) -> Action:
        action_index = self._parent.step_single(self._agent_id, obs)
        return Action(name=self._parent._action_names[action_index])

    def reset(self) -> None:
        self._parent.reset()


class StatefulAgentPolicy(AgentPolicy, Generic[StateType]):
    """AgentPolicy wrapper that manages internal state (e.g., for RNNs).

    This wraps a stateful policy implementation and maintains the state
    across step() calls, providing a stateless AgentPolicy interface.

    StateType can be any type representing the agent's internal state.
    For example, Tuple[torch.Tensor, torch.Tensor] for LSTM hidden states.
    """

    def __init__(self, base_policy: "StatefulPolicyImpl[StateType]", policy_env_info: PolicyEnvInterface):
        """Initialize stateful wrapper.

        Args:
            base_policy: The underlying stateful policy implementation
            policy_env_info: The policy environment information
        """
        super().__init__(policy_env_info)
        self._base_policy = base_policy
        self._state: Optional[StateType] = None

    def step(self, obs: AgentObservation) -> Action:
        """Get action and update hidden state."""
        assert self._state is not None, "reset() must be called before step()"
        action, self._state = self._base_policy.step_with_state(obs, self._state)
        return action

    def reset(self) -> None:
        """Reset the hidden state to initial state."""
        self._base_policy.reset()
        self._state = self._base_policy.initial_agent_state()


class StatefulPolicyImpl(Generic[StateType]):
    """Base class for stateful policy implementations.

    This is used internally by policies that need to manage state.
    It provides step_with_state() which returns both action and new state,
    and initial_agent_state() which returns the initial state for a new agent.
    """

    def reset(self) -> None:
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
