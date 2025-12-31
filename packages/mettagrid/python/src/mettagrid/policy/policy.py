"""Base policy classes and interfaces."""

import ctypes
from abc import abstractmethod
from pathlib import Path
from typing import Any, Generic, Optional, Sequence, Tuple, TypeVar, cast

import numpy as np
import torch.nn as nn
from pydantic import BaseModel, Field

from mettagrid.mettagrid_c import dtype_observations
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


class MultiAgentPolicy(metaclass=PolicyRegistryMeta):
    """Unified policy interface for multi-agent systems.

    This class manages policy lifecycle including:
    1. Creating per-agent policy instances (agent_policy)
    2. Serialization/deserialization (load_policy_data/save_policy_data)
    3. Optional: Providing network for training (network)
    4. Optional: Batch stepping optimization (step_batch)

    Subclasses can register themselves by defining:
    - short_names: list[str] = ["name1", "name2"] for one or more aliases
    """

    short_names: list[str] | None = None

    def __init__(self, policy_env_info: PolicyEnvInterface, device: str = "cpu", **kwargs: Any):
        self._policy_env_info = policy_env_info
        self._actions = policy_env_info.actions

    @abstractmethod
    def agent_policy(self, agent_id: int) -> AgentPolicy:
        """Get an AgentPolicy instance for a specific agent."""
        ...

    def load_policy_data(self, policy_data_path: str) -> None:
        """Load policy data from a file.

        Default implementation does nothing. Override to load weights/parameters.
        For trainable policies, override to load torch state_dict.
        """
        pass

    def save_policy_data(self, policy_data_path: str) -> None:
        """Save policy data to a file.

        Default implementation does nothing. Override to save weights/parameters.
        For trainable policies, override to save torch state_dict.
        """
        pass

    def network(self) -> Optional[nn.Module]:
        """Get the underlying neural network for training.

        Returns None if this policy is not trainable.
        Override this method in trainable policies to return the network.
        """
        return None

    @property
    def policy_env_info(self) -> PolicyEnvInterface:
        return self._policy_env_info

    def reset(self) -> None:
        """Reset any policy state; default no-op."""
        pass

    def step_batch(self, raw_observations: np.ndarray, raw_actions: np.ndarray) -> None:
        """Optional fast-path for policies that consume raw buffers.

        Override this method in policies that support batch stepping.
        The default implementation raises NotImplementedError.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement step_batch.")


class NimMultiAgentPolicy(MultiAgentPolicy):
    """Base class for Nim-backed multi-agent policies."""

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        nim_policy_factory,
        agent_ids: Sequence[int] | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(policy_env_info, device=device)
        self._nim_policy = nim_policy_factory(policy_env_info.to_json())
        self._num_agents = policy_env_info.num_agents
        obs_shape = policy_env_info.observation_space.shape
        self._num_tokens = obs_shape[0]
        self._token_dim = obs_shape[1]
        subset = np.array(list(agent_ids) if agent_ids is not None else range(self._num_agents), dtype=np.int32)
        self._default_subset = subset
        self._default_subset_len = subset.size
        self._default_subset_ptr = subset.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)) if subset.size > 0 else None
        self._single_agent_id = np.zeros(1, dtype=np.int32)
        self._full_obs_buffer = np.full(
            (self._num_agents, self._num_tokens, self._token_dim),
            fill_value=255,
            dtype=dtype_observations,
        )
        self._full_action_buffer = np.zeros(self._num_agents, dtype=np.int32)

    def step_batch(self, raw_observations: np.ndarray, raw_actions: np.ndarray) -> None:
        self._invoke_step_batch(self._default_subset, raw_observations, raw_actions)

    def step_single(self, agent_id: int, obs: AgentObservation) -> int:
        self._single_agent_id[0] = agent_id
        row = self._full_obs_buffer[agent_id]
        row.fill(255)
        for idx, token in enumerate(obs.tokens):
            if idx >= self._num_tokens:
                break
            token_values = token.raw_token
            row[idx, : len(token_values)] = token_values
        self._full_action_buffer[agent_id] = 0
        self._invoke_step_batch(self._single_agent_id, self._full_obs_buffer, self._full_action_buffer)
        return int(self._full_action_buffer[agent_id])

    def agent_policy(self, agent_id: int) -> AgentPolicy:
        return _NimAgentPolicy(self, agent_id)

    def _invoke_step_batch(
        self,
        agent_ids: np.ndarray,
        raw_observations: np.ndarray,
        raw_actions: np.ndarray,
    ) -> None:
        subset_len = agent_ids.shape[0]

        if raw_observations.shape[0] == self._num_agents:
            obs_buffer = raw_observations
        else:
            obs_buffer = self._full_obs_buffer
            obs_buffer[agent_ids] = raw_observations

        if raw_actions.shape[0] == self._num_agents:
            action_buffer = raw_actions
            needs_scatter = False
        else:
            action_buffer = self._full_action_buffer
            action_buffer[agent_ids] = 0
            needs_scatter = True

        agent_ids_ptr = (
            self._default_subset_ptr
            if agent_ids is self._default_subset
            else agent_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        )
        self._nim_policy.step_batch(
            agent_ids_ptr,
            subset_len,
            self._num_agents,
            self._num_tokens,
            self._token_dim,
            obs_buffer.ctypes.data,
            len(self.policy_env_info.action_names),
            action_buffer.ctypes.data,
        )

        if needs_scatter:
            raw_actions[...] = action_buffer[agent_ids]


class _NimAgentPolicy(AgentPolicy):
    """Lightweight proxy that delegates to the shared Nim multi-policy."""

    def __init__(self, parent: NimMultiAgentPolicy, agent_id: int):
        super().__init__(parent.policy_env_info)
        self._parent = parent
        self._agent_id = agent_id

    def step(self, obs: AgentObservation) -> Action:
        action_index = self._parent.step_single(self._agent_id, obs)
        return Action(name=self.policy_env_info.action_names[action_index])


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
        self._state_initialized = False

    def step(self, obs: AgentObservation) -> Action:
        """Get action and update hidden state."""
        if not self._state_initialized:
            self._initialize_state(self._simulation)
        if hasattr(self._base_policy, "set_active_agent"):
            self._base_policy.set_active_agent(self._agent_id)
        state = cast(StateType, self._state)
        action, self._state = self._base_policy.step_with_state(obs, state)
        if self._agent_id is not None:
            self._agent_states[self._agent_id] = self._state
        return action

    def reset(self, simulation: Optional[Simulation] = None) -> None:
        """Reset the hidden state to initial state."""
        self._initialize_state(simulation)

    def _initialize_state(self, simulation: Optional[Simulation]) -> None:
        self._simulation = simulation
        self._base_policy.reset()
        self._state = self._base_policy.initial_agent_state()
        self._agent_states.clear()
        self._state_initialized = True
        if self._agent_id is not None:
            self._agent_states[self._agent_id] = self._state


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

    def set_active_agent(self, agent_id: Optional[int]) -> None:
        """Optional hook for implementations that need the calling agent id."""
        _ = agent_id


class PolicySpec(BaseModel):
    """Specification for a policy used during evaluation."""

    class_path: str = Field(description="Local path to policy class, or shorthand")

    data_path: Optional[str] = Field(default=None, description="Local file path to policy weights, if applicable")

    init_kwargs: dict[str, Any] = Field(default_factory=dict)

    @property
    def name(self) -> str:
        parts = [
            self.class_path.split(".")[-1],
        ]
        if self.data_path:
            parts.append(Path(self.data_path).name)
        return "-".join(parts)
