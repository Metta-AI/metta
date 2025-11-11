"""Base policy classes and interfaces."""

from abc import abstractmethod
from pathlib import Path
from typing import Generic, Optional, Tuple, TypeVar

import numpy as np
import torch.nn as nn
from pydantic import BaseModel

from mettagrid.policy.policy_env_interface import PolicyEnvInterface
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
        self._uses_raw_numpy = False

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

    def step_batch(
        self,
        *,
        agent_id: int,
        simulation: Optional[Simulation],
        raw_observations: np.ndarray,
        raw_actions: np.ndarray,
    ) -> Action | None:
        """Optional batch-friendly hook.

        Default behavior simply defers to ``step`` using the agent's observation.
        Policies that operate directly on the raw simulator buffers can override
        this to avoid per-agent object construction.
        """

        del raw_observations, raw_actions
        if simulation is None:
            raise ValueError("simulation is required for AgentPolicy.step_batch default implementation")
        observation = simulation.agent(agent_id).observation
        return self.step(observation)

    @property
    def uses_raw_numpy(self) -> bool:
        return self._uses_raw_numpy

    @uses_raw_numpy.setter
    def uses_raw_numpy(self, value: bool) -> None:
        self._uses_raw_numpy = value


class MultiAgentPolicy:
    """Abstract base class for multi-agent policies.

    A Policy manages creating AgentPolicy instances for multiple agents.
    This is the class users instantiate and pass to training/play functions.
    Training uses the Policy directly, while play calls agent_policy() to
    get per-agent instances.
    """

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

    def step_batch(
        self,
        simulation: Simulation,
        out_actions: Optional[np.ndarray] = None,
        observations: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute batched actions for all agents.

        Subclasses can override for custom behavior. By default this loops over
        agents, calls their individual ``step`` (via AgentPolicy.step_batch),
        and writes the resulting action indices into ``out_actions``.
        """

        del observations
        num_agents = simulation.num_agents
        if out_actions is None:
            out_actions = simulation.raw_actions()
        raw_observations = simulation.raw_observations()
        raw_actions = out_actions
        action_ids = simulation.action_ids

        for agent_id in range(num_agents):
            policy = self.agent_policy(agent_id)
            maybe_action = policy.step_batch(
                agent_id=agent_id,
                simulation=simulation,
                raw_observations=raw_observations,
                raw_actions=raw_actions,
            )
            if maybe_action is None:
                continue
            out_actions[agent_id] = action_ids[maybe_action.name]

        return out_actions


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

    def reset(self, simulation: Optional[Simulation] = None) -> None:
        """Reset the hidden state to initial state."""
        self._base_policy.reset(simulation)
        self._state = self._base_policy.initial_agent_state(simulation)

    def step_batch(
        self,
        *,
        agent_id: int,
        simulation: Optional[Simulation],
        raw_observations: np.ndarray,
        raw_actions: np.ndarray,
    ) -> Action | None:
        del raw_observations, raw_actions
        if simulation is None:
            raise ValueError("simulation is required for StatefulAgentPolicy.step_batch")
        observation = simulation.agent(agent_id).observation
        return self.step(observation)


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
    def initial_agent_state(self, simulation: Optional[Simulation]) -> StateType:
        """Get the initial state for a new agent in a simulation.

        Args:
            simulation: The simulation to reset the policy state for

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
