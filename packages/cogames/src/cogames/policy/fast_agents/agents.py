from __future__ import annotations

import os
import sys
from typing import Callable

import numpy as np

from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation, Simulation

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "bindings/generated"))
import fast_agents as fa  # noqa: E402


class _FastAgentPolicy(AgentPolicy):
    """Minimal shim that forwards raw simulator buffers to the Nim agents."""

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        agent_id: int,
        agent_ctor: Callable[[int, str], fa.RandomAgent],
    ):
        super().__init__(policy_env_info)
        self._agent_id = agent_id
        self._agent = agent_ctor(agent_id, policy_env_info.to_json())
        self.uses_raw_numpy = True

    def step(self, obs: AgentObservation) -> Action:
        raise NotImplementedError("Fast agents operate on raw simulator buffers; call step_batch via MultiAgentPolicy.")

    def step_batch(
        self,
        *,
        agent_id: int,
        simulation: Simulation | None,
        raw_observations: np.ndarray,
        raw_actions: np.ndarray,
    ) -> Action | None:
        del simulation
        if agent_id != self._agent_id:
            raise ValueError(f"Mismatched agent id: expected {self._agent_id}, got {agent_id}")

        self._agent.step(
            num_agents=raw_observations.shape[0],
            num_tokens=raw_observations.shape[1],
            size_token=raw_observations.shape[2],
            raw_observations=raw_observations.ctypes.data,
            num_actions=raw_actions.shape[0],
            raw_actions=raw_actions.ctypes.data,
        )
        return None

    def reset(self, simulation: Simulation | None = None) -> None:
        del simulation
        self._agent.reset()


class RandomAgentPolicy(_FastAgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int):
        super().__init__(policy_env_info, agent_id, fa.RandomAgent)


class RandomAgentsMultiPolicy(MultiAgentPolicy):
    def agent_policy(self, agent_id: int) -> RandomAgentPolicy:
        return RandomAgentPolicy(self._policy_env_info, agent_id)


class ThinkyAgentPolicy(_FastAgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int):
        super().__init__(policy_env_info, agent_id, fa.ThinkyAgent)


class ThinkyAgentsMultiPolicy(MultiAgentPolicy):
    def agent_policy(self, agent_id: int) -> ThinkyAgentPolicy:
        return ThinkyAgentPolicy(self._policy_env_info, agent_id)


class RaceCarAgentPolicy(_FastAgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int):
        super().__init__(policy_env_info, agent_id, fa.RaceCarAgent)


class RaceCarAgentsMultiPolicy(MultiAgentPolicy):
    def agent_policy(self, agent_id: int) -> RaceCarAgentPolicy:
        return RaceCarAgentPolicy(self._policy_env_info, agent_id)
