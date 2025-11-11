import ctypes
import os
import sys
from typing import Callable

import numpy as np

from mettagrid.mettagrid_c import dtype_actions, dtype_observations
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation, Simulation

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "bindings/generated"))
import fast_agents as fa  # noqa: E402


class _FastAgentPolicy(AgentPolicy):
    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        agent_id: int,
        ctor: Callable[[int, str], fa.RandomAgent],
    ):
        super().__init__(policy_env_info)
        self.uses_raw_numpy = True
        self._agent_id = agent_id
        self._agent = ctor(agent_id, policy_env_info.to_json())
        self._num_agents = policy_env_info.num_agents
        self._num_tokens, self._token_dim = policy_env_info.observation_space.shape
        self._obs_dtype = dtype_observations
        self._action_names = [action.name for action in policy_env_info.actions.actions()]

    def step(self, obs: AgentObservation) -> Action:
        raw_observations = np.zeros(
            (self._num_agents, self._num_tokens, self._token_dim),
            dtype=self._obs_dtype,
        )
        raw_actions = np.zeros(self._num_agents, dtype=dtype_actions)

        self._encode_observation(obs, raw_observations[self._agent_id])
        self.step_batch(
            agent_id=self._agent_id,
            simulation=None,
            raw_observations=raw_observations,
            raw_actions=raw_actions,
        )
        action_idx = int(raw_actions[self._agent_id])
        return Action(name=self._action_names[action_idx])

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
            num_agents=int(raw_observations.shape[0]),
            num_tokens=int(raw_observations.shape[1]),
            size_token=int(raw_observations.shape[2]),
            raw_observations=ctypes.c_void_p(raw_observations.ctypes.data),
            num_actions=int(raw_actions.shape[0]),
            raw_actions=ctypes.c_void_p(raw_actions.ctypes.data),
        )
        return None

    def reset(self, simulation: Simulation = None) -> None:
        del simulation
        self._agent.reset()

    def _encode_observation(self, obs: AgentObservation, buffer: np.ndarray) -> None:
        max_tokens = min(self._num_tokens, buffer.shape[0])
        token_dim = buffer.shape[1]
        for idx, token in enumerate(obs.tokens[:max_tokens]):
            raw = token.raw_token
            for col in range(min(token_dim, len(raw))):
                buffer[idx, col] = raw[col]
        if max_tokens and len(obs.tokens) < max_tokens:
            buffer[len(obs.tokens), 1] = 0xFF


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
