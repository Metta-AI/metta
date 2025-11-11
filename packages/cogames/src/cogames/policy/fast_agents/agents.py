from __future__ import annotations

import os
import sys

import numpy as np

from mettagrid.mettagrid_c import dtype_actions, dtype_observations
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation, Simulation

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "bindings/generated"))
import fast_agents as fa  # noqa: E402

SENTINEL_FEATURE_ID = 0xFF


class _FastAgentPolicyBase(AgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int):
        super().__init__(policy_env_info)
        self.uses_raw_numpy = True
        self._agent_id = agent_id
        self._num_agents = policy_env_info.num_agents
        self._obs_shape = policy_env_info.observation_space.shape
        self._action_names = [action.name for action in policy_env_info.actions.actions()]

    def _allocate_observation_buffer(self) -> np.ndarray:
        return np.zeros((self._num_agents, *self._obs_shape), dtype=dtype_observations)

    def _allocate_action_buffer(self) -> np.ndarray:
        return np.zeros(self._num_agents, dtype=dtype_actions)

    def _pack_observation(self, obs: AgentObservation) -> np.ndarray:
        raw_obs = self._allocate_observation_buffer()
        agent_tokens = raw_obs[self._agent_id]
        max_tokens = self._obs_shape[0]
        token_width = self._obs_shape[1]
        for idx, token in enumerate(obs.tokens[:max_tokens]):
            raw_token = token.raw_token
            for col in range(min(token_width, len(raw_token))):
                agent_tokens[idx, col] = raw_token[col]
        sentinel_row = len(obs.tokens)
        if max_tokens and sentinel_row < max_tokens:
            agent_tokens[sentinel_row, 1] = SENTINEL_FEATURE_ID
        return raw_obs

    def step(self, obs: AgentObservation) -> Action:
        raw_obs = self._pack_observation(obs)
        raw_actions = self._allocate_action_buffer()
        self.step_batch(
            agent_id=self._agent_id,
            simulation=None,
            raw_observations=raw_obs,
            raw_actions=raw_actions,
        )
        action_idx = int(raw_actions[self._agent_id])
        return Action(name=self._action_names[action_idx])


class RandomAgentPolicy(_FastAgentPolicyBase):
    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int):
        super().__init__(policy_env_info, agent_id)
        self._agent = fa.RandomAgent(agent_id, policy_env_info.to_json())

    def step_batch(
        self,
        *,
        agent_id: int,
        simulation: Simulation | None,
        raw_observations: np.ndarray,
        raw_actions: np.ndarray,
    ) -> Action | None:
        del simulation
        assert agent_id == self._agent_id, "Mismatched agent id for fast agent"
        self._agent.step(
            num_agents=raw_observations.shape[0],
            num_tokens=raw_observations.shape[1],
            size_token=raw_observations.shape[2],
            raw_observations=raw_observations.ctypes.data,
            num_actions=raw_actions.shape[0],
            raw_actions=raw_actions.ctypes.data,
        )
        return None

    def reset(self, simulation: Simulation = None) -> None:
        self._agent.reset()


class RandomAgentsMultiPolicy(MultiAgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)

    def agent_policy(self, agent_id: int) -> RandomAgentPolicy:
        return RandomAgentPolicy(self._policy_env_info, agent_id)


class ThinkyAgentPolicy(_FastAgentPolicyBase):
    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int):
        super().__init__(policy_env_info, agent_id)
        self._agent = fa.ThinkyAgent(agent_id, policy_env_info.to_json())

    def step_batch(
        self,
        *,
        agent_id: int,
        simulation: Simulation | None,
        raw_observations: np.ndarray,
        raw_actions: np.ndarray,
    ) -> Action | None:
        del simulation
        assert agent_id == self._agent_id, "Mismatched agent id for fast agent"
        self._agent.step(
            num_agents=raw_observations.shape[0],
            num_tokens=raw_observations.shape[1],
            size_token=raw_observations.shape[2],
            raw_observations=raw_observations.ctypes.data,
            num_actions=raw_actions.shape[0],
            raw_actions=raw_actions.ctypes.data,
        )
        return None

    def reset(self, simulation: Simulation = None) -> None:
        self._agent.reset()


class ThinkyAgentsMultiPolicy(MultiAgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)

    def agent_policy(self, agent_id: int) -> ThinkyAgentPolicy:
        return ThinkyAgentPolicy(self._policy_env_info, agent_id)


class RaceCarAgentPolicy(_FastAgentPolicyBase):
    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int):
        super().__init__(policy_env_info, agent_id)
        self._agent = fa.RaceCarAgent(agent_id, policy_env_info.to_json())

    def step_batch(
        self,
        *,
        agent_id: int,
        simulation: Simulation | None,
        raw_observations: np.ndarray,
        raw_actions: np.ndarray,
    ) -> Action | None:
        del simulation
        assert agent_id == self._agent_id, "Mismatched agent id for fast agent"
        self._agent.step(
            num_agents=raw_observations.shape[0],
            num_tokens=raw_observations.shape[1],
            size_token=raw_observations.shape[2],
            raw_observations=raw_observations.ctypes.data,
            num_actions=raw_actions.shape[0],
            raw_actions=raw_actions.ctypes.data,
        )
        return None

    def reset(self, simulation: Simulation = None) -> None:
        self._agent.reset()


class RaceCarAgentsMultiPolicy(MultiAgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)

    def agent_policy(self, agent_id: int) -> RaceCarAgentPolicy:
        return RaceCarAgentPolicy(self._policy_env_info, agent_id)
