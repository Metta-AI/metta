import importlib
import os
import sys
from typing import Any, Optional

import numpy as np

from mettagrid.mettagrid_c import dtype_observations
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation, Simulation

current_dir = os.path.dirname(os.path.abspath(__file__))
bindings_dir = os.path.join(current_dir, "bindings/generated")
if bindings_dir not in sys.path:
    sys.path.append(bindings_dir)

fa = importlib.import_module("fast_agents")


class _FastAgentPolicyBase(AgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int):
        super().__init__(policy_env_info)
        obs_shape = policy_env_info.observation_space.shape
        self._agent_id = agent_id
        self._num_agents = policy_env_info.num_agents
        self._num_tokens = obs_shape[0]
        self._token_dim = obs_shape[1]
        self._batch_obs = np.empty((self._num_agents, *obs_shape), dtype=dtype_observations)
        self._batch_actions = np.zeros(self._num_agents, dtype=np.int32)
        self._action_names = policy_env_info.action_names
        self._agent: Any | None = None

    def _pack_raw_observation(self, target: np.ndarray, obs: AgentObservation) -> None:
        target.fill(255)
        for idx, token in enumerate(obs.tokens):
            if idx >= self._num_tokens:
                break
            token_values = token.raw_token
            target[idx, : len(token_values)] = token_values

    def _require_agent(self) -> Any:
        if self._agent is None:
            raise RuntimeError(f"{self.__class__.__name__} tried to act before the native agent was initialized")
        return self._agent

    def step_batch(self, raw_observations: np.ndarray, raw_actions: np.ndarray) -> None:
        agent = self._require_agent()
        agent.step(
            num_agents=raw_observations.shape[0],
            num_tokens=raw_observations.shape[1],
            size_token=raw_observations.shape[2],
            raw_observations=raw_observations.ctypes.data,
            num_actions=raw_actions.shape[0],
            raw_actions=raw_actions.ctypes.data,
        )

    def step(self, obs: AgentObservation) -> Action:
        self._batch_obs.fill(255)
        self._pack_raw_observation(self._batch_obs[self._agent_id], obs)
        self._batch_actions.fill(0)
        agent = self._require_agent()
        agent.step(
            num_agents=self._num_agents,
            num_tokens=self._num_tokens,
            size_token=self._token_dim,
            raw_observations=self._batch_obs.ctypes.data,
            num_actions=self._num_agents,
            raw_actions=self._batch_actions.ctypes.data,
        )
        action_index = int(self._batch_actions[self._agent_id])
        return Action(name=self._action_names[action_index])

    def reset(self, simulation: Optional[Simulation] = None) -> None:
        agent = self._require_agent()
        agent.reset()


class RandomAgentPolicy(_FastAgentPolicyBase):
    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int):
        super().__init__(policy_env_info, agent_id)
        self._agent = fa.RandomAgent(agent_id, policy_env_info.to_json())


class RandomAgentsMultiPolicy(MultiAgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)

    short_names = ["fast_random"]

    def agent_policy(self, agent_id: int) -> RandomAgentPolicy:
        return RandomAgentPolicy(self._policy_env_info, agent_id)


class ThinkyAgentPolicy(_FastAgentPolicyBase):
    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int):
        super().__init__(policy_env_info, agent_id)
        self._agent = fa.ThinkyAgent(agent_id, policy_env_info.to_json())


class ThinkyAgentsMultiPolicy(MultiAgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)

    short_names = ["fast_thinky"]

    def agent_policy(self, agent_id: int) -> ThinkyAgentPolicy:
        return ThinkyAgentPolicy(self._policy_env_info, agent_id)


class RaceCarAgentPolicy(_FastAgentPolicyBase):
    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int):
        super().__init__(policy_env_info, agent_id)
        self._agent = fa.RaceCarAgent(agent_id, policy_env_info.to_json())


class RaceCarAgentsMultiPolicy(MultiAgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)

    short_names = ["fast_race_car"]

    def agent_policy(self, agent_id: int) -> RaceCarAgentPolicy:
        return RaceCarAgentPolicy(self._policy_env_info, agent_id)
