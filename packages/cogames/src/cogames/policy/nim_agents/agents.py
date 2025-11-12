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

na = importlib.import_module("nim_agents")


class _NimAgentPolicyBase(AgentPolicy):
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


class RandomAgentPolicy(_NimAgentPolicyBase):
    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int):
        super().__init__(policy_env_info, agent_id)
        self._agent = na.RandomAgent(agent_id, policy_env_info.to_json())


class _NimAgentsMultiPolicyBase(MultiAgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface, policy_type: type[_NimAgentPolicyBase]):
        super().__init__(policy_env_info)
        self._policy_type = policy_type
        self._batch_agent: _NimAgentPolicyBase | None = None

    def agent_policy(self, agent_id: int) -> _NimAgentPolicyBase:
        return self._policy_type(self._policy_env_info, agent_id)

    def _require_batch_agent(self) -> _NimAgentPolicyBase:
        if self._batch_agent is None:
            self._batch_agent = self._policy_type(self._policy_env_info, agent_id=0)
        return self._batch_agent

    def step_batch(self, raw_observations: np.ndarray, raw_actions: np.ndarray) -> None:
        batch_agent = self._require_batch_agent()
        batch_agent.step_batch(raw_observations, raw_actions)


class RandomAgentsMultiPolicy(_NimAgentsMultiPolicyBase):
    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info, RandomAgentPolicy)

    short_names = ["fast_random"]


class ThinkyAgentPolicy(_NimAgentPolicyBase):
    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int):
        super().__init__(policy_env_info, agent_id)
        self._agent = na.ThinkyAgent(agent_id, policy_env_info.to_json())


class ThinkyAgentsMultiPolicy(_NimAgentsMultiPolicyBase):
    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info, ThinkyAgentPolicy)

    short_names = ["fast_thinky"]


class RaceCarAgentPolicy(_NimAgentPolicyBase):
    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int):
        super().__init__(policy_env_info, agent_id)
        self._agent = na.RaceCarAgent(agent_id, policy_env_info.to_json())


class RaceCarAgentsMultiPolicy(_NimAgentsMultiPolicyBase):
    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info, RaceCarAgentPolicy)

    short_names = ["fast_race_car"]
