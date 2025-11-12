import ctypes
import importlib
import os
import sys
from typing import Sequence

import numpy as np

from mettagrid.mettagrid_c import dtype_observations
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation

current_dir = os.path.dirname(os.path.abspath(__file__))
bindings_dir = os.path.join(current_dir, "bindings/generated")
if bindings_dir not in sys.path:
    sys.path.append(bindings_dir)

na = importlib.import_module("nim_agents")


class _NimPolicyHandle:
    def __init__(
        self,
        handle_ctor,
        step_batch_name: str,
        policy_env_info: PolicyEnvInterface,
        agent_ids: Sequence[int],
    ):
        self._handle = handle_ctor(policy_env_info.to_json())
        self._step_batch = getattr(self._handle, step_batch_name)
        self._num_agents = policy_env_info.num_agents
        obs_shape = policy_env_info.observation_space.shape
        self._num_tokens = obs_shape[0]
        self._token_dim = obs_shape[1]
        subset = np.array(agent_ids, dtype=np.int32)
        self._default_subset = subset.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)) if subset.size > 0 else None
        self._default_subset_len = subset.size

    def step_batch(
        self,
        raw_observations: np.ndarray,
        raw_actions: np.ndarray,
        agent_subset: np.ndarray | None = None,
    ) -> None:
        if agent_subset is not None:
            subset_ptr = agent_subset.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            subset_len = agent_subset.size
        else:
            subset_ptr = self._default_subset
            subset_len = self._default_subset_len
        self._step_batch(
            subset_ptr,
            subset_len,
            self._num_agents,
            self._num_tokens,
            self._token_dim,
            raw_observations.ctypes.data,
            self._num_agents,
            raw_actions.ctypes.data,
        )


class ThinkyAgentPolicy(AgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int, handle: _NimPolicyHandle | None = None):
        super().__init__(policy_env_info)
        obs_shape = policy_env_info.observation_space.shape
        self._agent_id = agent_id
        self._num_agents = policy_env_info.num_agents
        self._num_tokens = obs_shape[0]
        self._token_dim = obs_shape[1]
        self._batch_obs = np.empty((self._num_agents, *obs_shape), dtype=dtype_observations)
        self._batch_actions = np.zeros(self._num_agents, dtype=np.int32)
        self._action_names = policy_env_info.action_names
        self._single_subset = np.array([agent_id], dtype=np.int32)
        self._handle = handle or _NimPolicyHandle(
            na.ThinkyPolicy,
            "thinky_policy_step_batch",
            policy_env_info,
            [agent_id],
        )

    def _pack_obs(self, target: np.ndarray, obs: AgentObservation) -> None:
        target.fill(255)
        for idx, token in enumerate(obs.tokens):
            if idx >= self._num_tokens:
                break
            target[idx, : len(token.raw_token)] = token.raw_token

    def step(self, obs: AgentObservation) -> Action:
        self._batch_obs.fill(255)
        self._pack_obs(self._batch_obs[self._agent_id], obs)
        self._batch_actions.fill(0)
        self._handle.step_batch(self._batch_obs, self._batch_actions, agent_subset=self._single_subset)
        action_index = int(self._batch_actions[self._agent_id])
        return Action(name=self._action_names[action_index])


class ThinkyAgentsMultiPolicy(MultiAgentPolicy):
    short_names = ["nim_thinky", "fast_thinky"]

    def __init__(self, policy_env_info: PolicyEnvInterface, agent_ids: Sequence[int] | None = None):
        super().__init__(policy_env_info)
        ids = list(agent_ids) if agent_ids is not None else list(range(policy_env_info.num_agents))
        self._handle = _NimPolicyHandle(
            na.ThinkyPolicy,
            "thinky_policy_step_batch",
            policy_env_info,
            ids,
        )
        self._agent_ids = set(ids)

    def agent_policy(self, agent_id: int) -> ThinkyAgentPolicy:
        if agent_id not in self._agent_ids:
            raise ValueError(f"Agent id {agent_id} not handled by this policy")
        return ThinkyAgentPolicy(self._policy_env_info, agent_id, handle=self._handle)

    def step_batch(self, raw_observations: np.ndarray, raw_actions: np.ndarray) -> None:
        self._handle.step_batch(raw_observations, raw_actions)


class RandomAgentPolicy(AgentPolicy):
    short_names = ["nim_random", "fast_random"]

    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int, handle: _NimPolicyHandle | None = None):
        super().__init__(policy_env_info)
        obs_shape = policy_env_info.observation_space.shape
        self._agent_id = agent_id
        self._num_agents = policy_env_info.num_agents
        self._num_tokens = obs_shape[0]
        self._token_dim = obs_shape[1]
        self._batch_obs = np.empty((self._num_agents, *obs_shape), dtype=dtype_observations)
        self._batch_actions = np.zeros(self._num_agents, dtype=np.int32)
        self._action_names = policy_env_info.action_names
        self._single_subset = np.array([agent_id], dtype=np.int32)
        self._handle = handle or _NimPolicyHandle(
            na.RandomPolicy,
            "random_policy_step_batch",
            policy_env_info,
            [agent_id],
        )

    def _pack_obs(self, target: np.ndarray, obs: AgentObservation) -> None:
        target.fill(255)
        for idx, token in enumerate(obs.tokens):
            if idx >= self._num_tokens:
                break
            target[idx, : len(token.raw_token)] = token.raw_token

    def step(self, obs: AgentObservation) -> Action:
        self._batch_obs.fill(255)
        self._pack_obs(self._batch_obs[self._agent_id], obs)
        self._batch_actions.fill(0)
        self._handle.step_batch(self._batch_obs, self._batch_actions, agent_subset=self._single_subset)
        action_index = int(self._batch_actions[self._agent_id])
        return Action(name=self._action_names[action_index])


class RandomAgentsMultiPolicy(ThinkyAgentsMultiPolicy):
    short_names = ["nim_random", "fast_random"]

    def __init__(self, policy_env_info: PolicyEnvInterface, agent_ids: Sequence[int] | None = None):
        ids = list(agent_ids) if agent_ids is not None else list(range(policy_env_info.num_agents))
        self._handle = _NimPolicyHandle(
            na.RandomPolicy,
            "random_policy_step_batch",
            policy_env_info,
            ids,
        )
        self._agent_ids = set(ids)

    def agent_policy(self, agent_id: int) -> RandomAgentPolicy:
        if agent_id not in self._agent_ids:
            raise ValueError(f"Agent id {agent_id} not handled by this policy")
        return RandomAgentPolicy(self._policy_env_info, agent_id, handle=self._handle)


class RaceCarAgentPolicy(AgentPolicy):
    short_names = ["nim_race_car", "fast_race_car"]

    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int, handle: _NimPolicyHandle | None = None):
        super().__init__(policy_env_info)
        obs_shape = policy_env_info.observation_space.shape
        self._agent_id = agent_id
        self._num_agents = policy_env_info.num_agents
        self._num_tokens = obs_shape[0]
        self._token_dim = obs_shape[1]
        self._batch_obs = np.empty((self._num_agents, *obs_shape), dtype=dtype_observations)
        self._batch_actions = np.zeros(self._num_agents, dtype=np.int32)
        self._action_names = policy_env_info.action_names
        self._single_subset = np.array([agent_id], dtype=np.int32)
        self._handle = handle or _NimPolicyHandle(
            na.RaceCarPolicy,
            "race_car_policy_step_batch",
            policy_env_info,
            [agent_id],
        )

    def _pack_obs(self, target: np.ndarray, obs: AgentObservation) -> None:
        target.fill(255)
        for idx, token in enumerate(obs.tokens):
            if idx >= self._num_tokens:
                break
            target[idx, : len(token.raw_token)] = token.raw_token

    def step(self, obs: AgentObservation) -> Action:
        self._batch_obs.fill(255)
        self._pack_obs(self._batch_obs[self._agent_id], obs)
        self._batch_actions.fill(0)
        self._handle.step_batch(self._batch_obs, self._batch_actions, agent_subset=self._single_subset)
        action_index = int(self._batch_actions[self._agent_id])
        return Action(name=self._action_names[action_index])


class RaceCarAgentsMultiPolicy(ThinkyAgentsMultiPolicy):
    short_names = ["nim_race_car", "fast_race_car"]

    def __init__(self, policy_env_info: PolicyEnvInterface, agent_ids: Sequence[int] | None = None):
        ids = list(agent_ids) if agent_ids is not None else list(range(policy_env_info.num_agents))
        self._handle = _NimPolicyHandle(
            na.RaceCarPolicy,
            "race_car_policy_step_batch",
            policy_env_info,
            ids,
        )
        self._agent_ids = set(ids)

    def agent_policy(self, agent_id: int) -> RaceCarAgentPolicy:
        if agent_id not in self._agent_ids:
            raise ValueError(f"Agent id {agent_id} not handled by this policy")
        return RaceCarAgentPolicy(self._policy_env_info, agent_id, handle=self._handle)
