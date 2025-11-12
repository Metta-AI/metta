import ctypes
import importlib
import os
import sys
from typing import Any, Sequence

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


class _NimPolicyHandleManager:
    def __init__(
        self,
        handle_ctor: Any,
        step_batch_name: str,
        policy_env_info: PolicyEnvInterface,
        agent_ids: Sequence[int],
    ) -> None:
        agent_id_list = list(agent_ids)
        if not agent_id_list:
            raise ValueError("agent_ids must not be empty")
        self._handle = handle_ctor(policy_env_info.to_json())
        self._step_batch = getattr(self._handle, step_batch_name)
        self._num_agents = policy_env_info.num_agents
        obs_shape = policy_env_info.observation_space.shape
        self._num_tokens = obs_shape[0]
        self._token_dim = obs_shape[1]
        self._default_subset = np.array(agent_id_list, dtype=np.int32)
        self._default_subset_ptr = (
            self._default_subset.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
            if self._default_subset.size > 0
            else None
        )

    def _subset_ptr(self, subset: np.ndarray | None) -> tuple[ctypes.POINTER(ctypes.c_int32) | None, int]:
        if subset is None:
            return self._default_subset_ptr, int(self._default_subset.size)
        if subset.size == 0:
            return None, 0
        return subset.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), int(subset.size)

    def step_batch(
        self,
        raw_observations: np.ndarray,
        raw_actions: np.ndarray,
        subset: np.ndarray | None = None,
    ) -> None:
        subset_ptr, subset_len = self._subset_ptr(subset)
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


class _NimAgentPolicyBase(AgentPolicy):
    handle_cls: Any | None = None
    handle_step_batch: str = ""

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        agent_id: int,
        handle_manager: _NimPolicyHandleManager | None = None,
    ) -> None:
        super().__init__(policy_env_info)
        assert self.handle_cls is not None, "handle_cls must be set in subclasses"
        obs_shape = policy_env_info.observation_space.shape
        self._agent_id = agent_id
        self._num_agents = policy_env_info.num_agents
        self._num_tokens = obs_shape[0]
        self._token_dim = obs_shape[1]
        self._batch_obs = np.empty((self._num_agents, *obs_shape), dtype=dtype_observations)
        self._batch_actions = np.zeros(self._num_agents, dtype=np.int32)
        self._action_names = policy_env_info.action_names
        if handle_manager is None:
            handle_manager = _NimPolicyHandleManager(
                self.handle_cls,
                self.handle_step_batch,
                policy_env_info,
                [agent_id],
            )
        self._handle_manager = handle_manager
        self._single_subset = np.array([agent_id], dtype=np.int32)

    def _pack_raw_observation(self, target: np.ndarray, obs: AgentObservation) -> None:
        target.fill(255)
        for idx, token in enumerate(obs.tokens):
            if idx >= self._num_tokens:
                break
            token_values = token.raw_token
            target[idx, : len(token_values)] = token_values

    def step(self, obs: AgentObservation) -> Action:
        self._batch_obs.fill(255)
        self._pack_raw_observation(self._batch_obs[self._agent_id], obs)
        self._batch_actions.fill(0)
        self._handle_manager.step_batch(self._batch_obs, self._batch_actions, subset=self._single_subset)
        action_index = int(self._batch_actions[self._agent_id])
        return Action(name=self._action_names[action_index])


class _NimAgentsMultiPolicyBase(MultiAgentPolicy):
    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        policy_type: type[_NimAgentPolicyBase],
        agent_ids: Sequence[int] | None = None,
    ) -> None:
        super().__init__(policy_env_info)
        agent_id_list = list(agent_ids) if agent_ids is not None else list(range(policy_env_info.num_agents))
        self._policy_type = policy_type
        self._handle_manager = _NimPolicyHandleManager(
            policy_type.handle_cls,
            policy_type.handle_step_batch,
            policy_env_info,
            agent_id_list,
        )
        self._agent_ids = set(agent_id_list)

    def agent_policy(self, agent_id: int) -> _NimAgentPolicyBase:
        if agent_id not in self._agent_ids:
            raise ValueError(f"Agent id {agent_id} not handled by this policy")
        return self._policy_type(self._policy_env_info, agent_id, handle_manager=self._handle_manager)

    def step_batch(self, raw_observations: np.ndarray, raw_actions: np.ndarray) -> None:
        self._handle_manager.step_batch(raw_observations, raw_actions)


class RandomAgentPolicy(_NimAgentPolicyBase):
    handle_cls = na.RandomPolicy
    handle_step_batch = "random_policy_step_batch"


class RandomAgentsMultiPolicy(_NimAgentsMultiPolicyBase):
    short_names = ["nim_random", "fast_random"]

    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info, RandomAgentPolicy)


class ThinkyAgentPolicy(_NimAgentPolicyBase):
    handle_cls = na.ThinkyPolicy
    handle_step_batch = "thinky_policy_step_batch"


class ThinkyAgentsMultiPolicy(_NimAgentsMultiPolicyBase):
    short_names = ["nim_thinky", "fast_thinky"]

    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info, ThinkyAgentPolicy)


class RaceCarAgentPolicy(_NimAgentPolicyBase):
    handle_cls = na.RaceCarPolicy
    handle_step_batch = "race_car_policy_step_batch"


class RaceCarAgentsMultiPolicy(_NimAgentsMultiPolicyBase):
    short_names = ["nim_race_car", "fast_race_car"]

    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info, RaceCarAgentPolicy)
