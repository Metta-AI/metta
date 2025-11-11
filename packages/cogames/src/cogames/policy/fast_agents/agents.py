import ctypes
import os
import sys

import numpy as np

from mettagrid.mettagrid_c import dtype_observations
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation, Simulation

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "bindings/generated"))


def _import_fast_agents():
    """Import the generated bindings while forcing genny's `pointer` type to act like c_void_p."""
    pointer_backup = ctypes.pointer
    ctypes.pointer = ctypes.c_void_p
    try:
        import fast_agents as fa  # type: ignore import error handled by sys.path above
    finally:
        ctypes.pointer = pointer_backup
    return fa


class _FastAgentPolicyBase(AgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)
        obs_shape = policy_env_info.observation_space.shape
        self._num_tokens = obs_shape[0]
        self._token_dim = obs_shape[1]
        self._single_obs = np.empty((1, *obs_shape), dtype=dtype_observations)
        self._single_actions = np.zeros(1, dtype=np.int32)

    def _pack_raw_observation(self, obs: AgentObservation) -> np.ndarray:
        raw = self._single_obs[0]
        raw.fill(255)
        for idx, token in enumerate(obs.tokens):
            if idx >= self._num_tokens:
                break
            token_values = token.raw_token
            raw[idx, : len(token_values)] = token_values
        return self._single_obs


class RandomAgentPolicy(_FastAgentPolicyBase):
    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int):
        super().__init__(policy_env_info)
        fa = _import_fast_agents()
        self._agent = fa.RandomAgent(agent_id, policy_env_info.to_json())
        self._action_names = [action.name for action in policy_env_info.actions.actions()]

    def step_batch(self, raw_observations: np.ndarray, raw_actions: np.ndarray) -> None:
        self._agent.step(
            num_agents=raw_observations.shape[0],
            num_tokens=raw_observations.shape[1],
            size_token=raw_observations.shape[2],
            raw_observations=raw_observations.ctypes.data,
            num_actions=raw_actions.shape[0],
            raw_actions=raw_actions.ctypes.data,
        )

    def step(self, obs: AgentObservation) -> Action:
        raw_batch = np.ascontiguousarray(self._pack_raw_observation(obs))
        self._single_actions.fill(0)
        self._agent.step(
            num_agents=1,
            num_tokens=self._num_tokens,
            size_token=self._token_dim,
            raw_observations=raw_batch.ctypes.data,
            num_actions=1,
            raw_actions=self._single_actions.ctypes.data,
        )
        action_index = int(self._single_actions[0])
        return Action(name=self._action_names[action_index])

    def reset(self, simulation: Simulation = None) -> None:
        self._agent.reset()


class RandomAgentsMultiPolicy(MultiAgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)

    short_names = ["fast_random"]

    def agent_policy(self, agent_id: int) -> RandomAgentPolicy:
        return RandomAgentPolicy(self._policy_env_info, agent_id)


class ThinkyAgentPolicy(_FastAgentPolicyBase):
    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int):
        super().__init__(policy_env_info)
        fa = _import_fast_agents()
        self._agent = fa.ThinkyAgent(agent_id, policy_env_info.to_json())
        self._action_names = [action.name for action in policy_env_info.actions.actions()]

    def step_batch(self, raw_observations: np.ndarray, raw_actions: np.ndarray) -> None:
        self._agent.step(
            num_agents=raw_observations.shape[0],
            num_tokens=raw_observations.shape[1],
            size_token=raw_observations.shape[2],
            raw_observations=raw_observations.ctypes.data,
            num_actions=raw_actions.shape[0],
            raw_actions=raw_actions.ctypes.data,
        )

    def step(self, obs: AgentObservation) -> Action:
        raw_batch = np.ascontiguousarray(self._pack_raw_observation(obs))
        self._single_actions.fill(0)
        self._agent.step(
            num_agents=1,
            num_tokens=self._num_tokens,
            size_token=self._token_dim,
            raw_observations=raw_batch.ctypes.data,
            num_actions=1,
            raw_actions=self._single_actions.ctypes.data,
        )
        action_index = int(self._single_actions[0])
        return Action(name=self._action_names[action_index])

    def reset(self, simulation: Simulation = None) -> None:
        self._agent.reset()


class ThinkyAgentsMultiPolicy(MultiAgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)

    short_names = ["fast_thinky"]

    def agent_policy(self, agent_id: int) -> ThinkyAgentPolicy:
        return ThinkyAgentPolicy(self._policy_env_info, agent_id)


class RaceCarAgentPolicy(_FastAgentPolicyBase):
    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int):
        super().__init__(policy_env_info)
        fa = _import_fast_agents()
        self._agent = fa.RaceCarAgent(agent_id, policy_env_info.to_json())
        self._action_names = [action.name for action in policy_env_info.actions.actions()]

    def step_batch(self, raw_observations: np.ndarray, raw_actions: np.ndarray) -> None:
        self._agent.step(
            num_agents=raw_observations.shape[0],
            num_tokens=raw_observations.shape[1],
            size_token=raw_observations.shape[2],
            raw_observations=raw_observations.ctypes.data,
            num_actions=raw_actions.shape[0],
            raw_actions=raw_actions.ctypes.data,
        )

    def step(self, obs: AgentObservation) -> Action:
        raw_batch = np.ascontiguousarray(self._pack_raw_observation(obs))
        self._single_actions.fill(0)
        self._agent.step(
            num_agents=1,
            num_tokens=self._num_tokens,
            size_token=self._token_dim,
            raw_observations=raw_batch.ctypes.data,
            num_actions=1,
            raw_actions=self._single_actions.ctypes.data,
        )
        action_index = int(self._single_actions[0])
        return Action(name=self._action_names[action_index])

    def reset(self, simulation: Simulation = None) -> None:
        self._agent.reset()


class RaceCarAgentsMultiPolicy(MultiAgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)

    short_names = ["fast_race_car"]

    def agent_policy(self, agent_id: int) -> RaceCarAgentPolicy:
        return RaceCarAgentPolicy(self._policy_env_info, agent_id)
