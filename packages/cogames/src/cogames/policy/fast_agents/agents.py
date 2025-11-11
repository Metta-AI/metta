import ctypes
import os
import sys

import numpy as np

from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation, Simulation

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "bindings/generated"))


class RandomAgentPolicy(AgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int):
        super().__init__(policy_env_info)
        import fast_agents as fa  # isort: skip

        self._agent = fa.RandomAgent(agent_id, policy_env_info.to_json())
        self._action_names = [action.name for action in policy_env_info.actions.actions()]

    def step_batch(self, raw_observations: np.ndarray, raw_actions: np.ndarray) -> None:
        obs_ptr = raw_observations.ctypes.data_as(ctypes.c_void_p)
        action_ptr = raw_actions.ctypes.data_as(ctypes.c_void_p)
        self._agent.step_batch(
            num_agents=raw_observations.shape[0],
            num_tokens=raw_observations.shape[1],
            size_token=raw_observations.shape[2],
            raw_observations=obs_ptr,
            num_actions=raw_actions.shape[0],
            raw_actions=action_ptr,
        )

    def step(self, obs: AgentObservation) -> Action:
        if obs.raw_observation is None:
            raise ValueError("Nim agents require raw observation buffers.")
        raw = obs.raw_observation
        obs_ptr = raw.ctypes.data_as(ctypes.c_void_p)
        action_index = self._agent.step(
            num_tokens=raw.shape[0],
            size_token=raw.shape[1],
            raw_observation=obs_ptr,
        )
        return Action(name=self._action_names[action_index])

    def reset(self, simulation: Simulation = None) -> None:
        self._agent.reset()


class RandomAgentsMultiPolicy(MultiAgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)

    def agent_policy(self, agent_id: int) -> RandomAgentPolicy:
        return RandomAgentPolicy(self._policy_env_info, agent_id)


class ThinkyAgentPolicy(AgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int):
        super().__init__(policy_env_info)
        import fast_agents as fa  # isort: skip

        self._agent = fa.ThinkyAgent(agent_id, policy_env_info.to_json())
        self._action_names = [action.name for action in policy_env_info.actions.actions()]

    def step_batch(self, raw_observations: np.ndarray, raw_actions: np.ndarray) -> None:
        obs_ptr = raw_observations.ctypes.data_as(ctypes.c_void_p)
        action_ptr = raw_actions.ctypes.data_as(ctypes.c_void_p)
        self._agent.step_batch(
            num_agents=raw_observations.shape[0],
            num_tokens=raw_observations.shape[1],
            size_token=raw_observations.shape[2],
            raw_observations=obs_ptr,
            num_actions=raw_actions.shape[0],
            raw_actions=action_ptr,
        )

    def step(self, obs: AgentObservation) -> Action:
        if obs.raw_observation is None:
            raise ValueError("Nim agents require raw observation buffers.")
        raw = obs.raw_observation
        obs_ptr = raw.ctypes.data_as(ctypes.c_void_p)
        action_index = self._agent.step(
            num_tokens=raw.shape[0],
            size_token=raw.shape[1],
            raw_observation=obs_ptr,
        )
        return Action(name=self._action_names[action_index])

    def reset(self, simulation: Simulation = None) -> None:
        self._agent.reset()


class ThinkyAgentsMultiPolicy(MultiAgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)

    def agent_policy(self, agent_id: int) -> ThinkyAgentPolicy:
        return ThinkyAgentPolicy(self._policy_env_info, agent_id)


class RaceCarAgentPolicy(AgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int):
        super().__init__(policy_env_info)
        import fast_agents as fa  # isort: skip

        self._agent = fa.RaceCarAgent(agent_id, policy_env_info.to_json())
        self._action_names = [action.name for action in policy_env_info.actions.actions()]

    def step_batch(self, raw_observations: np.ndarray, raw_actions: np.ndarray) -> None:
        obs_ptr = raw_observations.ctypes.data_as(ctypes.c_void_p)
        action_ptr = raw_actions.ctypes.data_as(ctypes.c_void_p)
        self._agent.step_batch(
            num_agents=raw_observations.shape[0],
            num_tokens=raw_observations.shape[1],
            size_token=raw_observations.shape[2],
            raw_observations=obs_ptr,
            num_actions=raw_actions.shape[0],
            raw_actions=action_ptr,
        )

    def step(self, obs: AgentObservation) -> Action:
        if obs.raw_observation is None:
            raise ValueError("Nim agents require raw observation buffers.")
        raw = obs.raw_observation
        obs_ptr = raw.ctypes.data_as(ctypes.c_void_p)
        action_index = self._agent.step(
            num_tokens=raw.shape[0],
            size_token=raw.shape[1],
            raw_observation=obs_ptr,
        )
        return Action(name=self._action_names[action_index])

    def reset(self, simulation: Simulation = None) -> None:
        self._agent.reset()


class RaceCarAgentsMultiPolicy(MultiAgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)

    def agent_policy(self, agent_id: int) -> RaceCarAgentPolicy:
        return RaceCarAgentPolicy(self._policy_env_info, agent_id)
