import json
import os
import sys

import numpy as np

from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Simulation


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "bindings/generated"))


class RandomAgentPolicy(AgentPolicy):

    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int):
        super().__init__(policy_env_info)
        self.uses_raw_numpy = True
        import fast_agents as fa # Lint wants it here...
        self._agent = fa.RandomAgent(agent_id, policy_env_info.to_json())

    def step(self, raw_observations: np.ndarray, raw_actions: np.ndarray):
        # Pass everything to the Nim agent.
        self._agent.step(
            num_agents=raw_observations.shape[0],
            num_tokens=raw_observations.shape[1],
            size_token=raw_observations.shape[2],
            raw_observations=raw_observations.ctypes.data,
            num_actions=raw_actions.shape[0],
            raw_actions=raw_actions.ctypes.data,
        )

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
        self.uses_raw_numpy = True
        import fast_agents as fa # Lint wants it here...
        self._agent = fa.ThinkyAgent(agent_id, policy_env_info.to_json())

    def step(self, raw_observations: np.ndarray, raw_actions: np.ndarray):
        # Pass everything to the Nim agent.
        self._agent.step(
            num_agents=raw_observations.shape[0],
            num_tokens=raw_observations.shape[1],
            size_token=raw_observations.shape[2],
            raw_observations=raw_observations.ctypes.data,
            num_actions=raw_actions.shape[0],
            raw_actions=raw_actions.ctypes.data,
        )

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
        self.uses_raw_numpy = True
        import fast_agents as fa # Lint wants it here...
        self._agent = fa.RaceCarAgent(agent_id, policy_env_info.to_json())

    def step(self, raw_observations: np.ndarray, raw_actions: np.ndarray):
        # Pass everything to the Nim agent.
        self._agent.step(
            num_agents=raw_observations.shape[0],
            num_tokens=raw_observations.shape[1],
            size_token=raw_observations.shape[2],
            raw_observations=raw_observations.ctypes.data,
            num_actions=raw_actions.shape[0],
            raw_actions=raw_actions.ctypes.data,
        )

    def reset(self, simulation: Simulation = None) -> None:
        self._agent.reset()


class RaceCarAgentsMultiPolicy(MultiAgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)

    def agent_policy(self, agent_id: int) -> RaceCarAgentPolicy:
        return RaceCarAgentPolicy(self._policy_env_info, agent_id)
