import json
import os
import sys

import numpy as np

from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface


class FastAgentsPolicy(AgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int):
        super().__init__(policy_env_info)
        # Convert the policy_env_info to a JSON string.
        config = {
            "num_agents": policy_env_info.num_agents,
            "obs_width": policy_env_info.obs_width,
            "obs_height": policy_env_info.obs_height,
            "actions": [action.name for action in policy_env_info.actions.actions()],
            "tags": policy_env_info.tags,
            "obs_features": [],
        }
        for feature in policy_env_info.obs_features:
            config["obs_features"].append(
                {
                    "id": feature.id,
                    "name": feature.name,
                    "normalization": feature.normalization,
                }
            )
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(os.path.join(current_dir, "bindings/generated"))
        import fast_agents as fa

        self._agent = fa.FastAgents(agent_id, json.dumps(config))

    def step(self, raw_obs: np.ndarray, raw_action: np.ndarray):
        # Pass everything to the Nim agent.
        self._agent.step(
            num_agents=raw_obs.shape[0],
            num_tokens=raw_obs.shape[1],
            size_token=raw_obs.shape[2],
            raw_observations=raw_obs.ctypes.data,
            num_actions=raw_action.shape[0],
            raw_actions=raw_action.ctypes.data,
        )


class FastAgentsMultiPolicy(MultiAgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)

    def agent_policy(self, agent_id: int) -> FastAgentsPolicy:
        return FastAgentsPolicy(self._policy_env_info, agent_id)
