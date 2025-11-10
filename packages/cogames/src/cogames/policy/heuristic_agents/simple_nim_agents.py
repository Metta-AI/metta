import importlib
import json
import os
import sys

from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation, Simulation
from mettagrid.simulator.simulator import Buffers

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "bindings/generated"))

ha = importlib.import_module("heuristic_agents")

# ha.initCHook()


class HeuristicAgentPolicy(AgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int):
        super().__init__(policy_env_info)
        # Convert the policy_env_info to a JSON string.
        config = {
            "num_agents": policy_env_info.num_agents,
            "obs_width": policy_env_info.obs_width,
            "obs_height": policy_env_info.obs_height,
            "actions": [action.name for action in policy_env_info.actions.actions()],
            "type_names": {
                "agent": 0,
                "assembler": 1,
                "carbonExtractor": 2,
                "charger": 3,
                "chest": 4,
                "germaniumExtractor": 9,
                "oxygenExtractor": 10,
                "siliconExtractor": 11,
                "wall": 12,
            },
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
        self._agent = ha.HeuristicAgent(agent_id, json.dumps(config))
        self._agent_id = agent_id
        self._simulation: Simulation | None = None
        self._buffers: Buffers | None = None

    def reset(self, simulation: Simulation | None) -> None:
        if simulation is None:
            raise RuntimeError("HeuristicAgentPolicy requires simulation access; pass pass_sim_to_policies=True.")
        if simulation.buffers is None:
            raise RuntimeError("Simulation is not configured with shared buffers required by HeuristicAgentPolicy.")
        self._simulation = simulation
        self._buffers = simulation.buffers

    def step(self, obs: AgentObservation) -> Action:
        if self._simulation is None or self._buffers is None:
            raise RuntimeError("HeuristicAgentPolicy must be reset with a simulation before stepping.")

        raw_obs = self._buffers.observations
        raw_actions = self._buffers.actions

        self._agent.step(
            num_agents=raw_obs.shape[0],
            num_tokens=raw_obs.shape[1],
            size_token=raw_obs.shape[2],
            row_observations=raw_obs.ctypes.data,
            num_actions=raw_actions.shape[0],
            raw_actions=raw_actions.ctypes.data,
        )

        action_idx = int(raw_actions[self._agent_id])
        action_name = self._simulation.action_names[action_idx]
        return Action(name=action_name)


class HeuristicAgentsPolicy(MultiAgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)

    def agent_policy(self, agent_id: int) -> HeuristicAgentPolicy:
        return HeuristicAgentPolicy(self._policy_env_info, agent_id)
