import json
import os
import sys
from typing import Optional

import numpy as np

from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action, AgentObservation, Simulation
from mettagrid.simulator.simulator import Buffers

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "bindings/generated"))

import heuristic_agents as ha  # noqa: E402

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
            "tag_names": {
                "agent": 99,
                "assembler": 0,
                "carbonExtractor": 1,
                "charger": 2,
                "chest": 3,
                "germaniumExtractor": 4,
                "oxygenExtractor": 5,
                "siliconExtractor": 6,
                "wall": 7,
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
        self._simulation = simulation
        self._buffers = simulation.buffers

    def _resolve_arrays(self, simulation: Simulation) -> tuple[np.ndarray, np.ndarray]:
        buffers = simulation.buffers
        if buffers is not None:
            return buffers.observations, buffers.actions
        c_sim = simulation._c_sim  # Access shared C++ buffers when Python ones aren't provided
        return c_sim.observations(), c_sim.actions()

    def _step_with_arrays(self, simulation: Simulation, raw_obs: np.ndarray, raw_actions: np.ndarray) -> Action:
        self._agent.step(
            raw_obs.shape[0],
            raw_obs.shape[1],
            raw_obs.shape[2],
            raw_obs.ctypes.data,
            raw_actions.shape[0],
            raw_actions.ctypes.data,
        )

        action_idx = int(raw_actions[self._agent_id])
        action_name = simulation.action_names[action_idx]
        return Action(name=action_name)

    def step_with_simulation(self, simulation: Simulation | None) -> Action | None:
        if simulation is None:
            return None
        self._simulation = simulation
        self._buffers = simulation.buffers
        raw_obs, raw_actions = self._resolve_arrays(simulation)
        return self._step_with_arrays(simulation, raw_obs, raw_actions)

    def step(self, obs: AgentObservation) -> Action:
        if self._simulation is None:
            raise RuntimeError("HeuristicAgentPolicy must be reset with a simulation before stepping.")
        raw_obs, raw_actions = self._resolve_arrays(self._simulation)
        return self._step_with_arrays(self._simulation, raw_obs, raw_actions)


class HeuristicAgentsPolicy(MultiAgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)

    def agent_policy(self, agent_id: int) -> HeuristicAgentPolicy:
        return HeuristicAgentPolicy(self._policy_env_info, agent_id)

    def step_batch(
        self,
        simulation: Simulation,
        out_actions: Optional[np.ndarray] = None,
        observations: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        del observations

        num_agents = simulation.num_agents
        if out_actions is None:
            out_actions = np.empty(num_agents, dtype=np.int32)

        agents = simulation.agents()
        action_ids = simulation.action_ids

        for agent_id in range(num_agents):
            policy = self.agent_policy(agent_id)
            action = policy.step_with_simulation(simulation)
            if action is None:
                action = policy.step(agents[agent_id].observation)
            out_actions[agent_id] = action_ids[action.name]

        return out_actions
