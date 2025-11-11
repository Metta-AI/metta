from __future__ import annotations

import ctypes
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

ha.dll.heuristic_agents_heuristic_agent_step.argtypes = [
    ha.HeuristicAgent,
    ctypes.c_longlong,
    ctypes.c_longlong,
    ctypes.c_longlong,
    ctypes.c_void_p,
    ctypes.c_longlong,
    ctypes.c_void_p,
]


class HeuristicAgentPolicy(AgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int):
        super().__init__(policy_env_info)
        config = {
            "num_agents": policy_env_info.num_agents,
            "obs_width": policy_env_info.obs_width,
            "obs_height": policy_env_info.obs_height,
            "actions": [action.name for action in policy_env_info.actions.actions()],
            "tags": policy_env_info.tags,
            "obs_features": [
                {
                    "id": feature.id,
                    "name": feature.name,
                    "normalization": feature.normalization,
                }
                for feature in policy_env_info.obs_features
            ],
        }
        self._agent = ha.HeuristicAgent(agent_id, json.dumps(config))
        self._agent_id = agent_id
        self._simulation: Optional[Simulation] = None
        self._buffers: Optional[Buffers] = None

    def reset(self, simulation: Optional[Simulation]) -> None:
        if simulation is None or simulation.buffers is None:
            raise RuntimeError("HeuristicAgentPolicy requires shared buffers; pass a simulation with buffers attached.")
        self._simulation = simulation
        self._buffers = simulation.buffers

    def step(self, obs: AgentObservation) -> Action:
        del obs
        assert self._simulation is not None and self._buffers is not None
        raw_obs = self._buffers.observations
        raw_actions = self._buffers.actions

        self._agent.step(
            raw_obs.shape[0],
            raw_obs.shape[1],
            raw_obs.shape[2],
            raw_obs.ctypes.data,
            raw_actions.shape[0],
            raw_actions.ctypes.data,
        )

        action_idx = int(raw_actions[self._agent_id])
        action_name = self._simulation.action_names[action_idx]
        return Action(name=action_name)


class HeuristicAgentsPolicy(MultiAgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)
        self._policies: dict[int, HeuristicAgentPolicy] = {}

    def agent_policy(self, agent_id: int) -> HeuristicAgentPolicy:
        if agent_id not in self._policies:
            self._policies[agent_id] = HeuristicAgentPolicy(self._policy_env_info, agent_id)
        return self._policies[agent_id]

    def step_batch(
        self,
        simulation: Simulation,
        out_actions: Optional[np.ndarray] = None,
        observations: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        del observations
        if simulation.buffers is None:
            raise RuntimeError("Shared buffers must be provided for heuristic supervisor policies.")

        num_agents = simulation.num_agents
        if out_actions is None:
            out_actions = np.empty(num_agents, dtype=np.int32)

        action_ids = simulation.action_ids
        agents = simulation.agents()

        for agent_id in range(num_agents):
            policy = self.agent_policy(agent_id)
            policy.reset(simulation)
            action = policy.step(agents[agent_id].observation)
            out_actions[agent_id] = action_ids[action.name]

        return out_actions
