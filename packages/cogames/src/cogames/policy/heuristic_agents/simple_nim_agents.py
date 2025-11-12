from __future__ import annotations

import importlib
import json
import os
import sys
from typing import Any

import numpy as np

from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface

_HA_MODULE: Any | None = None


def _heuristic_agents_module() -> Any:
    global _HA_MODULE
    if _HA_MODULE is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        bindings_dir = os.path.join(current_dir, "bindings/generated")
        if bindings_dir not in sys.path:
            sys.path.append(bindings_dir)
        _HA_MODULE = importlib.import_module("heuristic_agents")
    return _HA_MODULE


class HeuristicAgentPolicy(AgentPolicy):
    def __init__(self, policy_env_info: PolicyEnvInterface, agent_id: int):
        super().__init__(policy_env_info)
        ha = _heuristic_agents_module()
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

    def step(self, raw_obs: np.ndarray, raw_action: np.ndarray):
        self._agent.step(
            raw_obs.shape[0],
            raw_obs.shape[1],
            raw_obs.shape[2],
            raw_obs.ctypes.data,
            raw_action.shape[0],
            raw_action.ctypes.data,
        )


class HeuristicAgentsPolicy(MultiAgentPolicy):
    short_names = ["heuristic_agents"]

    def __init__(self, policy_env_info: PolicyEnvInterface):
        super().__init__(policy_env_info)

    def agent_policy(self, agent_id: int) -> HeuristicAgentPolicy:
        return HeuristicAgentPolicy(self._policy_env_info, agent_id)
