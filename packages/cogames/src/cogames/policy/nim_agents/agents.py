import importlib
import os
import sys
from typing import Sequence

from mettagrid.policy.policy import NimMultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface

current_dir = os.path.dirname(os.path.abspath(__file__))
bindings_dir = os.path.join(current_dir, "bindings/generated")
if bindings_dir not in sys.path:
    sys.path.append(bindings_dir)

na = importlib.import_module("nim_agents")


def start_measure():
    na.start_measure()


def end_measure():
    na.end_measure()


class ThinkyAgentsMultiPolicy(NimMultiAgentPolicy):
    short_names = ["thinky"]

    def __init__(self, policy_env_info: PolicyEnvInterface, agent_ids: Sequence[int] | None = None):
        super().__init__(
            policy_env_info,
            nim_policy_factory=na.ThinkyPolicy,
            agent_ids=agent_ids,
        )


class RandomAgentsMultiPolicy(NimMultiAgentPolicy):
    short_names = ["nim_random"]

    def __init__(self, policy_env_info: PolicyEnvInterface, agent_ids: Sequence[int] | None = None):
        super().__init__(
            policy_env_info,
            nim_policy_factory=na.RandomPolicy,
            agent_ids=agent_ids,
        )


class RaceCarAgentsMultiPolicy(NimMultiAgentPolicy):
    short_names = ["race_car"]

    def __init__(self, policy_env_info: PolicyEnvInterface, agent_ids: Sequence[int] | None = None):
        super().__init__(
            policy_env_info,
            nim_policy_factory=na.RaceCarPolicy,
            agent_ids=agent_ids,
        )


class LadyBugAgentsMultiPolicy(NimMultiAgentPolicy):
    short_names = ["nim_ladybug"]

    def __init__(self, policy_env_info: PolicyEnvInterface, agent_ids: Sequence[int] | None = None):
        super().__init__(
            policy_env_info,
            nim_policy_factory=na.LadybugPolicy,
            agent_ids=agent_ids,
        )
