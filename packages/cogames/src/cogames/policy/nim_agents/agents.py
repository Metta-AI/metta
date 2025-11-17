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

# Lazy import - will be loaded when needed
_na_module = None


def _get_nim_agents_module():
    """Lazy import of nim_agents module."""
    global _na_module
    if _na_module is None:
        _na_module = importlib.import_module("nim_agents")
    return _na_module


def start_measure():
    _get_nim_agents_module().start_measure()


def end_measure():
    _get_nim_agents_module().end_measure()


class ThinkyAgentsMultiPolicy(NimMultiAgentPolicy):
    short_names = ["nim_thinky", "thinky"]

    def __init__(self, policy_env_info: PolicyEnvInterface, agent_ids: Sequence[int] | None = None):
        na = _get_nim_agents_module()
        super().__init__(
            policy_env_info,
            nim_policy_factory=na.ThinkyPolicy,
            agent_ids=agent_ids,
        )


class RandomAgentsMultiPolicy(NimMultiAgentPolicy):
    short_names = ["nim_random"]

    def __init__(self, policy_env_info: PolicyEnvInterface, agent_ids: Sequence[int] | None = None):
        na = _get_nim_agents_module()
        super().__init__(
            policy_env_info,
            nim_policy_factory=na.RandomPolicy,
            agent_ids=agent_ids,
        )


class RaceCarAgentsMultiPolicy(NimMultiAgentPolicy):
    short_names = ["nim_race_car"]

    def __init__(self, policy_env_info: PolicyEnvInterface, agent_ids: Sequence[int] | None = None):
        na = _get_nim_agents_module()
        super().__init__(
            policy_env_info,
            nim_policy_factory=na.RaceCarPolicy,
            agent_ids=agent_ids,
        )


class LadyBugAgentsMultiPolicy(NimMultiAgentPolicy):
    short_names = ["nim_ladybug"]

    def __init__(self, policy_env_info: PolicyEnvInterface, agent_ids: Sequence[int] | None = None):
        na = _get_nim_agents_module()
        super().__init__(
            policy_env_info,
            nim_policy_factory=na.LadybugPolicy,
            agent_ids=agent_ids,
        )


# Backwards compatibility for older import path (lowercase "b")
class LadybugAgentsMultiPolicy(LadyBugAgentsMultiPolicy):
    pass
