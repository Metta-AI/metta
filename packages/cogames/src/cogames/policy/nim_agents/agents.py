import importlib
import os
import sys
from typing import Sequence

from mettagrid.policy.policy import NimAgentPolicyBase, NimMultiAgentPolicy
from mettagrid.policy.policy_env_interface import PolicyEnvInterface

current_dir = os.path.dirname(os.path.abspath(__file__))
bindings_dir = os.path.join(current_dir, "bindings/generated")
if bindings_dir not in sys.path:
    sys.path.append(bindings_dir)

na = importlib.import_module("nim_agents")


class ThinkyAgentPolicy(NimAgentPolicyBase):
    pass


class ThinkyAgentsMultiPolicy(NimMultiAgentPolicy):
    short_names = ["nim_thinky"]

    def __init__(self, policy_env_info: PolicyEnvInterface, agent_ids: Sequence[int] | None = None):
        super().__init__(
            policy_env_info,
            handle_ctor=na.ThinkyPolicy,
            step_batch_name="thinky_policy_step_batch",
            agent_policy_cls=ThinkyAgentPolicy,
            agent_ids=agent_ids,
            reset_name="thinky_policy_reset",
        )


class RandomAgentPolicy(NimAgentPolicyBase):
    pass


class RandomAgentsMultiPolicy(NimMultiAgentPolicy):
    short_names = ["nim_random"]

    def __init__(self, policy_env_info: PolicyEnvInterface, agent_ids: Sequence[int] | None = None):
        super().__init__(
            policy_env_info,
            handle_ctor=na.RandomPolicy,
            step_batch_name="random_policy_step_batch",
            agent_policy_cls=RandomAgentPolicy,
            agent_ids=agent_ids,
            reset_name="random_policy_reset",
        )


class RaceCarAgentPolicy(NimAgentPolicyBase):
    pass


class RaceCarAgentsMultiPolicy(NimMultiAgentPolicy):
    short_names = ["nim_race_car"]

    def __init__(self, policy_env_info: PolicyEnvInterface, agent_ids: Sequence[int] | None = None):
        super().__init__(
            policy_env_info,
            handle_ctor=na.RaceCarPolicy,
            step_batch_name="race_car_policy_step_batch",
            agent_policy_cls=RaceCarAgentPolicy,
            agent_ids=agent_ids,
            reset_name="race_car_policy_reset",
        )
