"""CoGames policy module - re-exports from mettagrid.policy."""

from mettagrid.policy.policy import (
    AgentPolicy,
    PolicySpec,
    StatefulAgentPolicy,
    StatefulPolicyImpl,
    TrainablePolicy,
)
from mettagrid.policy.policy import MultiAgentPolicy as Policy

from .pufferlib_policy import PufferlibCogsPolicy

__all__ = [
    "AgentPolicy",
    "Policy",
    "PolicySpec",
    "PufferlibCogsPolicy",
    "StatefulAgentPolicy",
    "StatefulPolicyImpl",
    "TrainablePolicy",
]
