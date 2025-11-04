"""CoGames policy module - re-exports from mettagrid.policy."""

from mettagrid.policy.policy import (
    AgentPolicy,
    PolicySpec,
    StatefulAgentPolicy,
    StatefulPolicyImpl,
    TrainablePolicy,
)
from mettagrid.policy.policy import (
    MultiAgentPolicy as Policy,
)

from .nim_agent import NimScriptedPolicy

__all__ = [
    "AgentPolicy",
    "Policy",
    "PolicySpec",
    "StatefulAgentPolicy",
    "StatefulPolicyImpl",
    "TrainablePolicy",
    "NimScriptedPolicy",
]
