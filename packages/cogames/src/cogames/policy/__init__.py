"""Policy interfaces and implementations for CoGames."""

from cogames.policy.lstm import LSTMPolicy
from cogames.policy.policy import (
    AgentPolicy,
    Policy,
    StatefulAgentPolicy,
    StatefulPolicyImpl,
    TrainablePolicy,
)
from cogames.policy.random import RandomPolicy
from cogames.policy.registry import POLICY_SHORTCUTS, resolve_policy_class_path
from cogames.policy.simple import SimplePolicy

__all__ = [
    "AgentPolicy",
    "Policy",
    "StatefulAgentPolicy",
    "StatefulPolicyImpl",
    "TrainablePolicy",
    "LSTMPolicy",
    "RandomPolicy",
    "SimplePolicy",
    "POLICY_SHORTCUTS",
    "resolve_policy_class_path",
]
