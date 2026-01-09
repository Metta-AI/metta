"""Policy interfaces and implementations for MettaGrid."""

from mettagrid.policy.loader import (
    discover_and_register_policies,
    initialize_or_load_policy,
    resolve_policy_class_path,
)
from mettagrid.policy.policy import (
    AgentPolicy,
    MultiAgentPolicy,
    NimMultiAgentPolicy,
    PolicySpec,
    StatefulAgentPolicy,
    StatefulPolicyImpl,
)
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.policy.policy_registry import get_policy_registry

__all__ = [
    # Core policy classes
    "AgentPolicy",
    "MultiAgentPolicy",
    "NimMultiAgentPolicy",
    "PolicySpec",
    "StatefulAgentPolicy",
    "StatefulPolicyImpl",
    # Policy environment interface
    "PolicyEnvInterface",
    # Policy loader utilities
    "discover_and_register_policies",
    "initialize_or_load_policy",
    "resolve_policy_class_path",
    # Policy registry
    "get_policy_registry",
]
