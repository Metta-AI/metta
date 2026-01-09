"""Policy interfaces and implementations for MettaGrid.

This package contains policy classes and utilities for MettaGrid agents.
Import directly from submodules to avoid circular dependencies:

    from mettagrid.policy.policy import MultiAgentPolicy, PolicySpec
    from mettagrid.policy.loader import initialize_or_load_policy
    from mettagrid.policy.policy_env_interface import PolicyEnvInterface
"""

# This file intentionally contains no imports to avoid circular dependencies.
# All imports should be done directly from submodules.

__all__ = [
    # Note: Items in __all__ are not imported here to avoid circular imports.
    # Import them directly from their respective modules:
    # - policy: AgentPolicy, MultiAgentPolicy, NimMultiAgentPolicy, PolicySpec, etc.
    # - loader: discover_and_register_policies, initialize_or_load_policy, etc.
    # - policy_env_interface: PolicyEnvInterface
    # - policy_registry: get_policy_registry
]
