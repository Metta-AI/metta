"""CoGames policy namespace.

Provides backwards-compatible accessors for policy base classes that live in
``mettagrid.policy`` so existing string imports such as
``cogames.policy.StatefulPolicyImpl`` continue to resolve without reintroducing
``from … import …`` usage.  We also proxy access to legacy subpackages (e.g.
``cogames.policy.scripted_agent``) via ``__getattr__`` so import sites do not
need to change.
"""

import importlib

import mettagrid.policy.policy as policy_module

AgentPolicy = policy_module.AgentPolicy
StatefulAgentPolicy = policy_module.StatefulAgentPolicy
StatefulPolicyImpl = policy_module.StatefulPolicyImpl
TrainablePolicy = policy_module.TrainablePolicy
PolicySpec = policy_module.PolicySpec
MultiAgentPolicy = policy_module.MultiAgentPolicy
Policy = policy_module.MultiAgentPolicy


class MockPolicy:  # pragma: no cover - test helper
    """Placeholder policy used by legacy evaluation tests.

    The real implementation used to live in ``metta.agent.mocks`` but we only
    need a lightweight sentinel so that string-based imports resolve.
    """

    def __init__(self, *_, **__):
        pass


__all__ = [
    "AgentPolicy",
    "MockPolicy",
    "MultiAgentPolicy",
    "Policy",
    "PolicySpec",
    "StatefulAgentPolicy",
    "StatefulPolicyImpl",
    "TrainablePolicy",
    "scripted_agent",
]

_SUBMODULE_ALIASES = {
    "scripted_agent": "cogames.policy.scripted_agent",
}


def __getattr__(name: str):
    target = _SUBMODULE_ALIASES.get(name)
    if target is None:
        raise AttributeError(f"module 'cogames.policy' has no attribute '{name}'")
    module = importlib.import_module(target)
    globals()[name] = module
    return module
