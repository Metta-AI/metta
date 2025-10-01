"""Policy registry utilities for resolving shorthand names."""

from __future__ import annotations

from typing import Dict

POLICY_SHORTCUTS: Dict[str, str] = {
    "random": "cogames.policy.random.RandomPolicy",
    "simple": "cogames.policy.simple.SimplePolicy",
    "lstm": "cogames.policy.lstm.LSTMPolicy",
}


def resolve_policy_class_path(policy: str) -> str:
    """Resolve a policy shorthand into a fully-qualified class path."""
    return POLICY_SHORTCUTS.get(policy, policy)


__all__ = ["POLICY_SHORTCUTS", "resolve_policy_class_path"]
