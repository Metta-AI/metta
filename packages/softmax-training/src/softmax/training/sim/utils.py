"""Deprecated helpers kept for backward compatibility.

Use :mod:`metta.shared.policy_registry` instead.
"""

from __future__ import annotations

from warnings import warn

from metta.shared.policy_registry import get_or_create_policy_ids

warn(
    "softmax.training.sim.utils is deprecated; import get_or_create_policy_ids from metta.shared.policy_registry",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["get_or_create_policy_ids"]
