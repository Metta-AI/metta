"""Shared training type aliases to avoid circular imports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from metta.agent.policy import Policy as Policy
    from metta.agent.policy import PolicyArchitecture as PolicyArchitecture
else:  # pragma: no cover - runtime aliases only carry type information
    Policy = Any  # type: ignore[assignment]
    PolicyArchitecture = Any  # type: ignore[assignment]

__all__ = ["Policy", "PolicyArchitecture"]
