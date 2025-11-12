"""Compatibility shim for legacy prod benchmark recipes.

The original ViTReset policy config was removed when the ViT defaults were
refactored. Several prod benchmark recipes still import ViTResetConfig, so we
provide a thin wrapper around the current ViTDefaultConfig to avoid breaking
those entrypoints.
"""

from __future__ import annotations

from metta.agent.policies.vit import ViTDefaultConfig
from metta.agent.policy import PolicyArchitecture


class ViTResetConfig(ViTDefaultConfig):
    """Alias for the legacy ViTReset policy."""


def vit_reset_policy_config() -> PolicyArchitecture:
    """Factory retained for parity with older recipe code."""
    return ViTResetConfig()


__all__ = ["ViTResetConfig", "vit_reset_policy_config"]
