"""Factory functions for building Cortex stacks from configuration."""

from __future__ import annotations

from typing import Any

from cortex.config import CortexStackConfig
from cortex.stacks import CortexStack
from cortex.utils import configure_tf32_precision


def build_cortex(config: CortexStackConfig) -> CortexStack:
    """Build Cortex stack from configuration object."""
    configure_tf32_precision()
    return CortexStack(config)


def build_from_dict(data: dict[str, Any]) -> CortexStack:
    """Build Cortex stack from dictionary configuration."""
    cfg = CortexStackConfig(**data)
    return build_cortex(cfg)


__all__ = ["build_cortex", "build_from_dict"]
