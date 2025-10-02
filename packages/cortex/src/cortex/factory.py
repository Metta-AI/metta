from __future__ import annotations

from typing import Any

from cortex.config import CortexStackConfig
from cortex.stacks import CortexStack


def build_cortex(config: CortexStackConfig) -> CortexStack:
    """Instantiate a cortex stack from configuration."""
    return CortexStack(config)


def build_from_dict(data: dict[str, Any]) -> CortexStack:
    """Convenience helper to build from a raw dict."""
    cfg = CortexStackConfig(**data)
    return build_cortex(cfg)


__all__ = ["build_cortex", "build_from_dict"]
