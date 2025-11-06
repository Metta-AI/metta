"""Factory functions for building Cortex stacks from configuration."""

import typing

import cortex.config
import cortex.stacks
import cortex.utils


def build_cortex(config: cortex.config.CortexStackConfig) -> cortex.stacks.CortexStack:
    """Build Cortex stack from configuration object."""
    cortex.utils.configure_tf32_precision()
    return cortex.stacks.CortexStack(config)


def build_from_dict(data: dict[str, typing.Any]) -> cortex.stacks.CortexStack:
    """Build Cortex stack from dictionary configuration."""
    cfg = cortex.config.CortexStackConfig(**data)
    return build_cortex(cfg)


__all__ = ["build_cortex", "build_from_dict"]
