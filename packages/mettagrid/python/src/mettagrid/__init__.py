"""
MettaGrid - Multi-agent reinforcement learning grid environments.

This module provides various environment adapters for different RL frameworks:
- MettaGridCore: Core C++ wrapper (no training features)
- MettaGridEnv: Training environment (PufferLib-based with stats/replay)
- MettaGridGymEnv: Gymnasium adapter
- MettaGridPettingZooEnv: PettingZoo adapter

All adapters inherit from MettaGridCore and provide framework-specific interfaces.
For PufferLib integration, use PufferLib's MettaPuff wrapper directly.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Dict, Tuple

# Map attribute names to (module, attribute) for lazy loading.
_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    # Config
    "MettaGridConfig": ("mettagrid.config.mettagrid_config", "MettaGridConfig"),
    # Core classes
    "MettaGridCore": ("mettagrid.core", "MettaGridCore"),
    "MettaGridAction": ("mettagrid.core", "MettaGridAction"),
    "MettaGridObservation": ("mettagrid.core", "MettaGridObservation"),
    # Main environment (backward compatible)
    "MettaGridEnv": ("mettagrid.envs.mettagrid_env", "MettaGridEnv"),
    # Environment adapters
    "MettaGridPettingZooEnv": (
        "mettagrid.envs.pettingzoo_env",
        "MettaGridPettingZooEnv",
    ),
    # Data types (from C++)
    "dtype_actions": ("mettagrid.mettagrid_c", "dtype_actions"),
    "dtype_observations": ("mettagrid.mettagrid_c", "dtype_observations"),
    "dtype_rewards": ("mettagrid.mettagrid_c", "dtype_rewards"),
    "dtype_terminals": ("mettagrid.mettagrid_c", "dtype_terminals"),
    "dtype_truncations": ("mettagrid.mettagrid_c", "dtype_truncations"),
    "dtype_masks": ("mettagrid.mettagrid_c", "dtype_masks"),
    "dtype_success": ("mettagrid.mettagrid_c", "dtype_success"),
    # Type definitions
    "validate_observation_space": (
        "mettagrid.types",
        "validate_observation_space",
    ),
    "validate_action_space": ("mettagrid.types", "validate_action_space"),
    "get_observation_shape": ("mettagrid.types", "get_observation_shape"),
    "get_action_count": ("mettagrid.types", "get_action_count"),
    # Supporting classes
    "GameMap": ("mettagrid.map_builder.map_builder", "GameMap"),
    "StatsWriter": ("mettagrid.util.stats_writer", "StatsWriter"),
    "RenderMode": ("mettagrid.envs.mettagrid_env", "RenderMode"),
}

if TYPE_CHECKING:
    from mettagrid.config.mettagrid_config import MettaGridConfig
    from mettagrid.core import MettaGridAction, MettaGridCore, MettaGridObservation
    from mettagrid.envs.mettagrid_env import MettaGridEnv, RenderMode
    from mettagrid.envs.pettingzoo_env import MettaGridPettingZooEnv
    from mettagrid.map_builder.map_builder import GameMap
    from mettagrid.mettagrid_c import (
        dtype_actions,
        dtype_masks,
        dtype_observations,
        dtype_rewards,
        dtype_success,
        dtype_terminals,
        dtype_truncations,
    )
    from mettagrid.types import (
        get_action_count,
        get_observation_shape,
        validate_action_space,
        validate_observation_space,
    )
    from mettagrid.util.stats_writer import StatsWriter

    try:
        from mettagrid import mettascope
    except (ImportError, OSError):
        mettascope = None

__all__ = [
    # Config
    "MettaGridConfig",
    # Core classes
    "MettaGridCore",
    "MettaGridEnv",
    # Environment adapters
    "MettaGridPettingZooEnv",
    # Data types (from C++)
    "dtype_actions",
    "dtype_observations",
    "dtype_rewards",
    "dtype_terminals",
    "dtype_truncations",
    "dtype_masks",
    "dtype_success",
    # Type definitions
    "MettaGridObservation",
    "MettaGridAction",
    "validate_observation_space",
    "validate_action_space",
    "get_observation_shape",
    "get_action_count",
    # Supporting classes
    "GameMap",
    "StatsWriter",
    "RenderMode",
    # Optional visualization module
    "mettascope",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - thin import wrapper
    """Dynamically import public attributes on first access."""

    if name == "mettascope":
        try:
            module = importlib.import_module("mettagrid.mettascope")
        except (ImportError, OSError):
            module = None
        globals()[name] = module
        return module

    try:
        module_name, attribute_name = _LAZY_ATTRS[name]
    except KeyError as exc:  # pragma: no cover - mirrors built-in behaviour
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = importlib.import_module(module_name)
    attr = getattr(module, attribute_name)
    globals()[name] = attr
    return attr


def __dir__() -> list[str]:
    """Expose lazy attributes in dir() results."""

    return sorted({*__all__, *globals().keys()})
