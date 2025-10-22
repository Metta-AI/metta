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

import logging
import os
from typing import TYPE_CHECKING, Dict, Tuple


def _configure_logging() -> None:
    """Configure default logging for mettagrid package.

    Sets up INFO level logging with a readable format. Can be overridden
    via METTAGRID_LOG_LEVEL environment variable.
    """
    log_level_str = os.environ.get("METTAGRID_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Ensure root logger has at least one handler
    if not logging.root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s - %(name)s - %(message)s"))
        logging.root.addHandler(handler)

    # Set level for root logger
    if logging.root.level == logging.NOTSET or logging.root.level > log_level:
        logging.root.setLevel(log_level)


# Configure logging when package is imported
_configure_logging()

# Map attribute names to (module, attribute) for lazy loading.
_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    # Config
    "MettaGridConfig": ("mettagrid.config.mettagrid_config", "MettaGridConfig"),
    "Config": ("mettagrid.config", "Config"),
    # Core classes
    "Simulator": ("mettagrid.simulator.simulator", "Simulator"),
    "Action": ("mettagrid.simulator.interface", "Action"),
    "Observation": ("mettagrid.simulator.interface", "AgentObservation"),
    # Environments
    "MettaGridEnv": ("mettagrid.envs.mettagrid_puffer_env", "MettaGridPufferEnv"),
    # Supporting classes
    "GameMap": ("mettagrid.map_builder.map_builder", "GameMap"),
}

if TYPE_CHECKING:
    from mettagrid.config.mettagrid_config import MettaGridConfig
    from mettagrid.map_builder.map_builder import GameMap
    from mettagrid.simulator import Action, Simulation
    from mettagrid.simulator import AgentObservation as Observation

__all__ = [
    # Config
    "MettaGridConfig",
    # Core classes
    "Simulation",
    "Action",
    "Observation",
    # Supporting classes
    "GameMap",
]


def __dir__() -> list[str]:
    """Expose lazy attributes in dir() results."""

    return sorted({*__all__, *globals().keys()})


def __getattr__(name: str):
    """Lazily import attributes on first access."""
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        import importlib

        module = importlib.import_module(module_name)
        attr = getattr(module, attr_name)
        # Cache it in the module's namespace
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
