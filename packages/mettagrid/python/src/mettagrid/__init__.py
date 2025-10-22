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

from typing import TYPE_CHECKING, Dict, Tuple

# Map attribute names to (module, attribute) for lazy loading.
_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    # Config
    "MettaGridConfig": ("mettagrid.config.mettagrid_config", "MettaGridConfig"),
    # Core classes
    "Simulator": ("mettagrid.simulator", "Simulator"),
    "Action": ("mettagrid.simulator", "Action"),
    "Observation": ("mettagrid.simulator", "Observation"),
    # Supporting classes
    "GameMap": ("mettagrid.map_builder.map_builder", "GameMap"),
}

if TYPE_CHECKING:
    from mettagrid.config.mettagrid_config import MettaGridConfig
    from mettagrid.map_builder.map_builder import GameMap
    from mettagrid.simulator import Action, Observation, Simulator

__all__ = [
    # Config
    "MettaGridConfig",
    # Core classes
    "Simulator",
    "Action",
    "Observation",
    # Type definitions
    # Supporting classes
    "GameMap",
]


def __dir__() -> list[str]:
    """Expose lazy attributes in dir() results."""

    return sorted({*__all__, *globals().keys()})
