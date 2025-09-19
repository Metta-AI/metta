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

# Import environment classes
from metta.mettagrid.core import MettaGridCore

# Import other commonly used classes
from metta.mettagrid.gym_env import MettaGridGymEnv
from metta.mettagrid.map_builder.map_builder import GameMap

# Import data types from C++ module (source of truth)
from metta.mettagrid.mettagrid_c import (
    dtype_actions,
    dtype_masks,
    dtype_observations,
    dtype_rewards,
    dtype_success,
    dtype_terminals,
    dtype_truncations,
)
from metta.mettagrid.mettagrid_config import MettaGridConfig
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.mettagrid.pettingzoo_env import MettaGridPettingZooEnv
from metta.mettagrid.replay_writer import ReplayWriter
from metta.mettagrid.stats_writer import StatsWriter

__all__ = [
    # Config
    "MettaGridConfig",
    # Core classes
    "MettaGridCore",
    # Main environment (backward compatible)
    "MettaGridEnv",
    # Environment adapters
    "MettaGridGymEnv",
    "MettaGridPettingZooEnv",
    # Data types
    "dtype_actions",
    "dtype_observations",
    "dtype_rewards",
    "dtype_terminals",
    "dtype_truncations",
    "dtype_masks",
    "dtype_success",
    # Supporting classes
    "GameMap",
    "ReplayWriter",
    "StatsWriter",
]
