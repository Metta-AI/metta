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

from mettagrid.config.mettagrid_config import MettaGridConfig

# Import environment classes and types
from mettagrid.core import MettaGridAction, MettaGridCore, MettaGridObservation

# Import other commonly used classes
from mettagrid.envs.gym_env import MettaGridGymEnv
from mettagrid.envs.mettagrid_env import MettaGridEnv
from mettagrid.envs.pettingzoo_env import MettaGridPettingZooEnv
from mettagrid.map_builder.map_builder import GameMap

# Import data types from C++ module (source of truth)
from mettagrid.mettagrid_c import (
    dtype_actions,
    dtype_masks,
    dtype_observations,
    dtype_rewards,
    dtype_success,
    dtype_terminals,
    dtype_truncations,
)

# Import type validation functions
from mettagrid.types import (
    get_action_nvec,
    get_observation_shape,
    validate_action_space,
    validate_observation_space,
)
from mettagrid.util.replay_writer import ReplayWriter
from mettagrid.util.stats_writer import StatsWriter

# Import mettascope submodule (optional, for visualization)
try:
    from mettagrid import mettascope
except (ImportError, OSError):
    # On headless machines the optional Nim binding may try to load libGL
    # and fail; degrade gracefully by disabling mettascope instead of crashing.
    mettascope = None

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
    "get_action_nvec",
    # Supporting classes
    "GameMap",
    "ReplayWriter",
    "StatsWriter",
    # Optional visualization module
    "mettascope",
]
