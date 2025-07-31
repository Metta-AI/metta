"""
MettaGrid - Multi-agent reinforcement learning grid environments.

This module provides various environment adapters for different RL frameworks:
- MettaGridCore: Core C++ wrapper (no training features)
- MettaGridEnv: Training environment (PufferLib-based with stats/replay)
- MettaGridPufferEnv: Clean PufferLib adapter for users
- MettaGridGymEnv: Gymnasium adapter
- MettaGridPettingZooEnv: PettingZoo adapter

All adapters inherit from MettaGridCore and provide framework-specific interfaces.
"""

from __future__ import annotations

# Import environment classes
from metta.mettagrid.core import MettaGridCore

# Import other commonly used classes
from metta.mettagrid.curriculum.core import Curriculum
from metta.mettagrid.gym_env import MettaGridGymEnv, SingleAgentMettaGridGymEnv
from metta.mettagrid.level_builder import Level

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
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.mettagrid.pettingzoo_env import MettaGridPettingZooEnv
from metta.mettagrid.puffer_env import MettaGridPufferEnv
from metta.mettagrid.replay_writer import ReplayWriter
from metta.mettagrid.stats_writer import StatsWriter

__all__ = [
    # Core classes
    "MettaGridCore",
    # Main environment (backward compatible)
    "MettaGridEnv",
    # Environment adapters
    "MettaGridPufferEnv",
    "MettaGridGymEnv",
    "SingleAgentMettaGridGymEnv",
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
    "Curriculum",
    "Level",
    "ReplayWriter",
    "StatsWriter",
]
