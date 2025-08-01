"""
MettaGrid - Multi-agent reinforcement learning grid environments.

This module provides various environment adapters for different RL frameworks:
- MettaGridEnv: PufferLib adapter (default, backward compatible)
- MettaGridGymEnv: Gymnasium adapter
- MettaGridPettingZooEnv: PettingZoo adapter
- MettaGridCore: Low-level core environment
- MettaGridBaseEnv: Base environment class

All adapters support the same core functionality including curriculum learning,
stats collection, and replay recording.
"""

from __future__ import annotations

# Import all environment adapters
from metta.mettagrid.base_env import MettaGridEnv as MettaGridBaseEnv
from metta.mettagrid.core import MettaGridCore

# Import other commonly used classes
from metta.mettagrid.curriculum.curriculum import Curriculum
from metta.mettagrid.gym_env import MettaGridGymEnv, SingleAgentMettaGridGymEnv
from metta.mettagrid.level_builder import Level

# Import the main backward-compatible environment
# Import data types
from metta.mettagrid.mettagrid_env import (
    MettaGridEnv,
    dtype_actions,
    dtype_masks,
    dtype_observations,
    dtype_rewards,
    dtype_success,
    dtype_terminals,
    dtype_truncations,
)
from metta.mettagrid.pettingzoo_env import MettaGridPettingZooEnv
from metta.mettagrid.puffer_env import MettaGridPufferEnv
from metta.mettagrid.replay_writer import ReplayWriter
from metta.mettagrid.stats_writer import StatsWriter

__all__ = [
    # Main environment (backward compatible)
    "MettaGridEnv",
    # Environment adapters
    "MettaGridBaseEnv",
    "MettaGridCore",
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
