"""
MettaGrid - Multi-agent reinforcement learning grid environments.

This module provides various environment adapters for different RL frameworks:
- MettaGridEnv: Base concrete environment (uses C++ MettaGrid directly)
- MettaGridGymEnv: Gymnasium adapter
- MettaGridPettingZooEnv: PettingZoo adapter
- MettaGridPufferEnv: PufferLib adapter
- MettaGridBaseEnv: Base environment class (alias for MettaGridEnv)

All adapters support the same core functionality including curriculum learning,
stats collection, and replay recording.
"""

from __future__ import annotations

# Import all environment adapters
from metta.mettagrid.base_env import MettaGridEnv as MettaGridBaseEnv

# Import other commonly used classes
from metta.mettagrid.curriculum.core import Curriculum
from metta.mettagrid.gym_env import MettaGridGymEnv, SingleAgentMettaGridGymEnv
from metta.mettagrid.level_builder import Level

# Import the main environment (our own concrete implementation)
from metta.mettagrid.base_env import (
    MettaGridEnv,
    dtype_actions,
    dtype_observations,
    dtype_rewards,
    dtype_terminals,
    dtype_truncations,
)

# Additional data types (for backward compatibility)
dtype_masks = dtype_terminals  # masks same as terminals
dtype_success = dtype_terminals  # success same as terminals
from metta.mettagrid.pettingzoo_env import MettaGridPettingZooEnv
from metta.mettagrid.puffer_env import MettaGridPufferEnv
from metta.mettagrid.replay_writer import ReplayWriter
from metta.mettagrid.stats_writer import StatsWriter

__all__ = [
    # Main environment (backward compatible)
    "MettaGridEnv",
    # Environment adapters
    "MettaGridBaseEnv",
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
