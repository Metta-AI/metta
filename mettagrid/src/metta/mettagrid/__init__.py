"""
MettaGrid - Multi-agent reinforcement learning grid environments.

This module provides various environment adapters for different RL frameworks:
- MettaGridCore: Core C++ wrapper (no training features)
- AutoResetEnv: Training environment with auto-reset (PufferLib-based with stats/replay)
- CurriculumEnv: Wrapper that adds curriculum support to AutoResetEnv
- MettaGridGymEnv: Gymnasium adapter
- MettaGridPettingZooEnv: PettingZoo adapter

All adapters inherit from MettaGridCore and provide framework-specific interfaces.
For PufferLib integration, use PufferLib's MettaPuff wrapper directly.
"""

from __future__ import annotations

from metta.mettagrid.auto_reset_env import AutoResetEnv
from metta.mettagrid.config import EnvConfig

# Import environment classes
from metta.mettagrid.core import MettaGridCore

# Import other commonly used classes
from metta.mettagrid.curriculum.core import Curriculum
from metta.mettagrid.curriculum_env import CurriculumEnv
from metta.mettagrid.gym_env import MettaGridGymEnv, SingleAgentMettaGridGymEnv
from metta.mettagrid.level_builder import LevelMap

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
from metta.mettagrid.pettingzoo_env import MettaGridPettingZooEnv
from metta.mettagrid.replay_writer import ReplayWriter
from metta.mettagrid.stats_writer import StatsWriter

__all__ = [
    # Core classes
    "MettaGridCore",
    # Main environments
    "AutoResetEnv",
    "CurriculumEnv",
    # Configuration
    "EnvConfig",
    # Environment adapters
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
    "LevelMap",
    "ReplayWriter",
    "StatsWriter",
]
