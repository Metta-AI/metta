"""Backward-compatible MettaGridEnv using the new modular architecture."""

from __future__ import annotations

# Additional backward compatibility types
import numpy as np

# Import all the new adapters for easy access
from metta.mettagrid.base_env import MettaGridEnv as MettaGridBaseEnv
from metta.mettagrid.core import MettaGridCore
from metta.mettagrid.gym_env import MettaGridGymEnv, SingleAgentMettaGridGymEnv
from metta.mettagrid.pettingzoo_env import MettaGridPettingZooEnv

# Import data types for backward compatibility
from metta.mettagrid.puffer_env import (
    MettaGridPufferEnv,
    dtype_actions,
    dtype_observations,
    dtype_rewards,
    dtype_terminals,
    dtype_truncations,
)

# Import the new PufferLib adapter

dtype_masks = np.dtype(bool)
dtype_success = np.dtype(bool)


# Legacy function decorator for backward compatibility
def required(func):
    """Marks methods that PufferEnv requires but does not implement for override."""
    return func


# Backward compatibility aliases
class MettaGridEnv(MettaGridPufferEnv):
    """
    Main MettaGridEnv class - backward compatible with PufferLib adapter.

    This class is now a thin wrapper around MettaGridPufferEnv to maintain
    backward compatibility while using the new modular architecture.
    """

    pass


# All existing functionality is now inherited from MettaGridPufferEnv
# This provides full backward compatibility for existing code

# Export all the available environment types
__all__ = [
    "MettaGridEnv",
    "MettaGridBaseEnv",
    "MettaGridCore",
    "MettaGridPufferEnv",
    "MettaGridGymEnv",
    "SingleAgentMettaGridGymEnv",
    "MettaGridPettingZooEnv",
    "dtype_actions",
    "dtype_observations",
    "dtype_rewards",
    "dtype_terminals",
    "dtype_truncations",
    "dtype_masks",
    "dtype_success",
    "required",
]
