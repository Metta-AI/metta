"""Backward-compatible MettaGridEnv using the new modular architecture."""

from __future__ import annotations

# Additional backward compatibility types
import os
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


# Adapter mapping for configurable environment selection (training-compatible adapters only)
_ADAPTER_MAP = {
    "puffer": MettaGridPufferEnv,
    "pettingzoo": MettaGridPettingZooEnv,
    "core": MettaGridCore,
}


class MettaGridEnv:
    """
    Dynamic environment adapter that selects the appropriate adapter class at runtime.

    The adapter is selected based on the METTAGRID_ADAPTER environment variable:
    - "puffer" (default): Returns MettaGridPufferEnv and maintains backward compatibility
    - "pettingzoo": Returns MettaGridPettingZooEnv
    - "core": Returns MettaGridCore

    Note: MettaGridGymEnv is available as a standalone class for Gymnasium research
    but is not part of the configurable system since it's not training-compatible.
    """

    def __new__(cls, *args, **kwargs):
        """Create instance of the selected adapter class."""
        adapter_name = os.environ.get("METTAGRID_ADAPTER", "puffer").lower()

        if adapter_name not in _ADAPTER_MAP:
            raise ValueError(f"Unknown adapter '{adapter_name}'. Available adapters: {list(_ADAPTER_MAP.keys())}")

        adapter_class = _ADAPTER_MAP[adapter_name]
        return adapter_class(*args, **kwargs)


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
