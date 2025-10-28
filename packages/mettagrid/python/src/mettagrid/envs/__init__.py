"""Environment wrappers for mettagrid."""

from .mettagrid_puffer_env import MettaGridPufferEnv

# Backward compatibility alias
MettaGridEnv = MettaGridPufferEnv

__all__ = [
    "MettaGridEnv",
    "MettaGridPufferEnv",
]
