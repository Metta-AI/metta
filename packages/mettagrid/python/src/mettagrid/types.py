"""Validation functions for MettaGrid observation and action spaces.

Type definitions are in mettagrid.core (MettaGridObservation, MettaGridAction).
This module provides runtime validation for gymnasium spaces.
"""

import numpy as np
from gymnasium import spaces


def validate_observation_space(space: spaces.Space) -> None:
    """Validate that the observation space conforms to MettaGrid requirements.

    MettaGrid expects observations to be Box spaces with uint8 dtype.

    Args:
        space: The observation space to validate

    Raises:
        TypeError: If the space does not conform to expected type
    """
    if not isinstance(space, spaces.Box):
        raise TypeError(f"MettaGrid observation space must be Box, got {type(space).__name__}")
    if space.dtype != np.uint8:
        raise TypeError(f"MettaGrid observation space must have dtype uint8, got {space.dtype}")


def validate_action_space(space: spaces.Space) -> None:
    """Validate that the action space conforms to MettaGrid requirements.

    MettaGrid expects actions to be MultiDiscrete spaces.

    Args:
        space: The action space to validate

    Raises:
        TypeError: If the space does not conform to expected type
    """
    if not isinstance(space, spaces.MultiDiscrete):
        raise TypeError(f"MettaGrid action space must be MultiDiscrete, got {type(space).__name__}")


def get_observation_shape(space: spaces.Box) -> tuple[int, ...]:
    """Extract the shape of observations from the observation space.

    Args:
        space: The observation space

    Returns:
        Tuple representing the observation shape
    """
    return tuple(space.shape)


def get_action_nvec(space: spaces.MultiDiscrete) -> np.ndarray:
    """Extract the nvec array from the action space.

    Args:
        space: The action space

    Returns:
        Array of action dimension sizes
    """
    return space.nvec
