"""Shared configuration utilities for mettagrid demos.

This module provides a single, shared configuration that works across
all demo environments (Gymnasium, PettingZoo, PufferLib).
"""

from metta.mettagrid.config import EnvConfig, arena


def create_demo_config(**overrides) -> EnvConfig:
    """Create a demo EnvConfig that works for all demo environments.

    This single configuration is optimized to work well across all training
    environments: Gymnasium (single-agent), PettingZoo (multi-agent), and
    PufferLib (vectorized training).

    Args:
        **overrides: Any parameter overrides to customize the config

    Returns:
        EnvConfig ready for use in any demo environment
    """
    # Default parameters that work well for all demo types
    defaults = {
        "num_agents": 2,  # Works for both single-agent (uses 1) and multi-agent
        "combat": False,  # Keep it simple by default
        "max_steps": 80,  # Reasonable episode length
        "map_width": 10,  # Good size for visualization and learning
        "map_height": 10,
        "obs_width": 5,  # Standard observation window
        "obs_height": 5,
    }

    # Apply any overrides
    defaults.update(overrides)

    return arena(**defaults)
