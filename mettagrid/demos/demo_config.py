#!/usr/bin/env python3
"""Shared configuration for MettagGrid demos.

This module provides centralized configuration for all demo scripts to ensure
consistency and make maintenance easier. Configuration can be overridden as needed
for specific demos.
"""

from dataclasses import dataclass
from typing import Optional

from metta.mettagrid.builder.envs import make_arena, make_navigation
from metta.mettagrid.mettagrid_config import MettaGridConfig


@dataclass
class DemoConfig:
    """Configuration for MettagGrid demos."""

    # Environment settings
    render_mode: Optional[str] = None  # None for headless/CI, "human" for visualization
    seed: int = 42  # Random seed for reproducibility

    # Training/rollout settings
    max_steps_quick: int = 100  # For quick CI tests
    max_steps_training: int = 256  # For short training demonstrations
    max_steps_rollout: int = 300  # For longer rollouts

    # Gym-specific settings
    gym_num_agents: int = 1
    gym_num_vec_envs: int = 4  # For vectorized environment demos

    # PettingZoo-specific settings
    pettingzoo_is_training: bool = True

    # Puffer-specific settings
    puffer_num_agents: int = 24
    puffer_is_training: bool = True

    def get_gym_config(self):
        """Get configuration for Gym environment."""
        # Use navigation config for single-agent compatibility
        return make_navigation(num_agents=self.gym_num_agents)

    def get_pettingzoo_config(self):
        """Get configuration for PettingZoo environment."""
        return MettaGridConfig()

    def get_puffer_config(self):
        """Get configuration for Puffer environment."""
        return make_arena(num_agents=self.puffer_num_agents)


# Default configuration instance
DEFAULT_CONFIG = DemoConfig()


# Demo-specific presets
class DemoPresets:
    """Predefined configurations for different demo scenarios."""

    @staticmethod
    def ci_quick_test():
        """Configuration for quick CI tests."""
        return DemoConfig(
            max_steps_quick=50,
            max_steps_training=100,
            max_steps_rollout=150,
        )

    @staticmethod
    def interactive():
        """Configuration for interactive demos with rendering."""
        return DemoConfig(
            render_mode="human",
            max_steps_quick=200,
            max_steps_training=500,
            max_steps_rollout=1000,
        )

    @staticmethod
    def benchmark():
        """Configuration for performance benchmarking."""
        return DemoConfig(
            max_steps_quick=1000,
            max_steps_training=10000,
            max_steps_rollout=5000,
            gym_num_vec_envs=8,
            puffer_num_agents=48,
        )
