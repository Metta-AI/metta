#!/usr/bin/env python3
"""Shared configuration for MettaGrid demos.

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
    """Configuration for MettaGrid demos."""

    # Environment settings
    render_mode: Optional[str] = None  # None for headless/CI, "human" for visualization
    seed: int = 42  # Random seed for reproducibility

    # Training/rollout settings
    max_steps_quick: int = 100  # For quick CI tests
    max_steps_training: int = 256  # For short training demonstrations
    max_steps_rollout: int = 300  # For longer rollout sessions

    # Gym-specific settings
    gym_num_agents: int = 1
    gym_num_vec_envs: int = 4  # For vectorized environment demos

    # PettingZoo-specific settings
    pettingzoo_is_training: bool = True

    # Puffer-specific settings
    puffer_num_agents: int = 24
    puffer_is_training: bool = True

    # Gym-specific algorithm settings
    gym_sb3_policy: str = "MlpPolicy"  # SB3 policy type
    gym_sb3_verbose: int = 0  # SB3 verbosity level
    gym_test_steps_divisor: int = 2  # Divisor for test steps calculation

    # PettingZoo-specific algorithm settings
    pettingzoo_api_test_cycles: int = 2  # Number of cycles for API compliance test
    pettingzoo_learning_rate: float = 1.1  # Learning rate multiplier for simple training

    # Puffer-specific algorithm settings
    puffer_action_bins: int = 10  # Number of action bins for Box action spaces
    puffer_learning_rate: float = 1.1  # Learning rate multiplier

    # Display settings
    separator_short: int = 60  # Length for short separators (=== * n)
    separator_long: int = 80  # Length for long separators (=== * n)

    # Formatting settings
    reward_precision: int = 2  # Decimal places for reward display (.2f)
    avg_reward_precision: int = 3  # Decimal places for average reward display (.3f)
    time_precision: int = 1  # Decimal places for time display (.1f)

    # NumPy settings
    numpy_dtype_int32: str = "int32"  # Standard int32 dtype string

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

    @property
    def np_int32(self):
        """Get numpy int32 dtype as string."""
        return self.numpy_dtype_int32


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
            pettingzoo_api_test_cycles=1,  # Fewer cycles for faster CI
        )

    @staticmethod
    def interactive():
        """Configuration for interactive demos with rendering."""
        return DemoConfig(
            render_mode="human",
            max_steps_quick=200,
            max_steps_training=500,
            max_steps_rollout=1000,
            gym_sb3_verbose=1,  # More verbose for interactive mode
            reward_precision=3,  # Higher precision for better visibility
            avg_reward_precision=4,
            time_precision=2,
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
            puffer_action_bins=20,  # More granular action spaces
        )
