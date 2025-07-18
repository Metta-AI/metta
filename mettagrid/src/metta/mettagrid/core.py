"""
MettaGridCore - Stateless wrapper around C++ MettaGrid environment.

This class provides a thin wrapper around the C++ MettaGrid class that:
- Does not support reset() - environments must be recreated
- Requires explicit buffer management
- Provides a pure step/observation interface
- Serves as the base for all framework-specific adapters
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces

from metta.mettagrid.mettagrid_c import GameConfig as GameConfig_cpp
from metta.mettagrid.mettagrid_c import MettaGrid


class MettaGridCore:
    """
    Core wrapper around C++ MettaGrid environment.

    This class is stateless and does not support reset() - new instances
    must be created for each episode. It provides explicit buffer management
    and a pure step/observation interface.
    """

    def __init__(
        self,
        game_config: GameConfig_cpp,
        map_grid: List[List[str]],
        seed: int = 0,
        observation_buffer: Optional[np.ndarray] = None,
        terminal_buffer: Optional[np.ndarray] = None,
        truncation_buffer: Optional[np.ndarray] = None,
        reward_buffer: Optional[np.ndarray] = None,
    ):
        """
        Initialize MettaGridCore.

        Args:
            game_config: Game configuration
            map_grid: 2D grid as list of lists of strings
            seed: Random seed
            observation_buffer: Pre-allocated observation buffer (optional)
            terminal_buffer: Pre-allocated terminal buffer (optional)
            truncation_buffer: Pre-allocated truncation buffer (optional)
            reward_buffer: Pre-allocated reward buffer (optional)
        """
        self._c_env = MettaGrid(game_config, map_grid, seed)
        self._game_config = game_config
        self._map_grid = map_grid
        self._seed = seed

        # Add PufferLib compatibility attributes
        self.emulated = False  # Required by PufferLib vectorization

        # Store buffer references
        self._observation_buffer = observation_buffer
        self._terminal_buffer = terminal_buffer
        self._truncation_buffer = truncation_buffer
        self._reward_buffer = reward_buffer

        # If buffers provided, set them in C++ environment
        if all(buf is not None for buf in [observation_buffer, terminal_buffer, truncation_buffer, reward_buffer]):
            assert observation_buffer is not None
            assert terminal_buffer is not None
            assert truncation_buffer is not None
            assert reward_buffer is not None
            self._c_env.set_buffers(observation_buffer, terminal_buffer, truncation_buffer, reward_buffer)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            actions: Action array of shape (num_agents, 2) with dtype int32

        Returns:
            Tuple of (observations, rewards, terminals, truncations, infos)
        """
        if (
            self._observation_buffer is None
            or self._reward_buffer is None
            or self._terminal_buffer is None
            or self._truncation_buffer is None
        ):
            raise RuntimeError("Buffers must be set before stepping. Call set_buffers() first.")

        # Step the C++ environment
        self._c_env.step(actions)

        # Return buffer contents
        return (
            self._observation_buffer.copy(),
            self._reward_buffer.copy(),
            self._terminal_buffer.copy(),
            self._truncation_buffer.copy(),
            {},  # Empty info dict - framework adapters can add info
        )

    def get_initial_observations(self) -> np.ndarray:
        """
        Get initial observations after environment creation.

        Returns:
            Initial observation array
        """
        if self._observation_buffer is None:
            raise RuntimeError("Buffers must be set before getting observations. Call set_buffers() first.")

        obs, _ = self._c_env.reset()
        return obs

    def set_buffers(
        self,
        observation_buffer: np.ndarray,
        terminal_buffer: np.ndarray,
        truncation_buffer: np.ndarray,
        reward_buffer: np.ndarray,
    ) -> None:
        """
        Set buffers for environment step operations.

        Args:
            observation_buffer: Buffer for observations
            terminal_buffer: Buffer for terminal flags
            truncation_buffer: Buffer for truncation flags
            reward_buffer: Buffer for rewards
        """
        self._observation_buffer = observation_buffer
        self._terminal_buffer = terminal_buffer
        self._truncation_buffer = truncation_buffer
        self._reward_buffer = reward_buffer

        # Set buffers in C++ environment
        self._c_env.set_buffers(observation_buffer, terminal_buffer, truncation_buffer, reward_buffer)

    # Properties that expose C++ environment functionality
    @property
    def num_agents(self) -> int:
        return self._c_env.num_agents

    @property
    def max_steps(self) -> int:
        return self._c_env.max_steps

    @property
    def current_step(self) -> int:
        return self._c_env.current_step

    @property
    def obs_width(self) -> int:
        return self._c_env.obs_width

    @property
    def obs_height(self) -> int:
        return self._c_env.obs_height

    @property
    def map_width(self) -> int:
        return self._c_env.map_width

    @property
    def map_height(self) -> int:
        return self._c_env.map_height

    @property
    def observation_space(self) -> spaces.Box:
        return self._c_env.observation_space

    @property
    def action_space(self) -> spaces.MultiDiscrete:
        return self._c_env.action_space

    @property
    def action_names(self) -> List[str]:
        return self._c_env.action_names()

    @property
    def max_action_args(self) -> List[int]:
        action_args_array = self._c_env.max_action_args()
        return [int(x) for x in action_args_array]

    @property
    def object_type_names(self) -> List[str]:
        return self._c_env.object_type_names()

    @property
    def inventory_item_names(self) -> List[str]:
        return self._c_env.inventory_item_names()

    @property
    def feature_normalizations(self) -> Dict[int, float]:
        return self._c_env.feature_normalizations()

    @property
    def initial_grid_hash(self) -> int:
        return self._c_env.initial_grid_hash

    def get_episode_rewards(self) -> np.ndarray:
        return self._c_env.get_episode_rewards()

    def get_episode_stats(self) -> Dict[str, Any]:
        stats = self._c_env.get_episode_stats()
        return dict(stats)

    def get_observation_features(self) -> Dict[str, Dict]:
        """
        Get observation features for policy initialization.

        Returns:
            Dictionary mapping feature names to their properties
        """
        if hasattr(self._c_env, "feature_spec"):
            feature_spec = self._c_env.feature_spec()
            features = {}

            for feature_name, feature_info in feature_spec.items():
                feature_dict = {"id": feature_info["id"]}

                if "normalization" in feature_info:
                    feature_dict["normalization"] = feature_info["normalization"]

                features[feature_name] = feature_dict

            return features
        else:
            # Fallback for compatibility
            return {}

    def grid_objects(self) -> Dict[int, Dict[str, Any]]:
        """Get information about all grid objects."""
        return self._c_env.grid_objects()

    def get_agent_groups(self) -> np.ndarray:
        return self._c_env.get_agent_groups()

    @property
    def action_success(self) -> List[bool]:
        action_success_array = self._c_env.action_success()
        return [bool(x) for x in action_success_array]
