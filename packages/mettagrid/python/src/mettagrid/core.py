"""MettaGridCore - Core Python wrapper for MettaGrid C++ environment.

This class provides the base functionality for all framework-specific adapters,
without any training-specific features or framework dependencies."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces

from mettagrid.config.mettagrid_c_config import from_mettagrid_config
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.mettagrid_c import MettaGrid as MettaGridCpp
from mettagrid.mettagrid_c import (
    dtype_actions,
    dtype_observations,
    dtype_rewards,
    dtype_terminals,
    dtype_truncations,
)

# Type compatibility assertions - ensure C++ types match PufferLib expectations
# PufferLib expects particular datatypes - see pufferlib/vector.py
assert dtype_observations == np.dtype(np.uint8)
assert dtype_terminals == np.dtype(np.bool_)
assert dtype_truncations == np.dtype(np.bool_)
assert dtype_rewards == np.dtype(np.float32)
assert dtype_actions == np.dtype(np.int32)

if TYPE_CHECKING:
    # Import EpisodeStats type for type checking only
    # This avoids circular imports at runtime while still providing type hints
    from mettagrid.mettagrid_c import EpisodeStats

logger = logging.getLogger("MettaGridCore")


class MettaGridCore:
    """
    Core MettaGrid functionality without any training features.

    This class provides pure C++ wrapper functionality without training-specific
    features like stats writing, replay writing, or curriculum management.
    Use MettaGridEnv for training functionality.
    """

    def __init__(
        self,
        mg_config: MettaGridConfig,
        render_mode: Optional[str] = None,
    ):
        """Initialize core MettaGrid functionality."""
        if not isinstance(mg_config, MettaGridConfig):
            raise ValueError("mg_config must be an instance of MettaGridConfig")

        # We protect the env config with __ to avoid accidental modification
        # by subclasses. It should only be modified through set_mg_config.
        self.__mg_config = mg_config
        self._render_mode = render_mode
        self._renderer = None
        self._current_seed: int = 0
        self._map_builder = self.__mg_config.game.map_builder.create()

        # Set by PufferBase
        self.observations: np.ndarray
        self.terminals: np.ndarray
        self.truncations: np.ndarray
        self.rewards: np.ndarray

        # Initialize renderer class if needed (before C++ env creation)
        if self._render_mode is not None:
            self._initialize_renderer()

        self.__c_env_instance: MettaGridCpp = self._create_c_env()
        self._update_core_buffers()

    @property
    def mg_config(self) -> MettaGridConfig:
        """Get the environment configuration."""
        return self.__mg_config

    def set_mg_config(self, mg_config: MettaGridConfig) -> None:
        """Set the environment configuration."""
        self.__mg_config = mg_config
        self._map_builder = self.__mg_config.game.map_builder.create()

    @property
    def c_env(self) -> MettaGridCpp:
        """Get core environment instance, raising error if not initialized."""
        if self.__c_env_instance is None:
            raise RuntimeError("Environment not initialized")
        return self.__c_env_instance

    def _initialize_renderer(self) -> None:
        """Initialize renderer class based on render mode."""
        self._renderer = None
        self._renderer_class = None
        self._renderer_native = False
        if self._render_mode == "human":
            from mettagrid.renderer.nethack import NethackRenderer

            self._renderer_class = NethackRenderer
        elif self._render_mode == "miniscope":
            from mettagrid.renderer.miniscope import MiniscopeRenderer

            self._renderer_class = MiniscopeRenderer

    def _create_c_env(self) -> MettaGridCpp:
        game_map = self._map_builder.build()

        # Validate number of agents
        level_agents = np.count_nonzero(np.char.startswith(game_map.grid, "agent"))
        assert self.__mg_config.game.num_agents == level_agents, (
            f"Number of agents {self.__mg_config.game.num_agents} "
            f"does not match number  of agents in map {level_agents}"
        )
        game_config_dict = self.__mg_config.game.model_dump()

        # Create C++ config
        try:
            c_cfg = from_mettagrid_config(game_config_dict)
        except Exception as e:
            logger.error(f"Error creating C++ config: {e}")
            logger.error(f"Game config: {game_config_dict}")
            raise e

        # Create C++ environment
        c_env = MettaGridCpp(c_cfg, game_map.grid.tolist(), self._current_seed)
        self._update_core_buffers()

        # Initialize renderer if needed
        if (
            self._render_mode is not None
            and self._renderer is None
            and hasattr(self, "_renderer_class")
            and self._renderer_class is not None
        ):
            if self._renderer_native:
                self._renderer = self._renderer_class()
            else:
                self._renderer = self._renderer_class(c_env.object_type_names())

        self.__c_env_instance = c_env
        return c_env

    def _update_core_buffers(self) -> None:
        if hasattr(self, "observations") and self.observations is not None:
            self.__c_env_instance.set_buffers(self.observations, self.terminals, self.truncations, self.rewards)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._current_seed = seed

        # Recreate C++ environment with new config
        self._create_c_env()
        self._update_core_buffers()

        # Get initial observations from core environment
        obs, infos = self.__c_env_instance.reset()

        return obs, infos

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Execute one timestep of the environment dynamics with the given actions."""
        # Execute step in core environment
        return self.__c_env_instance.step(actions)

    def render(self) -> Optional[str]:
        """Render the environment."""
        if self._renderer is None or self.__c_env_instance is None:
            return None

        return self._renderer.render(self.__c_env_instance.current_step, self.__c_env_instance.grid_objects())

    def close(self) -> None:
        """Close the environment."""
        del self.__c_env_instance

    def get_episode_rewards(self) -> np.ndarray:
        """Get the episode rewards."""
        return self.__c_env_instance.get_episode_rewards()

    def get_episode_stats(self) -> EpisodeStats:
        """Get the episode stats."""
        return self.__c_env_instance.get_episode_stats()

    @property
    def render_mode(self) -> Optional[str]:
        """Get render mode."""
        return self._render_mode

    @property
    def core_env(self) -> Optional[MettaGridCpp]:
        """Get core environment instance."""
        return self.__c_env_instance

    # Properties that delegate to core environment
    @property
    def max_steps(self) -> int:
        return self.__c_env_instance.max_steps

    @property
    def num_agents(self) -> int:
        return self.__c_env_instance.num_agents

    @property
    def obs_width(self) -> int:
        return self.__c_env_instance.obs_width

    @property
    def obs_height(self) -> int:
        return self.__c_env_instance.obs_height

    @property
    def map_width(self) -> int:
        return self.__c_env_instance.map_width

    @property
    def map_height(self) -> int:
        return self.__c_env_instance.map_height

    @property
    def _observation_space(self) -> spaces.Box:
        """Internal observation space - use single_observation_space for PufferEnv compatibility."""
        return self.__c_env_instance.observation_space

    @property
    def _action_space(self) -> spaces.MultiDiscrete:
        """Internal action space - use single_action_space for PufferEnv compatibility."""
        return self.__c_env_instance.action_space

    @property
    def action_names(self) -> List[str]:
        return self.__c_env_instance.action_names()

    @property
    def max_action_args(self) -> List[int]:
        action_args_array = self.__c_env_instance.max_action_args()
        return [int(x) for x in action_args_array]

    @property
    def object_type_names(self) -> List[str]:
        return self.__c_env_instance.object_type_names()

    @property
    def resource_names(self) -> List[str]:
        return self.__c_env_instance.resource_names()

    @property
    def feature_normalizations(self) -> Dict[int, float]:
        """Get feature normalizations from C++ environment."""
        # Check if the C++ environment has the direct method
        if hasattr(self.__c_env_instance, "feature_normalizations"):
            return self.__c_env_instance.feature_normalizations()
        else:
            # Fallback to extracting from feature_spec (slower)
            feature_spec = self.__c_env_instance.feature_spec()
            return {int(spec["id"]): float(spec["normalization"]) for spec in feature_spec.values()}

    @property
    def initial_grid_hash(self) -> int:
        return self.__c_env_instance.initial_grid_hash

    @property
    def action_success(self) -> List[bool]:
        return self.__c_env_instance.action_success()

    def get_observation_features(self) -> Dict[str, Dict]:
        """Build the features dictionary for initialize_to_environment."""
        # Get feature spec from C++ environment
        feature_spec = self.__c_env_instance.feature_spec()

        features = {}
        for feature_name, feature_info in feature_spec.items():
            feature_dict: Dict[str, Any] = {"id": feature_info["id"]}

            # Add normalization if present
            if "normalization" in feature_info:
                feature_dict["normalization"] = feature_info["normalization"]

            features[feature_name] = feature_dict

        return features

    @property
    def grid_objects(self) -> Dict[int, Dict[str, Any]]:
        """Get grid objects information."""
        return self.__c_env_instance.grid_objects()
