"""
MettaGridCore - Core Python wrapper for MettaGrid C++ environment.

This class provides the base functionality for all framework-specific adapters,
without any training-specific features or framework dependencies.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces

from metta.mettagrid.level_builder import Level
from metta.mettagrid.mettagrid_c import MettaGrid as MettaGridCpp
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config

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
        level: Level,
        game_config_dict: Dict[str, Any],
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Initialize core MettaGrid functionality.

        Args:
            level: Level to use for the environment
            game_config_dict: Game configuration dictionary
            render_mode: Rendering mode (None, "human", "miniscope")
            **kwargs: Additional arguments passed to subclasses
        """
        self._render_mode = render_mode
        self._level = level
        self._renderer = None
        self._map_labels: List[str] = level.labels
        self._current_seed: int = 0

        # Environment metadata
        self.labels: List[str] = []
        self._should_reset = False

        # Initialize renderer class if needed (before C++ env creation)
        if self._render_mode is not None:
            self._initialize_renderer()

        # Create C++ environment immediately
        self._c_env_instance: Optional[MettaGridCpp] = self._create_c_env(game_config_dict)

    @property
    def c_env(self) -> MettaGridCpp:
        """Get core environment instance, raising error if not initialized."""
        if self._c_env_instance is None:
            raise RuntimeError("Environment not initialized")
        return self._c_env_instance

    def _initialize_renderer(self) -> None:
        """Initialize renderer class based on render mode."""
        self._renderer = None
        self._renderer_class = None

        if self._render_mode == "human":
            from metta.mettagrid.renderer.nethack import NethackRenderer

            self._renderer_class = NethackRenderer
        elif self._render_mode == "miniscope":
            from metta.mettagrid.renderer.miniscope import MiniscopeRenderer

            self._renderer_class = MiniscopeRenderer

    def _create_c_env(self, game_config_dict: Dict[str, Any], seed: Optional[int] = None) -> MettaGridCpp:
        """
        Create a new MettaGridCpp instance.

        Args:
            game_config_dict: Game configuration dictionary
            seed: Random seed for environment

        Returns:
            New MettaGridCpp instance
        """
        level = self._level

        # Validate number of agents
        level_agents = np.count_nonzero(np.char.startswith(level.grid, "agent"))
        assert game_config_dict["num_agents"] == level_agents, (
            f"Number of agents {game_config_dict['num_agents']} does not match number of agents in map {level_agents}"
        )

        # Ensure we have a dict
        if not isinstance(game_config_dict, dict):
            raise ValueError(f"Expected dict for game config, got {type(game_config_dict)}")

        # Create C++ config
        try:
            c_cfg = from_mettagrid_config(game_config_dict)
        except Exception as e:
            logger.error(f"Error creating C++ config: {e}")
            logger.error(f"Game config: {game_config_dict}")
            raise e

        # Create C++ environment
        current_seed = seed if seed is not None else self._current_seed
        c_env = MettaGridCpp(c_cfg, level.grid.tolist(), current_seed)

        # Initialize renderer if needed
        if (
            self._render_mode is not None
            and self._renderer is None
            and hasattr(self, "_renderer_class")
            and self._renderer_class is not None
        ):
            self._renderer = self._renderer_class(c_env.object_type_names())

        return c_env

    def reset(self, game_config_dict: Dict[str, Any], seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment.

        Args:
            game_config_dict: Game configuration dictionary
            seed: Random seed

        Returns:
            Tuple of (observations, info)
        """
        # Recreate C++ environment with new config
        self._c_env_instance = self._create_c_env(game_config_dict, seed)

        # Update seed
        self._current_seed = seed or 0

        # Reset flags
        self._should_reset = False

        # Get initial observations from core environment
        obs, infos = self.c_env.reset()

        return obs, infos

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Execute one timestep of the environment dynamics with the given actions.

        Args:
            actions: A numpy array of shape (num_agents, 2) with dtype np.int32

        Returns:
            Tuple of (observations, rewards, terminals, truncations, infos)
        """
        # Execute step in core environment
        obs, rewards, terminals, truncations, _ = self.c_env.step(actions)

        # Check for episode completion
        infos = {}
        if terminals.all() or truncations.all():
            self._should_reset = True

        return obs, rewards, terminals, truncations, infos

    def render(self) -> Optional[str]:
        """Render the environment."""
        if self._renderer is None or self._c_env_instance is None:
            return None

        return self._renderer.render(self.c_env.current_step, self.c_env.grid_objects())

    def close(self) -> None:
        """Close the environment."""
        if self._c_env_instance is not None:
            # Clean up any resources if needed
            self._c_env_instance = None

    # Properties that expose C++ environment functionality
    @property
    def done(self) -> bool:
        """Check if environment needs reset."""
        return self._should_reset

    @property
    def render_mode(self) -> Optional[str]:
        """Get render mode."""
        return self._render_mode

    @property
    def core_env(self) -> Optional[MettaGridCpp]:
        """Get core environment instance."""
        return self._c_env_instance

    # Properties that delegate to core environment
    @property
    def max_steps(self) -> int:
        return self.c_env.max_steps

    @property
    def num_agents(self) -> int:
        return self.c_env.num_agents

    @property
    def obs_width(self) -> int:
        return self.c_env.obs_width

    @property
    def obs_height(self) -> int:
        return self.c_env.obs_height

    @property
    def map_width(self) -> int:
        return self.c_env.map_width

    @property
    def map_height(self) -> int:
        return self.c_env.map_height

    @property
    def _observation_space(self) -> spaces.Box:
        """Internal observation space - use single_observation_space for PufferEnv compatibility."""
        return self.c_env.observation_space

    @property
    def _action_space(self) -> spaces.MultiDiscrete:
        """Internal action space - use single_action_space for PufferEnv compatibility."""
        return self.c_env.action_space

    @property
    def action_names(self) -> List[str]:
        return self.c_env.action_names()

    @property
    def max_action_args(self) -> List[int]:
        action_args_array = self.c_env.max_action_args()
        return [int(x) for x in action_args_array]

    @property
    def object_type_names(self) -> List[str]:
        return self.c_env.object_type_names()

    @property
    def inventory_item_names(self) -> List[str]:
        return self.c_env.inventory_item_names()

    @property
    def feature_normalizations(self) -> Dict[int, float]:
        """Get feature normalizations from C++ environment."""
        # Check if the C++ environment has the direct method
        if hasattr(self.c_env, "feature_normalizations"):
            return self.c_env.feature_normalizations()
        else:
            # Fallback to extracting from feature_spec (slower)
            feature_spec = self.c_env.feature_spec()
            return {int(spec["id"]): float(spec["normalization"]) for spec in feature_spec.values()}

    @property
    def initial_grid_hash(self) -> int:
        return self.c_env.initial_grid_hash

    @property
    def action_success(self) -> List[bool]:
        return self.c_env.action_success()

    @property
    def global_features(self) -> List[Any]:
        """Global features for compatibility."""
        return []

    def get_observation_features(self) -> Dict[str, Dict]:
        """
        Build the features dictionary for initialize_to_environment.

        Returns:
            Dictionary mapping feature names to their properties
        """
        # Get feature spec from C++ environment
        feature_spec = self.c_env.feature_spec()

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
        return self.c_env.grid_objects()
