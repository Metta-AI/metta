"""MettaGridCore - Core Python wrapper for MettaGrid C++ environment.

This class provides the base functionality for all framework-specific adapters,
without any training-specific features or framework dependencies."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
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


# MettaGrid Type Definitions
# Observations are token-based: shape (num_tokens, 3) where each token is [PackedCoordinate, key, value]
# - PackedCoordinate: uint8 packed (x, y) coordinate
# - key: uint8 feature key
# - value: uint8 feature value
MettaGridObservation = npt.NDArray[np.uint8]  # Shape: (num_tokens, 3)

# Actions are Discrete: single integer index representing unique action choices
MettaGridAction = npt.NDArray[np.int32]  # Shape: ()


@dataclass
class ObsFeature:
    id: int
    normalization: float
    name: str


@dataclass
class BoundingBox:
    min_row: int
    max_row: int
    min_col: int
    max_col: int


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
    ):
        """Initialize core MettaGrid functionality."""
        if not isinstance(mg_config, MettaGridConfig):
            raise ValueError("mg_config must be an instance of MettaGridConfig")

        # We protect the env config with __ to avoid accidental modification
        # by subclasses. It should only be modified through set_mg_config.
        self.__mg_config = mg_config
        self._renderer = None
        self._current_seed: int = 0

        self._map_builder = self.__mg_config.game.map_builder.create()

        # Set by PufferBase
        self.observations: np.ndarray
        self.terminals: np.ndarray
        self.truncations: np.ndarray
        self.rewards: np.ndarray

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

    def _create_c_env(self) -> MettaGridCpp:
        game_map = self._map_builder.build()

        # Validate number of agents
        level_agents = np.count_nonzero(np.char.startswith(game_map.grid, "agent"))
        assert self.__mg_config.game.num_agents == level_agents, (
            f"Number of agents {self.__mg_config.game.num_agents} does not match number of agents in map {level_agents}"
            f". This may be because your map, after removing border width, is too small to fit the number of agents."
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

        # Validate that C++ environment conforms to expected types
        self._validate_c_env_types(c_env)

        self.__c_env_instance = c_env
        return c_env

    def _validate_c_env_types(self, c_env: MettaGridCpp) -> None:
        """Validate that the C++ environment conforms to expected MettaGrid types."""
        from mettagrid.types import validate_action_space, validate_observation_space

        try:
            validate_observation_space(c_env.observation_space)
        except TypeError as e:
            raise TypeError(f"C++ environment observation space does not conform to MettaGrid types: {e}") from e

        try:
            validate_action_space(c_env.action_space)
        except TypeError as e:
            raise TypeError(f"C++ environment action space does not conform to MettaGrid types: {e}") from e

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

    def step(
        self, actions: np.ndarray | int | Sequence[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Execute one timestep of the environment dynamics with the given actions."""
        arr = np.asarray(actions, dtype=dtype_actions)
        if arr.ndim != 1 or arr.shape[0] != self.num_agents:
            raise ValueError(
                f"Expected actions of shape ({self.num_agents},) but received {arr.shape}; "
                "ensure policies emit a scalar action id per agent"
            )
        return self.__c_env_instance.step(arr)

    def render(self) -> None:
        """Render the environment."""
        # Rendering is now handled via the renderer parameter passed to MettaGridEnv
        # This method is kept for backward compatibility but does nothing
        pass

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
    def _action_space(self) -> spaces.Discrete:
        """Internal action space - use single_action_space for PufferEnv compatibility."""
        return self.__c_env_instance.action_space

    @property
    def action_names(self) -> List[str]:
        return self.__c_env_instance.action_names()

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

    @property
    def observation_features(self) -> Dict[str, ObsFeature]:
        """Build the features dictionary for initialize_to_environment."""
        # Get feature spec from C++ environment
        feature_spec = self.__c_env_instance.feature_spec()

        features = {}
        for feature_name, feature_info in feature_spec.items():
            feature = ObsFeature(
                id=int(feature_info["id"]), normalization=feature_info["normalization"], name=feature_name
            )
            features[feature_name] = feature

        return features

    def grid_objects(
        self, bbox: Optional[BoundingBox] = None, ignore_types: Optional[List[str]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """Get grid objects information, optionally filtered by bounding box and type.

        Args:
            bbox: Bounding box, None for no limit
            ignore_types: List of type names to exclude from results (e.g., ["wall"])

        Returns:
            Dictionary mapping object IDs to object dictionaries
        """
        if bbox is None:
            bbox = BoundingBox(min_row=-1, max_row=-1, min_col=-1, max_col=-1)

        ignore_list = ignore_types if ignore_types is not None else []
        return self.__c_env_instance.grid_objects(bbox.min_row, bbox.max_row, bbox.min_col, bbox.max_col, ignore_list)

    def set_inventory(self, agent_id: int, inventory: Dict[str, int]) -> None:
        """Set an agent's inventory by resource name.

        Any resources not mentioned will be cleared in the underlying C++ call.
        """
        if not isinstance(agent_id, int):
            raise TypeError("agent_id must be an int")
        if not isinstance(inventory, dict):
            raise TypeError("inventory must be a dict[str, int]")

        # Build mapping from resource name to id
        name_to_id = {name: idx for idx, name in enumerate(self.resource_names)}

        # Convert names to ids, validating inputs
        inv_by_id: Dict[int, int] = {}
        for name, amount in inventory.items():
            if name not in name_to_id:
                raise KeyError(f"Unknown resource name: {name}")
            if not isinstance(amount, (int, np.integer)):
                raise TypeError(f"Amount for {name} must be int")
            inv_by_id[int(name_to_id[name])] = int(amount)

        # Forward to C++ binding
        self.__c_env_instance.set_inventory(agent_id, inv_by_id)
