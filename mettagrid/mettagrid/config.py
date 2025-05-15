import copy
import logging
from typing import Any, Dict, Optional, Tuple, cast

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class MettaGridConfig:
    """
    A wrapper class for MettaGrid configuration data that's used by both Task and MettaGridEnv.
    This class encapsulates the env_cfg (OmegaConf config) and map (numpy array) needed to initialize MettaGrid.
    """

    def __init__(
        self,
        env_cfg: DictConfig,
        env_map: Optional[np.ndarray] = None,
        map_builder_override: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a MettaGridConfig.

        Args:
            env_cfg: OmegaConf configuration for the environment
            env_map: Optional pre-generated map. If None, a map will be generated using the map_builder from env_cfg
            map_builder_override: Optional parameters to override the map_builder configuration
        """
        self._env_cfg = OmegaConf.create(copy.deepcopy(env_cfg))
        self._env_map = env_map
        self._map_builder_override = map_builder_override
        self._map_builder = None

        # Resolve the configuration
        OmegaConf.resolve(self._env_cfg)

    @property
    def env_cfg(self) -> DictConfig:
        """Get the environment configuration."""
        return self._env_cfg

    @property
    def env_map(self) -> Optional[np.ndarray]:
        """Get the environment map if it exists."""
        return self._env_map

    def generate_map(self) -> np.ndarray:
        """
        Generate a map using the map_builder from env_cfg.
        This is useful when env_map was not provided at initialization.
        """
        if self._env_map is not None:
            return self._env_map

        # Create the map builder configuration with any overrides
        map_builder_cfg = self._env_cfg.game.map_builder
        if self._map_builder_override:
            for key, value in self._map_builder_override.items():
                OmegaConf.update(map_builder_cfg, key, value)

        # Instantiate the map builder and build the map
        self._map_builder = hydra.utils.instantiate(
            map_builder_cfg, recursive=self._env_cfg.game.get("recursive_map_builder", True)
        )

        env_map = self._map_builder.build()

        # Validate the map
        map_agents = np.count_nonzero(np.char.startswith(env_map, "agent"))
        assert self._env_cfg.game.num_agents == map_agents, (
            f"Number of agents {self._env_cfg.game.num_agents} does not match number of agents in map {map_agents}"
        )

        self._env_map = env_map
        return env_map

    @property
    def map_builder(self):
        """Get the map builder if it was created."""
        return self._map_builder

    def to_c_args(self) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        returns the inputs needed to initialize MettaGrid:
        dict-form config and map. If env_map wasn't provided at initialization, it will be generated.

        This allows using the config directly as: config_dict, map_array = metta_grid_config()
        """
        env_map = self.generate_map() if self._env_map is None else self._env_map

        # Convert to container for C++ code with explicit casting to Dict[str, Any]
        config_dict = cast(Dict[str, Any], OmegaConf.to_container(self.env_cfg))

        # Convert string array to list of strings for C++ compatibility
        env_map_list = env_map.tolist()
        env_map = np.array(env_map_list)

        return config_dict, env_map
