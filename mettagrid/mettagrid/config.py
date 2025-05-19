import copy
from typing import Any, Dict, Optional, Tuple, cast

import numpy as np
from omegaconf import DictConfig, OmegaConf

from mettagrid.util.hydra import simple_instantiate


class MettaGridConfig:
    """
    A wrapper class for MettaGrid configuration data that's used by both Task and MettaGridEnv.
    This class encapsulates the env_cfg (OmegaConf config) and map (numpy array) needed to initialize MettaGrid.
    """

    def __init__(
        self,
        env_cfg: DictConfig,
        env_map: Optional[np.ndarray] = None,
    ):
        """
        Initialize a MettaGridConfig.

        Args:
            env_cfg: OmegaConf configuration for the environment
            env_map: Optional pre-generated map. If None, a map will be generated using the map_builder from env_cfg
        """
        self.env_cfg = OmegaConf.create(copy.deepcopy(env_cfg))
        self.env_map = env_map
        self._map_builder = None

        # Resolve the configuration
        OmegaConf.resolve(self.env_cfg)

    def generate_map(self) -> np.ndarray:
        """
        Generate a map using the map_builder from env_cfg.
        This is useful when env_map was not provided at initialization.
        """
        if self.env_map is not None:
            return self.env_map

        # Instantiate the map builder and build the map
        self._map_builder = simple_instantiate(
            self.env_cfg.game.map_builder,
            recursive=self.env_cfg.game.get("recursive_map_builder", True),
        )
        self.env_map = self._map_builder.build()

        # Validate the map
        map_agents = np.count_nonzero(np.char.startswith(self.env_map, "agent"))
        assert self.env_cfg.game.num_agents == map_agents, (
            f"Number of agents {self.env_cfg.game.num_agents} does not match number of agents in map {map_agents}"
        )

        return self.env_map

    def to_c_args(self) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        returns the inputs needed to initialize MettaGrid:
        dict-form config and map. If env_map wasn't provided at initialization, it will be generated.

        This allows using the config directly as: config_dict, map_array = metta_grid_config.to_c_args()
        """
        if self._map_builder is None:
            self.generate_map()

        if self.env_map is None:
            raise ValueError("generate_map failed to create a valid env_map")

        # Convert string array to list of strings for C++ compatibility
        env_map_list = self.env_map.tolist()
        env_map = np.array(env_map_list)

        # Convert to container for C++ code with explicit casting to Dict[str, Any]
        config_dict = cast(Dict[str, Any], OmegaConf.to_container(self.env_cfg))

        return config_dict, env_map

    def map_labels(self) -> list[str]:
        if self._map_builder is None:
            self.generate_map()
        if self._map_builder is None:
            raise ValueError("generate_map failed to create a valid _map_builder")
        return self._map_builder.labels or []
