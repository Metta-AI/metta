from typing import Any, Dict, Optional, Tuple, cast

import numpy as np
from omegaconf import DictConfig, OmegaConf

from mettagrid.level_builder import Level
from mettagrid.util.hydra import simple_instantiate


class MettaGridConfig:
    """
    A wrapper class for MettaGrid configuration data that's used by both Task and MettaGridEnv.
    This class encapsulates the env_cfg (OmegaConf config) and level (Level object) needed to initialize MettaGrid.
    """

    def __init__(
        self,
        env_cfg: DictConfig,
        level: Optional[Level] = None,
    ):
        """
        Initialize a MettaGridConfig.

        Args:
            env_cfg: OmegaConf configuration for the environment
            level: Optional pre-generated level. If None, a level will be generated using the map_builder from env_cfg.
        """
        self.env_cfg = env_cfg
        self.level = level
        self._map_builder = None

    def generate_level(self) -> Level:
        """
        Generate a level using the map_builder from env_cfg.
        This is useful when level was not provided at initialization.
        """
        if self.level is not None:
            return self.level

        # Instantiate the map builder and build the level
        self._map_builder = simple_instantiate(
            self.env_cfg.game.map_builder,
            recursive=self.env_cfg.game.get("recursive_map_builder", True),
        )
        self.level = self._map_builder.build()

        # Validate the level
        level_agents = np.count_nonzero(np.char.startswith(self.level.grid, "agent"))
        assert self.env_cfg.game.num_agents == level_agents, (
            f"Number of agents {self.env_cfg.game.num_agents} does not match number of agents in map {level_agents}"
        )

        return self.level

    def to_c_args(self) -> Tuple[Dict[str, Any], Level]:
        """
        Returns the inputs needed to initialize MettaGrid:
        dict-form config and level. If level wasn't provided at initialization, it will be generated.

        This allows using the config directly as: config_dict, level = metta_grid_config.to_c_args()
        """
        if self._map_builder is None:
            self.generate_level()

        if self.level is None:
            raise ValueError("generate_level failed to create a valid level")

        # Convert to container for C++ code with explicit casting to Dict[str, Any]
        config_dict = cast(Dict[str, Any], OmegaConf.to_container(self.env_cfg))

        return config_dict, self.level
