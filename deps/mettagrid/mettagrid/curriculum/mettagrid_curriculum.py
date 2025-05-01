import copy
import logging
from typing import Any, Dict, Optional

import numpy as np
from omegaconf import DictConfig, OmegaConf

from mettagrid.config.utils import simple_instantiate
from mettagrid.curriculum.curriculum import Curriculum, Task

logger = logging.getLogger(__name__)


class MettaGridTask(Task):
    """Task implementation specific to MettaGrid environments."""

    def __init__(self, id: str, curriculum: "Curriculum", env_cfg: DictConfig):
        """Initialize a MettaGrid task.

        Args:
            id: Unique task identifier
            curriculum: Parent curriculum
            game_cfg: Game configuration
        """
        self._env_cfg = env_cfg
        self._map_builder = simple_instantiate(
            self._env_cfg.map_builder,
            recursive=self._env_cfg.get("recursive_map_builder", True),
        )
        self._level_map = self._map_builder.build()

        map_agents = np.count_nonzero(np.char.startswith(self._level_map, "agent"))
        assert self._env_cfg.num_agents == map_agents, (
            f"Number of agents {self._env_cfg.num_agents} does not match number of agents in map {map_agents}"
        )
        super().__init__(id, curriculum, env_cfg)

    def env_cfg(self) -> DictConfig:
        """Get the environment configuration.

        Returns:
            DictConfig: Environment configuration for this task
        """
        return self._env_cfg

    def level_map(self) -> Optional[np.ndarray]:
        """Get the level map.

        Returns:
            Optional[np.ndarray]: Level map array if available
        """
        return self._level_map

    def complete(self, infos: Dict[str, Any]):
        """Complete the task with episode information.

        Args:
            infos: Dictionary containing episode metrics
        """
        super().complete(infos["episode_rewards"].sum())
        # self.labels = self._task.env_cfg().get("labels", None)
