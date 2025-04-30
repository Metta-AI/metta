import logging
import random
from typing import Any, Dict, List, Optional

import numpy as np
from omegaconf import DictConfig

from mettagrid.config.utils import simple_instantiate

logger = logging.getLogger(__name__)


class Task:
    def __init__(self, id: str, curriculum: "Curriculum"):
        self._id = id
        self._is_complete = False
        self._curriculum = curriculum

    def complete(self, score: float):
        assert not self._is_complete, "Task is already complete"
        self._is_complete = True
        self._curriculum.complete_task(self._id, score)

    def is_complete(self):
        return self._is_complete


class Curriculum:
    def get_task(self) -> Task:
        raise NotImplementedError("Subclasses must implement this method")

    def complete_task(self, id: str, score: float):
        raise NotImplementedError("Subclasses must implement this method")


class RandomCurriculum(Curriculum):
    def __init__(self, tasks: List[Task]):
        self.tasks = tasks

    def get_task(self) -> Task:
        return random.choice(self.tasks)

    def complete_task(self, id: str, score: float):
        pass


class MettaGridTask(Task):
    def __init__(self, id: str, curriculum: "Curriculum", game_cfg: DictConfig, game_map: Optional[np.ndarray] = None):
        super().__init__(id, curriculum)
        self._game_cfg = game_cfg
        if game_map is None:
            self._map_builder = simple_instantiate(
                self._game_cfg.map_builder,
                recursive=self._game_cfg.get("recursive_map_builder", True),
            )
            game_map = self._map_builder.build()

        map_agents = np.count_nonzero(np.char.startswith(game_map, "agent"))
        assert self._game_cfg.num_agents == map_agents, (
            f"Number of agents {self._game_cfg.num_agents} does not match number of agents in map {map_agents}"
        )
        self._level_map = game_map

    def game_cfg(self) -> DictConfig:
        return self._game_cfg

    def level_map(self) -> Optional[np.ndarray]:
        return self._level_map

    def complete(self, infos: Dict[str, Any]):
        super().complete(infos["episode_rewards"].sum())
        # self.labels = self._task.env_cfg().get("labels", None)


class MettaGridCurriculum(Curriculum):
    def __init__(self, game_cfg_template: DictConfig):
        self.game_cfg_template = game_cfg_template

    def get_task(self) -> MettaGridTask:
        # env_cfg = OmegaConf.create(copy.deepcopy(self._cfg_template))
        # OmegaConf.resolve(env_cfg)
        # return env_cfg
        logger.info(f"Creating MettaGridTask with game_cfg_template: {self.game_cfg_template}")

        return MettaGridTask("default", self, self.game_cfg_template)
