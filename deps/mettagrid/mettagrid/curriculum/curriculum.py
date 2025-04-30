import copy
import logging
import random
from typing import Any, Dict, Optional

import numpy as np
from omegaconf import DictConfig, OmegaConf

from metta.util.config import config_from_path
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
    def __init__(self, game: DictConfig, sampling: float = 0):
        self._cfg_template = game
        self._sampling = sampling

    def get_task(self) -> MettaGridTask:
        cfg = OmegaConf.create(copy.deepcopy(self._cfg_template))
        OmegaConf.resolve(cfg)
        return MettaGridTask("default", self, cfg)

    def complete_task(self, id: str, score: float):
        logger.info(f"Completing task {id} with score {score}")


class MultiEnvCurriculum(Curriculum):
    def __init__(self, tasks: Dict[str, float], game: DictConfig, **kwargs):
        overrides = DictConfig({"game": game})
        self._tasks = {t: MettaGridCurriculum(config_from_path(t, overrides).game) for t in tasks.keys()}
        self._task_weights = tasks

    def get_task(self) -> Task:
        task_id = random.choices(list(self._tasks.keys()), weights=list(self._task_weights.values()))[0]
        task_cfg = self._tasks[task_id].get_task().game_cfg()
        logger.info(f"Selected task {task_id} with max_steps {task_cfg.max_steps}")
        return MettaGridTask(task_id, self, task_cfg)

    def complete_task(self, id: str, score: float):
        logger.info(f"Completing task {id} with score {score}")
