import copy
import logging
import random
from typing import Dict, Optional, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from mettagrid.util.hydra import config_from_path

logger = logging.getLogger(__name__)


class Curriculum:
    def get_task(self) -> "Task":
        raise NotImplementedError("Subclasses must implement this method")

    def complete_task(self, id: str, score: float):
        logger.info(f"Task completed: {id} -> {score:.5f}")

    @staticmethod
    def from_config_path(config_path: str, env_overrides: Optional[DictConfig] = None) -> "Curriculum":
        cfg = config_from_path(config_path, OmegaConf.create({"env_overrides": env_overrides}))
        if "_target_" in cfg:
            return hydra.utils.instantiate(cfg)
        else:
            # If this is an environment rather than a curriculum, we need to wrap it in a curriculum
            # but we have to sample it first.
            task = SamplingCurriculum(config_path, env_overrides).get_task()
            return SingleTaskCurriculum(task.id(), task.env_cfg())


class Task:
    def __init__(self, id: str, curriculum: "Curriculum", env_cfg: Optional[DictConfig] = None):
        self._id = id
        self._is_complete = False
        self._curriculums = [(curriculum, id)]
        self._env_cfg = env_cfg
        self._name = self._id

    def complete(self, score: float):
        assert not self._is_complete, "Task is already complete"
        for curriculum, id in self._curriculums:
            curriculum.complete_task(id, score)
        self._is_complete = True
        logger.info(f"Task completed: {self.name()} -> {score:.5f}")

    def is_complete(self):
        return self._is_complete

    def env_cfg(self) -> DictConfig:
        assert self._env_cfg is not None, "Task has no environment configuration"
        return self._env_cfg

    def id(self) -> str:
        return self._id

    def name(self) -> str:
        return self._name

    def add_parent(self, parent_curriculum: "Curriculum", parent_id: str):
        self._curriculums.append((parent_curriculum, parent_id))
        self._name = f"{parent_id}:{self._name}"


class SingleTaskCurriculum(Curriculum):
    """Curriculum that only contains a single task."""

    def __init__(self, task_id: str, task_cfg: DictConfig):
        self._task_id = task_id
        self._task_cfg = task_cfg

    def get_task(self) -> Task:
        return Task(self._task_id, self, self._task_cfg)


class MultiTaskCurriculum(Curriculum):
    """Base class for curricula with multiple tasks."""

    def __init__(self, tasks: Dict[str, float], env_overrides: DictConfig):
        self._curriculums = {t: Curriculum.from_config_path(t, env_overrides) for t in tasks.keys()}
        self._task_weights = tasks

        num_agents = None
        for task_id, curriculum in self._curriculums.items():
            cfg_num_agents = curriculum.get_task().env_cfg().game.num_agents
            if num_agents is None:
                num_agents = cfg_num_agents
            else:
                assert cfg_num_agents == num_agents, (
                    f"Task {task_id} has num_agents {cfg_num_agents}, expected {num_agents}"
                )


class RandomCurriculum(MultiTaskCurriculum):
    """Curriculum that samples from multiple environment types with fixed weights."""

    def get_task(self) -> Task:
        task_id = random.choices(list(self._curriculums.keys()), weights=list(self._task_weights.values()))[0]
        task = self._curriculums[task_id].get_task()
        task.add_parent(self, task_id)
        logger.debug(f"Task selected: {task.name()}")
        return task


class LowRewardCurriculum(RandomCurriculum):
    """Curriculum that adaptively samples tasks to focus on low-reward scenarios."""

    def __init__(self, tasks: Dict[str, float], env_overrides: DictConfig):
        super().__init__(tasks, env_overrides)
        self._reward_averages = {task_id: 0.0 for task_id in tasks.keys()}
        self._reward_maxes = {task_id: 0.0 for task_id in tasks.keys()}
        self._alpha = 0.01  # Smoothing factor for moving average

    def complete_task(self, id: str, score: float):
        # Update moving average for the completed task
        old_average = self._reward_averages[id]
        self._reward_averages[id] = (1 - self._alpha) * self._reward_averages[id] + self._alpha * score
        logger.debug(
            f"Updated task {id} "
            + f"reward mean({old_average:.3f} -> {self._reward_averages[id]:.3f}), "
            + f"max({self._reward_maxes[id]:.3f})"
        )
        self._reward_maxes[id] = max(self._reward_maxes[id], score)
        self._task_weights = {
            t: 1e-6 + self._reward_maxes[t] / (self._reward_averages[t] + 1e-6) for t in self._curriculums.keys()
        }
        super().complete_task(id, score)


class SamplingCurriculum(Curriculum):
    def __init__(self, env_cfg_template: str, env_overrides: Optional[DictConfig] = None):
        self._cfg_template = cast(DictConfig, config_from_path(env_cfg_template, env_overrides))

    def get_task(self) -> Task:
        cfg = OmegaConf.create(copy.deepcopy(self._cfg_template))
        OmegaConf.resolve(cfg)
        return Task(f"sample({self._cfg_template.sampling})", self, cfg)


class ProgressiveCurriculum(SamplingCurriculum):
    def __init__(self, env_cfg_template: str, env_overrides: Optional[DictConfig] = None):
        super().__init__(env_cfg_template, env_overrides)
        self._width = 10
        self._height = 10

    def get_task(self) -> Task:
        cfg = OmegaConf.create(copy.deepcopy(self._cfg_template))
        cfg.game.map.width = self._width
        cfg.game.map.height = self._height
        OmegaConf.resolve(cfg)
        return Task(f"sample({self._cfg_template.sampling})", self, cfg)

    def complete_task(self, id: str, score: float):
        if score > 0.5:
            self._width = min(self._width * 2, 100)
            self._height = min(self._height * 2, 100)
        super().complete_task(id, score)
