import logging
from typing import List

import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class Curriculum:
    def get_task(self) -> "Task":
        raise NotImplementedError("Subclasses must implement this method")

    def complete_task(self, id: str, score: float):
        # logger.info(f"Task completed: {id} -> {score:.5f}")
        pass

    def completed_tasks(self) -> List[str]:
        """Return a list of completed task identifiers."""
        return []

    def get_completion_rates(self):
        """Return a dictionary of completion rates for each task."""
        return {}

    def get_task_probs(self) -> dict[str, float]:
        """Return the current task probabilities for logging purposes."""
        return {}

    def get_curriculum_stats(self) -> dict:
        """Return curriculum statistics for logging purposes (default: empty)."""
        return {}

    def stats(self) -> dict:
        """Return curriculum statistics for logging purposes."""
        return {
            "completed_tasks": self.completed_tasks(),
            "completion_rates": self.get_completion_rates(),
            "task_probs": self.get_task_probs(),
        }


class Task:
    def __init__(self, id: str, curriculum: "Curriculum", env_cfg: DictConfig):
        self._id = id
        self._is_complete = False
        self._curricula = [(curriculum, id)]
        # We may have been lazy about instantiation up to this point, since that allows us to
        # override the config. Now we complete the instantiation.
        self._env_cfg = hydra.utils.instantiate(env_cfg)
        self._name = self._id

    def complete(self, score: float):
        assert not self._is_complete, "Task is already complete"
        for curriculum, id in self._curricula:
            curriculum.complete_task(id, score)
        self._is_complete = True
        # logger.info(f"Task completed: {self.name()} -> {score:.5f}")

    def is_complete(self):
        return self._is_complete

    def env_cfg(self) -> DictConfig:
        assert self._env_cfg is not None, "Task has no environment configuration"
        return self._env_cfg

    def id(self) -> str:
        return self._id

    def name(self) -> str:
        return self._name

    def short_name(self) -> str:
        return self._name.split("/")[-1]

    def add_parent(self, parent_curriculum: "Curriculum", parent_id: str):
        self._curricula.append((parent_curriculum, parent_id))
        self._name = f"{parent_id}:{self._name}"


class SingleTaskCurriculum(Curriculum):
    """Curriculum that only contains a single task."""

    def __init__(self, task_id: str, task_cfg: DictConfig):
        self._task_id = task_id
        self._task_cfg = task_cfg

    def get_task(self) -> Task:
        return Task(self._task_id, self, self._task_cfg)
