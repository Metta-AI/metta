import logging
from typing import List, Tuple

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


class Task:
    _name: str
    _id: str
    _curricula: List[Tuple[Curriculum, str]]

    def __init__(self, id: str, curriculum: Curriculum):
        self._id = id
        self._name = id
        self._curricula = [(curriculum, id)]

    def complete_trial(self, score: float) -> bool:
        """Lets the task know that a trial has been completed.

        Based on this, the task should expose a new trial, or become complete.
        """
        pass

    def is_complete(self):
        """True if the task is complete, false otherwise."""
        pass

    def env_cfg(self) -> DictConfig:
        """Returns the environment configuration for the current trial."""
        # TODO: ideally we'd have a separate config for the task itself (same for a separate config for the curriculum)
        pass

    def id(self) -> str:
        """Returns the id of the task."""
        return self._id

    def name(self) -> str:
        """Returns the name of the task."""
        return self._name

    def short_name(self) -> str:
        """Returns the short name of the task."""
        return self.name().split("/")[-1]

    def add_parent(self, parent_curriculum: Curriculum, parent_id: str):
        """Adds a parent to the task. Parents are notified when the task is completed."""
        self._curricula.append((parent_curriculum, parent_id))
        self._name = f"{parent_id}:{self._name}"


class SingleTrialTask(Task):
    """A task that only has a single trial. This task may be repeated multiple times."""

    def __init__(self, id: str, curriculum: Curriculum, env_cfg: DictConfig):
        super().__init__(id, curriculum)
        self._total_score = 0.0
        self._num_trials = env_cfg.get("num_trials", 1)
        self._current_trial = 0
        self._is_complete = False
        # We may have been lazy about instantiation up to this point, since that allows us to
        # override the config. Now we complete the instantiation.
        self._env_cfg = hydra.utils.instantiate(env_cfg)

    def complete_trial(self, score: float):
        assert not self._is_complete, "Task is already complete"
        self._current_trial += 1
        self._total_score += score
        if self._current_trial >= self._num_trials:
            self._is_complete = True
            for curriculum, id in self._curricula:
                curriculum.complete_task(id, self._total_score)

    def is_complete(self):
        return self._is_complete

    def env_cfg(self) -> DictConfig:
        assert self._env_cfg is not None, "Task has no environment configuration"
        return self._env_cfg


class SingleTaskCurriculum(Curriculum):
    """Curriculum that only contains a single task."""

    def __init__(self, task_id: str, task_cfg: DictConfig):
        self._task_id = task_id
        self._task_cfg = task_cfg

    def get_task(self) -> Task:
        return SingleTrialTask(self._task_id, self, self._task_cfg)

    def get_task_probs(self) -> dict[str, float]:
        return {self._task_id: 1.0}
