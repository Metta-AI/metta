import copy
import logging
import random
from typing import Dict, Optional

from omegaconf import DictConfig, OmegaConf

from metta.util.config import config_from_path

logger = logging.getLogger(__name__)


class Task:
    """Base class representing a task in a curriculum.

    A task is a unit of work that can be completed and tracked by a curriculum.
    """

    def __init__(self, id: str, curriculum: "Curriculum", env_cfg: DictConfig):
        """Initialize a task with an ID and parent curriculum.

        Args:
            id: Unique identifier for this task
            curriculum: Parent curriculum that manages this task
        """
        self._id = id
        self._is_complete = False
        self._curriculum = curriculum
        self._env_cfg = env_cfg

    def complete(self, score: float):
        """Mark task as complete with a score.

        Args:
            score: Performance score achieved on this task
        """
        assert not self._is_complete, "Task is already complete"
        self._curriculum.complete_task(self._id, score)
        self._is_complete = True

    def is_complete(self):
        """Check if task has been completed.

        Returns:
            bool: True if task is complete, False otherwise
        """
        return self._is_complete

    def env_cfg(self) -> DictConfig:
        """Get the environment configuration for this task.

        Returns:
            DictConfig: Environment configuration for this task
        """
        return self._env_cfg


class Curriculum:
    """Base class for all curriculums."""

    def get_task(self) -> Task:
        """Get a task from the curriculum."""
        raise NotImplementedError("Subclasses must implement this method")

    def complete_task(self, id: str, score: float):
        """Log task completion.

        Args:
            id: Task ID
            score: Score achieved
        """
        logger.info(f"Task completed: {id} -> {score}")


class RandomCurriculum(Curriculum):
    """Curriculum that samples from multiple environment types with fixed weights."""

    def __init__(self, tasks: Dict[str, float], env_overrides: DictConfig):
        """Initialize with task types and their weights.

        Args:
            tasks: Mapping of task IDs to selection weights
            game: Base game configuration
        """
        self._task_configs = {t: config_from_path(t, env_overrides) for t in tasks.keys()}
        self._task_weights = tasks

    def get_task(self) -> Task:
        """Sample a task according to the weights.

        Returns:
            Task: Randomly selected task
        """
        task_id = random.choices(list(self._task_configs.keys()), weights=list(self._task_weights.values()))[0]
        task = Task(task_id, self, self._task_configs[task_id])
        logger.info(f"Task selected: {task_id}")
        return task

    def complete_task(self, id: str, score: float):
        """Log task completion.

        Args:
            id: Task ID
            score: Score achieved
        """
        logger.info(f"Task completed: {id} -> {score:.5f}")


class LowRewardCurriculum(Curriculum):
    """Curriculum that adaptively samples tasks to focus on low-reward scenarios."""

    def __init__(self, tasks: Dict[str, float], game: DictConfig, **kwargs):
        """Initialize with task types and tracking for reward averages.

        Args:
            tasks: Mapping of task IDs to initial weights
            game: Base game configuration
            **kwargs: Additional arguments
        """
        super().__init__(tasks, game, **kwargs)
        self._reward_averages = {task_id: 0.0 for task_id in tasks.keys()}
        self._alpha = 0.01  # Smoothing factor for moving average

    def get_task(self) -> Task:
        """Sample a task with weights inversely proportional to average rewards.

        Returns:
            Task: Selected task biased toward low-reward scenarios
        """
        # Invert rewards so lower rewards get higher weights
        weights = [1.0 / (self._reward_averages[task_id] + 1e-6) for task_id in self._tasks.keys()]
        task_weights = dict(zip(self._tasks.keys(), weights, strict=False))
        logger.info("Task weights before selection:")
        for task_id, weight in task_weights.items():
            logger.info(f"  {task_id:<30} {weight:>10.3f}")

        task_id = random.choices(list(self._tasks.keys()), weights=weights)[0]
        task_cfg = self._tasks[task_id].get_task().game_cfg()
        return Task(task_id, self, task_cfg)

    def complete_task(self, id: str, score: float):
        """Update reward tracking and log completion.

        Args:
            id: Task ID
            score: Score achieved
        """
        # Update moving average for the completed task
        old_average = self._reward_averages[id]
        self._reward_averages[id] = (1 - self._alpha) * self._reward_averages[id] + self._alpha * score
        logger.info(f"Updated reward average for task {id} from {old_average:.3f} to {self._reward_averages[id]:.3f}")
        super().complete_task(id, score)


class SamplingCurriculum(Curriculum):
    def __init__(self, env_cfg_template: str, sampling: float = 0, env_overrides: Optional[DictConfig] = None):
        """Initialize curriculum with environment config.

        Args:
            env_cfg: Environment configuration template
            sampling: Sampling parameter (0-1)
        """
        self._cfg_template = config_from_path(env_cfg_template, env_overrides)
        self._sampling = sampling

    def get_task(self) -> Task:
        """Create a new task from the template.

        Returns:
            Task: New task instance
        """
        cfg = OmegaConf.create(copy.deepcopy(self._cfg_template))
        cfg.sampling = self._sampling
        OmegaConf.resolve(cfg)
        return Task("default", self, cfg)
