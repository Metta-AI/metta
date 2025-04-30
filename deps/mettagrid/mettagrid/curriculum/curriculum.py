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
    """Base class representing a task in a curriculum.

    A task is a unit of work that can be completed and tracked by a curriculum.
    """

    def __init__(self, id: str, curriculum: "Curriculum"):
        """Initialize a task with an ID and parent curriculum.

        Args:
            id: Unique identifier for this task
            curriculum: Parent curriculum that manages this task
        """
        self._id = id
        self._is_complete = False
        self._curriculum = curriculum

    def complete(self, score: float):
        """Mark task as complete with a score.

        Args:
            score: Performance score achieved on this task
        """
        assert not self._is_complete, "Task is already complete"
        self._is_complete = True
        self._curriculum.complete_task(self._id, score)

    def is_complete(self):
        """Check if task has been completed.

        Returns:
            bool: True if task is complete, False otherwise
        """
        return self._is_complete


class Curriculum:
    """Base class for curriculum learning implementations."""

    def get_task(self) -> Task:
        """Get the next task from the curriculum.

        Returns:
            Task: The next task to be completed
        """
        raise NotImplementedError("Subclasses must implement this method")

    def complete_task(self, id: str, score: float):
        """Mark a task as complete with its score.

        Args:
            id: ID of completed task
            score: Score achieved on the task
        """
        raise NotImplementedError("Subclasses must implement this method")


class MettaGridTask(Task):
    """Task implementation specific to MettaGrid environments."""

    def __init__(self, id: str, curriculum: "Curriculum", game_cfg: DictConfig, game_map: Optional[np.ndarray] = None):
        """Initialize a MettaGrid task.

        Args:
            id: Unique task identifier
            curriculum: Parent curriculum
            game_cfg: Game configuration
            game_map: Optional pre-built game map
        """
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
        """Get the game configuration.

        Returns:
            DictConfig: Game configuration for this task
        """
        return self._game_cfg

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


class MettaGridCurriculum(Curriculum):
    """Basic curriculum implementation for MettaGrid that serves a single task type."""

    def __init__(self, game: DictConfig, sampling: float = 0):
        """Initialize curriculum with game config.

        Args:
            game: Game configuration template
            sampling: Sampling parameter (unused)
        """
        self._cfg_template = game
        self._sampling = sampling

    def get_task(self) -> MettaGridTask:
        """Create a new task from the template.

        Returns:
            MettaGridTask: New task instance
        """
        cfg = OmegaConf.create(copy.deepcopy(self._cfg_template))
        OmegaConf.resolve(cfg)
        logger.info(f"Creating task with config with frozen_duration {cfg.agent.freeze_duration}")
        return MettaGridTask("default", self, cfg)

    def complete_task(self, id: str, score: float):
        """Log task completion.

        Args:
            id: Task ID
            score: Score achieved
        """
        logger.info(f"Completing task {id} with score {score}")


class MultiEnvCurriculum(Curriculum):
    """Curriculum that samples from multiple environment types with fixed weights."""

    def __init__(self, tasks: Dict[str, float], game: DictConfig, **kwargs):
        """Initialize with task types and their weights.

        Args:
            tasks: Mapping of task IDs to selection weights
            game: Base game configuration
            **kwargs: Additional arguments
        """
        overrides = DictConfig({"game": game})
        self._tasks = {t: MettaGridCurriculum(config_from_path(t, overrides).game) for t in tasks.keys()}
        self._task_weights = tasks

    def get_task(self) -> Task:
        """Sample a task according to the weights.

        Returns:
            Task: Randomly selected task
        """
        task_id = random.choices(list(self._tasks.keys()), weights=list(self._task_weights.values()))[0]
        task_cfg = self._tasks[task_id].get_task().game_cfg()
        logger.info(f"Selected task {task_id} with max_steps {task_cfg.max_steps}")
        return MettaGridTask(task_id, self, task_cfg)

    def complete_task(self, id: str, score: float):
        """Log task completion.

        Args:
            id: Task ID
            score: Score achieved
        """
        logger.info(f"Completing task {id} with score {score}")


class LowRewardCurriculum(MultiEnvCurriculum):
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
        return MettaGridTask(task_id, self, task_cfg)

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
