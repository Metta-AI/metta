"""Core curriculum implementations and utilities."""

from __future__ import annotations

import random
from typing import Annotated, ClassVar, Union

from pydantic import ConfigDict, Field, field_validator

from metta.common.config import Config
from metta.mettagrid.mettagrid_config import EnvConfig

from .task_generator import (
    BucketedTaskGeneratorConfig,
    SingleTaskGeneratorConfig,
    TaskGeneratorSetConfig,
)


class CurriculumTask:
    """A task instance with a task_id and env_cfg."""

    def __init__(self, task_id: int, env_cfg: EnvConfig):
        self._task_id = task_id
        self._env_cfg = env_cfg
        self._num_completions = 0
        self._total_score = 0.0
        self._mean_score = 0.0
        self._num_scheduled = 0

    def complete(self, score: float):
        """Notify curriculum that a task has been completed with given score."""
        self._num_completions += 1
        self._total_score += score
        self._mean_score = self._total_score / self._num_completions

    def get_env_cfg(self) -> EnvConfig:
        """Get the env_cfg for the task."""
        return self._env_cfg


class CurriculumConfig(Config):
    """Base configuration for Curriculum."""

    task_generator: Annotated[
        Union[SingleTaskGeneratorConfig, TaskGeneratorSetConfig, BucketedTaskGeneratorConfig],
        Field(discriminator="type", description="TaskGenerator configuration"),
    ]
    max_task_id: int = Field(default=1000000, gt=0, description="Maximum task id to generate")

    num_active_tasks: int = Field(default=100, gt=0, description="Number of active tasks to maintain")
    new_task_rate: float = Field(default=0.01, ge=0, le=1.0, description="Rate of new tasks to generate")

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )

    @field_validator("num_active_tasks")
    @classmethod
    def validate_num_active_tasks(cls, v, info):
        max_task_id = info.data.get("max_task_id", 1000000)
        if v > max_task_id:
            raise ValueError("num_active_tasks must be less than max_task_id")
        return v

    def make(self) -> Curriculum:
        """Make a Curriculum from this configuration."""
        return Curriculum(self)


class Curriculum:
    """Base curriculum class that uses TaskGenerator to generate EnvConfigs and returns Tasks.

    Curriculum takes a CurriculumConfig, and supports get_task(). It uses the task generator
    to generate the EnvConfig and then returns a Task(env_cfg).
    """

    def __init__(self, config: CurriculumConfig, seed: int = 0):
        self._config = config
        self._task_generator = config.task_generator.create()
        self._rng = random.Random(seed)
        self._tasks: dict[int, CurriculumTask] = {}
        self._task_ids: set[int] = set()
        self._num_created = 0
        self._num_evicted = 0

    def get_task(self) -> CurriculumTask:
        """Sample a task from the population."""
        if len(self._tasks) < self._config.num_active_tasks:
            task = self._create_task()
        elif self._rng.random() < self._config.new_task_rate:
            self._evict_task()
            task = self._create_task()
        else:
            task = self._choose_task()

        task._num_scheduled += 1
        return task

    def _choose_task(self) -> CurriculumTask:
        """Choose a task from the population."""
        return self._tasks[self._rng.choice(list(self._tasks.keys()))]

    def _create_task(self) -> CurriculumTask:
        """Create a new task."""
        task_id = self._rng.randint(0, self._config.max_task_id)
        while task_id in self._task_ids:
            task_id = self._rng.randint(0, self._config.max_task_id)
        self._task_ids.add(task_id)
        env_cfg = self._task_generator.get_task(task_id)
        task = CurriculumTask(task_id, env_cfg)
        self._tasks[task_id] = task
        self._num_created += 1
        return task

    def _evict_task(self):
        """Evict a task from the population."""
        task_id = self._rng.choice(list(self._task_ids))
        self._task_ids.remove(task_id)
        self._tasks.pop(task_id)
        self._num_evicted += 1

    def stats(self) -> dict:
        """Return curriculum statistics for logging purposes."""
        return {
            "num_created": self._num_created,
            "num_evicted": self._num_evicted,
            "num_completed": sum(task._num_completions for task in self._tasks.values()),
            "num_scheduled": sum(task._num_scheduled for task in self._tasks.values()),
            "num_active_tasks": len(self._tasks),
        }
