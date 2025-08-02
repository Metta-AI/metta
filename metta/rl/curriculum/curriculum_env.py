"""MettaGridEnv with integrated new curriculum system."""

import logging
from typing import Optional

import numpy as np
from omegaconf import DictConfig

from metta.mettagrid.curriculum.core import Curriculum as OldCurriculum
from metta.mettagrid.curriculum.core import Task as OldTask
from metta.mettagrid.mettagrid_env import MettaGridEnv

from .client import CurriculumClient
from .generator import TaskGenerator
from .task import Task

logger = logging.getLogger(__name__)


class CurriculumAdapter(OldCurriculum):
    """Adapts the new curriculum system to the old Curriculum interface."""

    def __init__(self, curriculum_client: CurriculumClient, task_generator: TaskGenerator):
        """Initialize the adapter."""
        self._client = curriculum_client
        self._generator = task_generator
        self._current_task: Optional[Task] = None
        self._current_env_cfg: Optional[DictConfig] = None

    def get_task(self) -> OldTask:
        """Get a new task from the curriculum."""
        # Get task from new curriculum system
        self._current_task = self._client.get_task()
        self._current_env_cfg = self._generator.generate(self._current_task.task_id)

        # Create adapter task that wraps the new task
        return CurriculumTaskAdapter(self._current_task, self._current_env_cfg)

    def short_name(self) -> str:
        """Return the short name of the curriculum."""
        return "new_curriculum"


class CurriculumTaskAdapter(OldTask):
    """Adapts the new Task to the old Task interface."""

    def __init__(self, task: Task, env_cfg: DictConfig):
        """Initialize the adapter."""
        self._task = task
        self._env_cfg = env_cfg
        self._agent_rewards = []  # List of rewards per agent across all timesteps

    def env_cfg(self) -> DictConfig:
        """Return the environment configuration for this task."""
        return self._env_cfg

    def complete(self, reward: float):
        """Complete the task with the given reward.

        Note: The old system only passes mean reward, but our new system
        needs both mean and variance. We'll calculate variance from
        accumulated rewards across agents.
        """
        # This is called by the old system - we'll ignore it and calculate in finalize_episode
        pass

    def short_name(self) -> str:
        """Return a short name for this task."""
        return f"task_{self._task.task_id}"

    def finalize_episode(self):
        """Called when episode is done to actually complete the task."""
        if self._agent_rewards:
            # Calculate total reward per agent across all timesteps
            agent_total_rewards = np.sum(self._agent_rewards, axis=0)
            # Calculate mean and variance across agents
            reward_mean = np.mean(agent_total_rewards)
            reward_var = np.var(agent_total_rewards) if len(agent_total_rewards) > 1 else 0.0
            self._task.complete(reward_mean, reward_var)


class CurriculumEnv(MettaGridEnv):
    """MettaGridEnv with integrated new curriculum system."""

    def __init__(
        self,
        curriculum_client: CurriculumClient,
        task_generator: TaskGenerator,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the curriculum environment.

        Args:
            curriculum_client: Client for selecting tasks from the curriculum pool
            task_generator: Generator for creating env configs from task IDs
            render_mode: Rendering mode for the environment
            **kwargs: Additional arguments passed to MettaGridEnv
        """
        self._curriculum_client = curriculum_client
        self._task_generator = task_generator

        # Create curriculum adapter
        curriculum_adapter = CurriculumAdapter(curriculum_client, task_generator)

        # Initialize parent with our adapted curriculum
        super().__init__(curriculum=curriculum_adapter, render_mode=render_mode, **kwargs)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        """Reset the environment with a new task from the curriculum."""
        # Complete previous task if we have episode rewards
        if hasattr(self, "_task") and isinstance(self._task, CurriculumTaskAdapter):
            self._task.finalize_episode()

        # Call parent reset which will get a new task
        obs, infos = super().reset(seed)

        # Add task info to infos
        if hasattr(self, "_task") and hasattr(self._task, "_task"):
            infos["task_id"] = self._task._task.task_id
            infos["curriculum/task_id"] = self._task._task.task_id

        return obs, infos

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """Execute one timestep and track rewards for task completion."""
        obs, rewards, terminals, truncations, infos = super().step(actions)

        # Track per-agent rewards across timesteps
        if hasattr(self, "_task") and isinstance(self._task, CurriculumTaskAdapter):
            # Store rewards for all agents at this timestep
            self._task._agent_rewards.append(rewards.copy())

        # Add curriculum info
        if hasattr(self, "_task") and hasattr(self._task, "_task"):
            infos["curriculum/task_id"] = self._task._task.task_id

        return obs, rewards, terminals, truncations, infos

    def close(self):
        """Clean up resources."""
        # Complete final task if needed
        if hasattr(self, "_task") and isinstance(self._task, CurriculumTaskAdapter):
            self._task.finalize_episode()
        super().close()
