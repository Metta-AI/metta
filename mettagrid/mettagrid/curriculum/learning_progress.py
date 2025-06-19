from pdb import set_trace as T

import numpy as np
from collections import defaultdict

import pufferlib
from gymnasium.spaces import Discrete

from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf

from metta.util.config import config_from_path

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import torch

from mettagrid.curriculum.curriculum import Curriculum, Task

if TYPE_CHECKING:
    from mettagrid.mettagrid_env import MettaGridEnv


@dataclass
class BidirectionalLearningProgess:
    """Tracks learning progress for a task using bidirectional learning progress."""

    ema_alpha: float = 0.001
    _ema_reward: Optional[float] = None
    _prev_ema_reward: Optional[float] = None

    def update(self, reward: float) -> float:
        """Update the learning progress estimate with a new reward."""
        if self._ema_reward is None:
            self._ema_reward = reward
            self._prev_ema_reward = reward
            return 0.0

        self._prev_ema_reward = self._ema_reward
        self._ema_reward = (1 - self.ema_alpha) * self._ema_reward + self.ema_alpha * reward
        return self._ema_reward - self._prev_ema_reward


class MettaGridEnvLPSet(Curriculum):
    """A set of MettaGrid environments with learning progress tracking."""

    def __init__(
        self,
        env_cfg: DictConfig,
        ema_alpha: float = 0.001,
        lp_weight: float = 0.0,
        p_theta: float = 0.05,
        num_active_tasks: int = 16,
    ):
        """Initialize the learning progress curriculum.

        Args:
            env_cfg: Environment configuration
            ema_alpha: EMA decay rate for learning progress
            lp_weight: Weight of learning progress in task selection (0 to 1)
            p_theta: Probability threshold for task selection
            num_active_tasks: Number of active tasks to maintain
        """
        super().__init__()
        self.env_cfg = env_cfg
        self.ema_alpha = ema_alpha
        self.lp_weight = lp_weight
        self.p_theta = p_theta
        self.num_active_tasks = num_active_tasks

        self.tasks: List[Task] = []
        self.learning_progress: Dict[int, BidirectionalLearningProgess] = {}
        self._setup_tasks()

    def _setup_tasks(self):
        """Set up the tasks from the environment configuration."""
        # Import here to avoid circular import
        from mettagrid.mettagrid_env import MettaGridEnv

        for env_path in self.env_cfg.envs:
            env = MettaGridEnv.from_config_path(env_path, self.env_cfg)
            task = Task(env=env)
            self.tasks.append(task)
            self.learning_progress[task.id] = BidirectionalLearningProgess(
                ema_alpha=self.ema_alpha
            )

    def update(self, task_id: int, reward: float) -> None:
        """Update the learning progress for a task."""
        if task_id in self.learning_progress:
            self.learning_progress[task_id].update(reward)

    def sample(self) -> Tuple[Task, float]:
        """Sample a task based on learning progress."""
        if not self.tasks:
            raise ValueError("No tasks available")

        # Calculate task probabilities
        lp_values = np.array([
            self.learning_progress[task.id]._ema_reward
            if self.learning_progress[task.id]._ema_reward is not None
            else 0.0
            for task in self.tasks
        ])

        # Normalize learning progress values
        if lp_values.std() > 0:
            lp_values = (lp_values - lp_values.mean()) / lp_values.std()

        # Combine uniform and learning progress probabilities
        uniform_probs = np.ones(len(self.tasks)) / len(self.tasks)
        lp_probs = torch.softmax(torch.tensor(lp_values), dim=0).numpy()

        probs = (1 - self.lp_weight) * uniform_probs + self.lp_weight * lp_probs

        # Sample task
        task_idx = np.random.choice(len(self.tasks), p=probs)
        task = self.tasks[task_idx]

        return task, probs[task_idx]
