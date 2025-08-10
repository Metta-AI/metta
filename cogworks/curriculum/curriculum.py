from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from cogworks.curriculum.task_set import TaskSet, create_task_set_from_config
from metta.rl.env_config import EnvConfig
from .config import (
    CurriculumConfig,
    RandomCurriculumConfig,
    LearningProgressCurriculumConfig,
    TaskSetConfig
)

logger = logging.getLogger(__name__)


class Task:
    """Task now always wraps a deterministic EnvConfig."""
    
    def __init__(self, env_cfg: EnvConfig, task_id: str | None = None):
        self.env_cfg = env_cfg
        self.task_id = task_id or str(hash(env_cfg.model_dump_json()))
        
    def get_env_config(self) -> EnvConfig:
        """Get the deterministic environment configuration."""
        return self.env_cfg
        
    def get_id(self) -> str:
        """Get task identifier."""
        return self.task_id
        
    def __eq__(self, other) -> bool:
        if not isinstance(other, Task):
            return False
        return self.env_cfg == other.env_cfg
        
    def __hash__(self) -> int:
        return hash(self.env_cfg.model_dump_json())


class Curriculum(ABC):
    """Base curriculum class that uses TaskSet to generate EnvConfigs and returns Tasks.
    
    Curriculum takes a CurriculumConfig, and supports get_task(). It uses the taskset 
    to generate the EnvConfig and then returns a Task(env_cfg).
    """
    
    def __init__(self, config: CurriculumConfig, seed: int = 0):
        self.config = config
        self.task_set = create_task_set_from_config(config.task_set_config)
        self.rng = random.Random(seed)
        
    @abstractmethod
    def get_task(self, seed: int) -> Task:
        """Generate a task using the task set with the provided seed."""
        pass
        
    def complete_task(self, task: Task, score: float):
        """Notify curriculum that a task has been completed with given score."""
        pass
        
    def get_task_probs(self) -> dict[str, float]:
        """Return the current task probabilities for logging purposes."""
        return {}
        
    def get_curriculum_stats(self) -> dict:
        """Return curriculum statistics for logging purposes."""
        return {}


class RandomCurriculum(Curriculum):
    """RandomCurriculum generates tasks using the provided seed."""
    
    def __init__(self, config: RandomCurriculumConfig, seed: int = 0):
        super().__init__(config, seed)
        self.config: RandomCurriculumConfig = config
        
    def get_task(self, seed: int) -> Task:
        """Generate task with the provided seed."""
        # Generate the EnvConfig using the task_set with seed
        env_cfg = self.task_set.get_task(seed)
        
        # Return wrapped Task
        return Task(env_cfg, task_id=f"random_{seed}")


class LearningProgressCurriculum(Curriculum):
    """LearningProgressCurriculum generates N tasks upfront and then samples based on learning progress."""
    
    def __init__(self, config: LearningProgressCurriculumConfig, seed: int = 0):
        super().__init__(config, seed)
        self.config: LearningProgressCurriculumConfig = config
        
        # Initialize empty task list - will be populated on first get_task call
        self.tasks = []
        self._tasks_initialized = False
        self.task_outcomes = {task.get_id(): [] for task in self.tasks}
        
        # Learning progress tracking state
        self.p_fast: np.ndarray | None = None
        self.p_slow: np.ndarray | None = None
        self.p_true: np.ndarray | None = None
        self.random_baseline: np.ndarray | None = None
        self.task_weights = np.ones(len(self.tasks)) / len(self.tasks)
        self.active_task_indices = list(range(min(self.config.num_active_tasks, len(self.tasks))))
        
    def _generate_n_tasks(self, base_seed: int) -> List[Task]:
        """Generate N tasks using different seeds."""
        tasks = []
        for i in range(self.config.n_tasks):
            seed = base_seed + i
            env_cfg = self.task_set.get_task(seed)
            task = Task(env_cfg, task_id=f"lp_{seed}")
            tasks.append(task)
        return tasks
            
    def get_task(self, seed: int) -> Task:
        """Sample task based on learning progress using the provided seed."""
        # Initialize tasks on first call
        if not self._tasks_initialized:
            self.tasks = self._generate_n_tasks(seed)
            self.task_outcomes = {task.get_id(): [] for task in self.tasks}
            self.task_weights = np.ones(len(self.tasks)) / len(self.tasks)
            self.active_task_indices = list(range(min(self.config.num_active_tasks, len(self.tasks))))
            self._tasks_initialized = True
        
        # Use internal RNG for consistent sampling behavior
        if not self.active_task_indices:
            # If no active tasks, select randomly
            task_idx = self.rng.randint(0, len(self.tasks) - 1)
        else:
            # Sample from active tasks based on weights
            active_weights = [self.task_weights[i] for i in self.active_task_indices]
            total_weight = sum(active_weights)
            
            if total_weight <= 0:
                # Uniform sampling if weights are invalid
                task_idx = self.rng.choice(self.active_task_indices)
            else:
                # Weighted sampling
                normalized_weights = [w / total_weight for w in active_weights]
                selected_idx = self.rng.choices(self.active_task_indices, weights=normalized_weights)[0]
                task_idx = selected_idx
                
        return self.tasks[task_idx]
        
    def complete_task(self, task: Task, score: float):
        """Update learning progress based on completed task."""
        task_id = task.get_id()
        
        if task_id not in self.task_outcomes:
            logger.warning(f"Unknown task completed: {task_id}")
            return
            
        # Store outcome
        self.task_outcomes[task_id].append(max(0.0, min(1.0, score)))
        
        # Keep only recent outcomes
        self.task_outcomes[task_id] = self.task_outcomes[task_id][-self.config.memory:]
        
        # Update learning progress
        self._update_learning_progress()
        
    def _update_learning_progress(self):
        """Update learning progress tracking."""
        # Calculate success rates
        success_rates = []
        for task in self.tasks:
            outcomes = self.task_outcomes[task.get_id()]
            if outcomes:
                success_rates.append(np.mean(outcomes))
            else:
                success_rates.append(0.0)
                
        success_rates = np.array(success_rates)
        
        # Initialize baseline if needed
        if self.random_baseline is None:
            self.random_baseline = np.minimum(success_rates, 0.75)  # Cap at 75%
            
        # Normalize success rates
        denominator = 1.0 - self.random_baseline
        denominator = np.where(denominator <= 0, 1.0, denominator)
        normalized_rates = np.maximum(success_rates - self.random_baseline, 0.0) / denominator
        
        # Update EMA trackers
        if self.p_fast is None:
            self.p_fast = normalized_rates.copy()
            self.p_slow = normalized_rates.copy()
            self.p_true = success_rates.copy()
        else:
            self.p_fast = normalized_rates * self.config.ema_timescale + self.p_fast * (1.0 - self.config.ema_timescale)
            self.p_slow = self.p_fast * self.config.ema_timescale + self.p_slow * (1.0 - self.config.ema_timescale)
            self.p_true = success_rates * self.config.ema_timescale + self.p_true * (1.0 - self.config.ema_timescale)
            
        # Calculate learning progress
        learning_progress = np.abs(self._reweight(self.p_fast) - self._reweight(self.p_slow))
        
        # Update task weights based on learning progress
        if np.std(learning_progress) > 0:
            normalized_lp = (learning_progress - np.mean(learning_progress)) / np.std(learning_progress)
            weights = self._sigmoid(normalized_lp)
            weights = weights / np.sum(weights)
            self.task_weights = weights
        else:
            self.task_weights = np.ones(len(self.tasks)) / len(self.tasks)
            
        # Update active tasks
        self._update_active_tasks(learning_progress)
        
    def _reweight(self, probs: np.ndarray) -> np.ndarray:
        """Apply progress smoothing reweighting."""
        numerator = probs * (1.0 - self.config.progress_smoothing)
        denominator = probs + self.config.progress_smoothing * (1.0 - 2.0 * probs)
        denominator = np.where(denominator <= 0, 1.0, denominator)
        return numerator / denominator
        
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Apply sigmoid function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
        
    def _update_active_tasks(self, learning_progress: np.ndarray):
        """Update the set of active tasks based on learning progress."""
        # Select top tasks by learning progress, plus some random ones
        n_progress_tasks = int(self.config.num_active_tasks * (1.0 - self.config.rand_task_rate))
        n_random_tasks = self.config.num_active_tasks - n_progress_tasks
        
        # Top tasks by learning progress
        top_indices = np.argsort(learning_progress)[-n_progress_tasks:].tolist()
        
        # Random tasks
        all_indices = set(range(len(self.tasks)))
        remaining_indices = all_indices - set(top_indices)
        if remaining_indices and n_random_tasks > 0:
            random_indices = self.rng.sample(list(remaining_indices), min(n_random_tasks, len(remaining_indices)))
        else:
            random_indices = []
            
        self.active_task_indices = top_indices + random_indices
        
    def get_task_probs(self) -> dict[str, float]:
        """Return current task probabilities for logging."""
        probs = {}
        for i, task in enumerate(self.tasks):
            if i in self.active_task_indices:
                probs[task.get_id()] = float(self.task_weights[i])
            else:
                probs[task.get_id()] = 0.0
        return probs
        
    def get_curriculum_stats(self) -> dict:
        """Return learning progress statistics."""
        if self.p_fast is None:
            return {}
            
        stats = {
            "lp/num_active_tasks": len(self.active_task_indices),
            "lp/mean_task_weight": float(np.mean(self.task_weights)),
            "lp/num_zero_weight_tasks": int(np.sum(self.task_weights == 0)),
        }
        
        # Success rate statistics
        success_rates = []
        for task in self.tasks:
            outcomes = self.task_outcomes[task.get_id()]
            if outcomes:
                success_rates.append(np.mean(outcomes))
            else:
                success_rates.append(0.0)
                
        if success_rates:
            stats.update({
                "lp/mean_success_rate": float(np.mean(success_rates)),
                "lp/min_success_rate": float(np.min(success_rates)),
                "lp/max_success_rate": float(np.max(success_rates)),
            })
            
        return stats