import random
from typing import Dict, List, Optional

import numpy as np

from .curriculum import (
    CurriculumAlgorithm,
    CurriculumAlgorithmConfig,
    CurriculumTask,
    TaskSample,
)
from .task_tracker import TaskTracker


class LearningProgressConfig(CurriculumAlgorithmConfig):
    """Configuration for the learning progress algorithm."""

    type: str = "learning_progress"

    # Core algorithm parameters
    ema_timescale: float = 0.001
    exploration_bonus: float = 0.1

    # Performance and memory management
    min_presentations_for_eviction: int = 5
    max_memory_tasks: int = 1000
    max_bucket_axes: int = 3
    enable_detailed_bucket_logging: bool = False

    def algorithm_type(self) -> str:
        return "learning_progress"

    def create(self, num_tasks: int) -> "LearningProgressAlgorithm":
        return LearningProgressAlgorithm(num_tasks, self)


class EMAScorer:
    """Learning progress scoring using EMA variance calculation"""

    def __init__(self, timescale: float, exploration_bonus: float):
        self.timescale = timescale
        self.exploration_bonus = exploration_bonus
        self._task_emas: Dict[int, tuple[float, float]] = {}  # (ema_score, ema_squared)

    def update_task_ema(self, task: TaskSample) -> None:
        """Update EMA tracking for a task"""
        if task.num_samples == 0:
            return

        current_score = task.get_mean_score()

        if task.task_id not in self._task_emas:
            self._task_emas[task.task_id] = (current_score, current_score * current_score)
        else:
            ema_score, ema_squared = self._task_emas[task.task_id]
            alpha = min(1.0, self.timescale * task.num_samples)

            new_ema_score = (1 - alpha) * ema_score + alpha * current_score
            new_ema_squared = (1 - alpha) * ema_squared + alpha * (current_score * current_score)

            self._task_emas[task.task_id] = (new_ema_score, new_ema_squared)

    def get_learning_progress_score(self, task_id: int, task_tracker: TaskTracker) -> float:
        """Calculate variance-based learning progress score - test interface"""
        task_stats = task_tracker.get_task_stats(task_id)
        if not task_stats or task_stats["completion_count"] < 2:
            return self.exploration_bonus

        if task_id not in self._task_emas:
            return self.exploration_bonus

        ema_score, ema_squared = self._task_emas[task_id]
        variance = max(0.0, ema_squared - ema_score * ema_score)
        return float(np.sqrt(variance))

    def get_learning_progress_score_from_task(self, task: TaskSample) -> float:
        """Calculate variance-based learning progress score - original interface"""
        if task.num_samples < 2:
            return self.exploration_bonus

        if task.task_id not in self._task_emas:
            return self.exploration_bonus

        ema_score, ema_squared = self._task_emas[task.task_id]
        variance = max(0.0, ema_squared - ema_score * ema_score)
        return float(np.sqrt(variance))

    def remove_task(self, task_id: int) -> None:
        """Clean up EMA data for evicted task"""
        self._task_emas.pop(task_id, None)

    def get_stats(self, prefix: str = "") -> Dict[str, float]:
        """Get EMA scoring statistics"""
        if not self._task_emas:
            return {
                f"{prefix}num_tracked_tasks": 0.0,
                f"{prefix}mean_ema_score": 0.0,
                f"{prefix}mean_ema_variance": 0.0,
            }

        ema_scores = [ema[0] for ema in self._task_emas.values()]
        ema_variances = [max(0.0, ema[1] - ema[0] * ema[0]) for ema in self._task_emas.values()]

        return {
            f"{prefix}num_tracked_tasks": float(len(self._task_emas)),
            f"{prefix}mean_ema_score": float(np.mean(ema_scores)),
            f"{prefix}mean_ema_variance": float(np.mean(ema_variances)),
        }


class LearningProgressEvictionPolicy:
    """Eviction policy based on learning progress (variance) rather than mean score"""

    def __init__(self, min_samples: int = 5, bottom_percentile: float = 0.2, ema_scorer: EMAScorer = None):
        self.min_samples = min_samples
        self.bottom_percentile = bottom_percentile
        self.ema_scorer = ema_scorer

    def should_evict_task(self, task: TaskSample) -> bool:
        """Check if task meets basic eviction criteria"""
        return task.num_samples >= self.min_samples

    def recommend_eviction(self, evictable_tasks: List[TaskSample]) -> Optional[int]:
        """Recommend task with lowest learning progress for eviction"""
        if not evictable_tasks:
            return None

        # Sort by learning progress score (ascending) - lower scores = lower learning progress
        def get_lp_score(task):
            if self.ema_scorer:
                return self.ema_scorer.get_learning_progress_score_from_task(task)
            return 0.0

        sorted_tasks = sorted(evictable_tasks, key=get_lp_score)

        # Evict from bottom percentile
        threshold_index = max(0, int(len(sorted_tasks) * self.bottom_percentile))
        return sorted_tasks[threshold_index].task_id if sorted_tasks else None


class LearningProgressAlgorithm(CurriculumAlgorithm):
    """Learning progress algorithm using composition"""

    def __init__(self, num_tasks: int, hypers: LearningProgressConfig):
        self.num_tasks = num_tasks
        self.hypers = hypers

        self.ema_scorer = EMAScorer(hypers.ema_timescale, hypers.exploration_bonus)
        # Alias for test compatibility
        self.lp_scorer = self.ema_scorer

        self.eviction_policy = LearningProgressEvictionPolicy(
            min_samples=hypers.min_presentations_for_eviction, bottom_percentile=0.2, ema_scorer=self.ema_scorer
        )

        # Initialize task tracker for test compatibility
        self.task_tracker = TaskTracker(
            max_memory_tasks=hypers.max_memory_tasks,
            max_bucket_axes=hypers.max_bucket_axes,
            enable_detailed_bucket_logging=hypers.enable_detailed_bucket_logging,
        )

        # Will be set by Curriculum
        self.task_pool = None

        # For task sampling in tests
        self._rng = random.Random()

        # For test compatibility - store tasks created via get_task_from_pool
        self._test_tasks: Dict[int, TaskSample] = {}

    def set_dependencies(self, task_pool, task_tracker) -> None:
        """Dependency injection from Curriculum"""
        self.task_pool = task_pool
        # Only override task_tracker if provided and not already initialized for tests
        if task_tracker is not None:
            self.task_tracker = task_tracker

    def score_tasks(self, task_ids: List[int]) -> Dict[int, float]:
        """Score tasks based on learning progress"""
        scores = {}
        for task_id in task_ids:
            if self.task_pool:
                task = self.task_pool.get_task(task_id)
                if task:
                    score = self.ema_scorer.get_learning_progress_score_from_task(task)
                    # Ensure we never return 0.0 scores as this breaks sampling
                    scores[task_id] = max(score, 0.001)
                else:
                    scores[task_id] = self.hypers.exploration_bonus
            else:
                # Test mode - use task tracker
                score = self.lp_scorer.get_learning_progress_score(task_id, self.task_tracker)
                scores[task_id] = max(score, 0.001)
        return scores

    def _choose_task_from_list(self, task_ids: List[int]) -> int:
        """Choose a task from a list based on learning progress scores (for tests)"""
        if not task_ids:
            raise ValueError("Cannot choose from empty task list")

        scores = {}
        for task_id in task_ids:
            scores[task_id] = self.lp_scorer.get_learning_progress_score(task_id, self.task_tracker)

        # Convert scores to softmax-like weights to prevent extreme bias
        weights = list(scores.values())
        if sum(weights) <= 0:
            # If all weights are zero, use uniform sampling
            return self._rng.choice(task_ids)

        # Apply softmax to reduce extreme differences
        import math

        max_weight = max(weights)
        exp_weights = [math.exp(w - max_weight) for w in weights]  # Subtract max for numerical stability
        sum_exp = sum(exp_weights)

        if sum_exp > 0:
            normalized_weights = [w / sum_exp for w in exp_weights]
        else:
            normalized_weights = [1.0 / len(weights)] * len(weights)

        return self._rng.choices(task_ids, weights=normalized_weights)[0]

    def recommend_eviction(self, task_ids: List[int]) -> Optional[int]:
        """Recommend task for eviction based on learning progress"""
        evictable_tasks = []

        # Check task pool first (production mode)
        if self.task_pool:
            for task_id in task_ids:
                task = self.task_pool.get_task(task_id)
                if task and self.eviction_policy.should_evict_task(task):
                    evictable_tasks.append(task)
        # Fall back to test tasks (test mode)
        else:
            for task_id in task_ids:
                if task_id in self._test_tasks:
                    task = self._test_tasks[task_id]
                    if self.eviction_policy.should_evict_task(task):
                        evictable_tasks.append(task)

        return self.eviction_policy.recommend_eviction(evictable_tasks)

    def should_evict_task(self, task_id: int, min_presentations: int = 5) -> bool:
        """Check if task should be evicted based on learning progress criteria"""
        if not self.task_pool:
            return False

        task = self.task_pool.get_task(task_id)
        if not task:
            return False

        # Check if task has enough samples/presentations to be considered for eviction
        return task.num_samples >= min_presentations

    def update_task_performance(self, task_id: int, score: float) -> None:
        """Update task performance and EMA tracking"""
        # Update task tracker for test compatibility
        if self.task_tracker:
            self.task_tracker.update_task_performance(task_id, score)

        # Update task pool if available (production mode)
        if self.task_pool:
            self.task_pool.update_task_performance(task_id, score)
            task = self.task_pool.get_task(task_id)
            if task:
                self.ema_scorer.update_task_ema(task)
        # Update test tasks if in test mode
        elif task_id in self._test_tasks:
            task = self._test_tasks[task_id]
            task.score += score
            task.num_samples += 1
            self.ema_scorer.update_task_ema(task)

    def on_task_evicted(self, task_id: int) -> None:
        """Clean up when task is evicted"""
        self.ema_scorer.remove_task(task_id)
        if self.task_tracker:
            self.task_tracker.remove_task(task_id)

    def on_task_created(self, task: CurriculumTask) -> None:
        """Handle new task creation"""
        if self.task_tracker:
            self.task_tracker.track_task_creation(task._task_id)

    def stats(self, prefix: str = "") -> Dict[str, float]:
        """Get comprehensive learning progress statistics"""
        stats = {}

        # EMA scorer stats - use "lp/" as key for backward compatibility
        ema_stats = self.ema_scorer.get_stats(f"{prefix}lp/")
        stats.update(ema_stats)

        # Task tracker stats
        if self.task_tracker:
            tracker_stats = self.task_tracker.get_global_stats()
            for key, value in tracker_stats.items():
                stats[f"{prefix}tracker/{key}"] = value

        # Eviction policy stats
        stats[f"{prefix}eviction/min_samples"] = float(self.eviction_policy.min_samples)
        stats[f"{prefix}eviction/bottom_percentile"] = self.eviction_policy.bottom_percentile

        # Algorithm config stats
        stats[f"{prefix}config/ema_timescale"] = self.hypers.ema_timescale
        stats[f"{prefix}config/exploration_bonus"] = self.hypers.exploration_bonus

        return stats

    # Backwards compatibility for tests
    def get_task_from_pool(self, task_generator, rng):
        """Backwards compatibility method for tests"""
        from .curriculum import CurriculumTask, TaskSample

        task_id = rng.randint(0, 1000000)
        env_cfg = task_generator.get_task(task_id)
        task = CurriculumTask(task_id, env_cfg)

        # Create corresponding TaskSample for EMA tracking
        task_sample = TaskSample(
            task_id=task_id,
            score=0.0,
            num_samples=0,
            env_class=type(env_cfg),
            seed=getattr(env_cfg, "seed", None),
            bucket_values=getattr(env_cfg, "bucket_values", {}),
        )
        self._test_tasks[task_id] = task_sample

        # Track the task creation for test compatibility
        self.on_task_created(task)

        return task

    def set_curriculum_reference(self, curriculum) -> None:
        """Backwards compatibility method for tests"""
        # Store reference for backwards compatibility
        self._curriculum = curriculum
