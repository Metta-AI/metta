"""Configuration for weighted curriculum algorithm that uses task type sampling."""

from __future__ import annotations
from typing import Dict, TYPE_CHECKING
import random

if TYPE_CHECKING:
    from .curriculum import CurriculumAlgorithm

from .curriculum import CurriculumAlgorithmConfig, Curriculum, CurriculumConfig
from .learning_progress_algorithm import LearningProgressConfig
from .weighted_task_generator import WeightedTaskGenerator
from .curriculum import CurriculumTask


class WeightedCurriculumAlgorithmConfig(CurriculumAlgorithmConfig):
    """Configuration for weighted curriculum algorithm using O(1) task type sampling."""

    type: str = "weighted_curriculum"

    # Learning progress parameters
    ema_timescale: float = 0.001
    exploration_bonus: float = 0.1
    enable_detailed_bucket_logging: bool = False

    def algorithm_type(self) -> str:
        return "weighted_curriculum"

    def create(self, num_tasks: int) -> "WeightedCurriculumAlgorithm":
        # Create learning progress config with our parameters
        lp_config = LearningProgressConfig(
            ema_timescale=self.ema_timescale,
            exploration_bonus=self.exploration_bonus,
            enable_detailed_bucket_logging=self.enable_detailed_bucket_logging,
        )
        return WeightedCurriculumAlgorithm(lp_config)
    
    def make_curriculum(self, config: CurriculumConfig, seed: int = 0) -> "WeightedCurriculum":
        """Create a WeightedCurriculum instead of standard Curriculum."""
        return WeightedCurriculum(config, seed, self)


class WeightedCurriculumAlgorithm:
    """Curriculum algorithm that uses O(1) weighted sampling over task types."""

    def __init__(self, config: LearningProgressConfig):
        self._config = config
        self._task_types = []
        self._task_type_weights = {}
        self._task_type_counts = {}
        self._learning_progress = None
        self._curriculum_ref = None
        self._rng = random.Random(0)
        self._initialized = False

    def set_curriculum_reference(self, curriculum):
        """Set reference to the curriculum for task generation."""
        self._curriculum_ref = curriculum

    def _initialize_task_types(self):
        """Initialize task types from curriculum's task generator."""
        if self._curriculum_ref and not self._initialized:
            task_generator = self._curriculum_ref._task_generator
            self._task_types = task_generator.get_all_task_types()
            self._task_type_weights = {task_type: 1.0 for task_type in self._task_types}
            self._task_type_counts = {task_type: 0 for task_type in self._task_types}
            self._learning_progress = TaskTypeLearningProgress(self._config, self._task_types)
            self._normalize_weights()
            self._initialized = True

    def get_task_from_pool(self, task_generator, rng) -> CurriculumTask:
        """Get a task using weighted sampling over task types."""
        self._initialize_task_types()

        if not self._task_types:
            # Fallback to default behavior if no task types available
            task_id = rng.randint(0, 1000000)
            env_cfg = task_generator.get_task(task_id)
            return CurriculumTask(task_id, env_cfg)

        # O(1) weighted sampling over task types
        task_type = self._rng.choices(
            population=list(self._task_type_weights.keys()), weights=list(self._task_type_weights.values())
        )[0]

        # Generate fresh task instance from selected type
        task_id = rng.randint(0, 1000000)
        env_cfg = task_generator.get_task_by_type(task_type, task_id)
        task = CurriculumTask(task_id, env_cfg)

        # Register task with learning progress
        if self._learning_progress:
            self._learning_progress.register_task(task_id, task_type)

        self._task_type_counts[task_type] += 1
        return task

    def update_task_performance(self, task_id: int, score: float, bucket_values: Dict = None):
        """Update learning progress and task type weights."""
        if self._learning_progress:
            # Pass reference to this algorithm so it can update weights
            self._learning_progress.update_task_performance(task_id, score, bucket_values, self)

    def update_task_type_weights(self, task_type_scores: Dict[str, float]) -> None:
        """Update weights for task types based on learning progress scores."""
        for task_type, score in task_type_scores.items():
            if task_type in self._task_type_weights:
                self._task_type_weights[task_type] = max(0.001, score)
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1."""
        if not self._task_type_weights:
            return

        total_weight = sum(self._task_type_weights.values())
        if total_weight > 0:
            for task_type in self._task_type_weights:
                self._task_type_weights[task_type] /= total_weight
        else:
            uniform_weight = 1.0 / len(self._task_type_weights)
            for task_type in self._task_type_weights:
                self._task_type_weights[task_type] = uniform_weight

    def recommend_eviction(self, evictable_tasks):
        """Recommend tasks for eviction - not needed for weighted approach."""
        return None  # No evictions needed since we don't store tasks

    def score_tasks(self, task_ids):
        """Score tasks for selection - return uniform scores since we don't use stored tasks."""
        # This method is required by CurriculumAlgorithm interface
        # But our weighted approach bypasses stored task selection
        return {task_id: 1.0 for task_id in task_ids}

    def on_task_created(self, task: CurriculumTask) -> None:
        """Notification that a new task has been created."""
        pass  # No tracking needed for weighted approach

    def stats(self) -> Dict[str, float]:
        """Return algorithm statistics."""
        stats = {}
        if self._initialized:
            stats.update(
                {
                    "num_task_types": float(len(self._task_types)),
                }
            )

            for task_type, count in self._task_type_counts.items():
                stats[f"task_type_{task_type}_count"] = float(count)
                stats[f"task_type_{task_type}_weight"] = self._task_type_weights.get(task_type, 0.0)

            if self._learning_progress:
                stats.update(self._learning_progress.get_stats())

        return stats


class WeightedCurriculum(Curriculum):
    """Curriculum that uses weighted sampling over task types for O(1) performance."""
    
    def __init__(self, config: CurriculumConfig, seed: int = 0, weighted_config: WeightedCurriculumAlgorithmConfig = None):
        # Create learning progress config
        lp_config = LearningProgressConfig(
            ema_timescale=weighted_config.ema_timescale if weighted_config else 0.001,
            exploration_bonus=weighted_config.exploration_bonus if weighted_config else 0.1,
            enable_detailed_bucket_logging=weighted_config.enable_detailed_bucket_logging if weighted_config else False,
        )
        
        # Create base task generator
        base_generator = config.task_generator.create()
        
        # Wrap with weighted generator
        self._weighted_generator = WeightedTaskGenerator(base_generator, lp_config, seed)
        
        # Override the task generator in config temporarily
        original_generator = config.task_generator
        
        # Create a dummy config that uses our weighted generator
        # We'll manually set the _task_generator after super().__init__
        super().__init__(config, seed)
        
        # Replace the task generator with our weighted one
        self._task_generator = self._weighted_generator
        
        # Restore original config
        config.task_generator = original_generator

    def update_task_performance(self, task_id: int, score: float):
        """Update task performance through both standard algorithm and weighted generator."""
        # Standard algorithm update
        super().update_task_performance(task_id, score)
        
        # Weighted generator update for task type learning
        bucket_values = {}
        if task_id in self._tasks:
            bucket_values = self._tasks[task_id].get_bucket_values()
        self._weighted_generator.update_task_performance(task_id, score, bucket_values)

    def stats(self) -> Dict[str, float]:
        """Return curriculum statistics including weighted generator stats."""
        stats = super().stats()
        stats.update(self._weighted_generator.get_stats())
        return stats
