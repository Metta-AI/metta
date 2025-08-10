"""Builder classes for easy construction of TaskSet and Curriculum configurations."""

from __future__ import annotations

from typing import Any

from metta.rl.env_config import EnvConfig
from .config import (
    TaskSetConfig,
    WeightedTaskSetConfig,
    BuckettedTaskSetConfig,
    WeightedTaskSetItem,
    BucketValue,
    CurriculumConfig,
    RandomCurriculumConfig,
    LearningProgressCurriculumConfig,
)


class TaskSetBuilder:
    """Builder for TaskSet configurations."""
    
    @staticmethod
    def weighted() -> WeightedTaskSetBuilder:
        """Create a WeightedTaskSet builder."""
        return WeightedTaskSetBuilder()
        
    @staticmethod
    def bucketed(base_config: EnvConfig | None = None) -> BuckettedTaskSetBuilder:
        """Create a BuckettedTaskSet builder."""
        if base_config is None:
            base_config = EnvConfig()
        return BuckettedTaskSetBuilder(base_config=base_config)


class WeightedTaskSetBuilder:
    """Builder for WeightedTaskSetConfig."""
    
    def __init__(self):
        self.items: list[WeightedTaskSetItem] = []
        self.overrides: dict[str, Any] | list[str] | None = None
        
    def add_env_config(self, env_config: EnvConfig, weight: float = 1.0) -> WeightedTaskSetBuilder:
        """Add an environment configuration with weight."""
        item = WeightedTaskSetItem(env_config=env_config, weight=weight)
        self.items.append(item)
        return self
        
    def add_task_set_config(self, task_set_config: TaskSetConfig, weight: float = 1.0) -> WeightedTaskSetBuilder:
        """Add a nested task set configuration with weight."""
        item = WeightedTaskSetItem(task_set_config=task_set_config, weight=weight)
        self.items.append(item)
        return self
        
    def add_task_set(self, task_set_builder: WeightedTaskSetBuilder | BuckettedTaskSetBuilder, weight: float = 1.0) -> WeightedTaskSetBuilder:
        """Add a nested task set from another builder with weight."""
        return self.add_task_set_config(task_set_builder.build(), weight)
        
    def with_overrides(self, overrides: dict[str, Any] | list[str]) -> WeightedTaskSetBuilder:
        """Add overrides to apply to generated configs."""
        self.overrides = overrides
        return self
        
    def with_dict_overrides(self, **kwargs: Any) -> WeightedTaskSetBuilder:
        """Add overrides as keyword arguments."""
        if self.overrides is None:
            self.overrides = {}
        if isinstance(self.overrides, dict):
            self.overrides.update(kwargs)
        else:
            # Convert list to dict first
            self.overrides = self._parse_list_overrides(self.overrides)
            self.overrides.update(kwargs)
        return self
        
    def _parse_list_overrides(self, overrides: list[str]) -> dict[str, Any]:
        """Parse list overrides to dict format."""
        parsed = {}
        for item in overrides:
            if ":" in item:
                key, value = item.split(":", 1)
                key = key.strip()
                value = value.strip()
                # Try to parse value as number or boolean
                try:
                    if value.lower() in ("true", "false"):
                        value = value.lower() == "true"
                    elif "." in value:
                        value = float(value)
                    elif value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
                        value = int(value)
                except ValueError:
                    pass  # Keep as string
                parsed[key] = value
        return parsed
        
    def build(self) -> WeightedTaskSetConfig:
        """Build the WeightedTaskSetConfig."""
        return WeightedTaskSetConfig(
            items=self.items,
            overrides=self.overrides
        )


class BuckettedTaskSetBuilder:
    """Builder for BuckettedTaskSetConfig."""
    
    def __init__(self, base_config: EnvConfig | None = None):
        self.base_config = base_config or EnvConfig()
        self.buckets: dict[str, list[BucketValue]] = {}
        
    def with_base_config(self, base_config: EnvConfig) -> BuckettedTaskSetBuilder:
        """Set the base configuration."""
        self.base_config = base_config
        return self
        
    def add_bucket(self, key: str, values: list[Any]) -> BuckettedTaskSetBuilder:
        """Add a bucket with list of values."""
        bucket_values = []
        for value in values:
            if isinstance(value, (list, tuple)) and len(value) == 2:
                bucket_values.append(BucketValue(range_min=value[0], range_max=value[1]))
            else:
                bucket_values.append(BucketValue(value=value))
        self.buckets[key] = bucket_values
        return self
        
    def add_value_bucket(self, key: str, values: list[Any]) -> BuckettedTaskSetBuilder:
        """Add a bucket with discrete values only."""
        bucket_values = [BucketValue(value=value) for value in values]
        self.buckets[key] = bucket_values
        return self
        
    def add_range_bucket(self, key: str, ranges: list[tuple[Any, Any]]) -> BuckettedTaskSetBuilder:
        """Add a bucket with ranges only."""
        bucket_values = [BucketValue(range_min=r[0], range_max=r[1]) for r in ranges]
        self.buckets[key] = bucket_values
        return self
        
    def add_single_range_bucket(self, key: str, min_val: Any, max_val: Any) -> BuckettedTaskSetBuilder:
        """Add a bucket with a single range."""
        bucket_values = [BucketValue(range_min=min_val, range_max=max_val)]
        self.buckets[key] = bucket_values
        return self
        
    def build(self) -> BuckettedTaskSetConfig:
        """Build the BuckettedTaskSetConfig."""
        return BuckettedTaskSetConfig(
            base_config=self.base_config,
            buckets=self.buckets
        )


class CurriculumBuilder:
    """Builder for Curriculum configurations."""
    
    @staticmethod
    def random(task_set_config: TaskSetConfig | WeightedTaskSetBuilder | BuckettedTaskSetBuilder) -> RandomCurriculumBuilder:
        """Create a RandomCurriculum builder."""
        if isinstance(task_set_config, (WeightedTaskSetBuilder, BuckettedTaskSetBuilder)):
            task_set_config = task_set_config.build()
        return RandomCurriculumBuilder(task_set_config=task_set_config)
        
    @staticmethod
    def learning_progress(task_set_config: TaskSetConfig | WeightedTaskSetBuilder | BuckettedTaskSetBuilder) -> LearningProgressCurriculumBuilder:
        """Create a LearningProgressCurriculum builder."""
        if isinstance(task_set_config, (WeightedTaskSetBuilder, BuckettedTaskSetBuilder)):
            task_set_config = task_set_config.build()
        return LearningProgressCurriculumBuilder(task_set_config=task_set_config)


class RandomCurriculumBuilder:
    """Builder for RandomCurriculumConfig."""
    
    def __init__(self, task_set_config: TaskSetConfig):
        self.task_set_config = task_set_config
        
    def build(self) -> RandomCurriculumConfig:
        """Build the RandomCurriculumConfig."""
        return RandomCurriculumConfig(
            task_set_config=self.task_set_config
        )


class LearningProgressCurriculumBuilder:
    """Builder for LearningProgressCurriculumConfig."""
    
    def __init__(self, task_set_config: TaskSetConfig):
        self.task_set_config = task_set_config
        self.n_tasks: int = 100
        self.ema_timescale: float = 0.001
        self.progress_smoothing: float = 0.05
        self.num_active_tasks: int = 16
        self.rand_task_rate: float = 0.25
        self.sample_threshold: int = 10
        self.memory: int = 25
        
    def with_n_tasks(self, n_tasks: int) -> LearningProgressCurriculumBuilder:
        """Set the number of tasks to generate."""
        self.n_tasks = n_tasks
        return self
        
        
    def with_ema_timescale(self, ema_timescale: float) -> LearningProgressCurriculumBuilder:
        """Set the EMA timescale for learning progress."""
        self.ema_timescale = ema_timescale
        return self
        
    def with_progress_smoothing(self, progress_smoothing: float) -> LearningProgressCurriculumBuilder:
        """Set the progress smoothing factor."""
        self.progress_smoothing = progress_smoothing
        return self
        
    def with_num_active_tasks(self, num_active_tasks: int) -> LearningProgressCurriculumBuilder:
        """Set the number of active tasks to maintain."""
        self.num_active_tasks = num_active_tasks
        return self
        
    def with_rand_task_rate(self, rand_task_rate: float) -> LearningProgressCurriculumBuilder:
        """Set the rate of random task selection."""
        self.rand_task_rate = rand_task_rate
        return self
        
    def with_sample_threshold(self, sample_threshold: int) -> LearningProgressCurriculumBuilder:
        """Set the minimum samples before task becomes active."""
        self.sample_threshold = sample_threshold
        return self
        
    def with_memory(self, memory: int) -> LearningProgressCurriculumBuilder:
        """Set the number of recent outcomes to remember per task."""
        self.memory = memory
        return self
        
    def build(self) -> LearningProgressCurriculumConfig:
        """Build the LearningProgressCurriculumConfig."""
        return LearningProgressCurriculumConfig(
            task_set_config=self.task_set_config,
            n_tasks=self.n_tasks,
            ema_timescale=self.ema_timescale,
            progress_smoothing=self.progress_smoothing,
            num_active_tasks=self.num_active_tasks,
            rand_task_rate=self.rand_task_rate,
            sample_threshold=self.sample_threshold,
            memory=self.memory
        )