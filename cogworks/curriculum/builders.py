"""Builder classes for easy construction of TaskGenerator and Curriculum configurations."""

from __future__ import annotations

from typing import Any

from metta.rl.env_config import EnvConfig

from .learning_progress import LearningProgressCurriculumConfig

# from .random import RandomCurriculumConfig  # Module deleted in branch
from .task.generator import (
    BucketedTaskGeneratorConfig,
    SingleTaskGeneratorConfig,
    TaskGeneratorConfig,
    ValueRange,
    WeightedTaskSetGeneratorConfig,
    WeightedTaskSetGeneratorItem,
)


class TaskGeneratorBuilder:
    """Builder for TaskGenerator configurations."""

    @staticmethod
    def single(env_config: EnvConfig) -> SingleTaskGeneratorConfig:
        """Create a SingleTaskGenerator configuration."""
        return SingleTaskGeneratorConfig(env_config=env_config)

    @staticmethod
    def weighted() -> WeightedTaskSetGeneratorBuilder:
        """Create a WeightedTaskSetGenerator builder."""
        return WeightedTaskSetGeneratorBuilder()

    @staticmethod
    def bucketed(base_generator: TaskGeneratorConfig | None = None) -> BucketedTaskGeneratorBuilder:
        """Create a BucketedTaskGenerator builder."""
        if base_generator is None:
            # Create a default single task generator
            base_generator = SingleTaskGeneratorConfig(env_config=EnvConfig())
        return BucketedTaskGeneratorBuilder(base_generator_config=base_generator)


class WeightedTaskSetGeneratorBuilder:
    """Builder for WeightedTaskSetGeneratorConfig."""

    def __init__(self):
        self._items: list[WeightedTaskSetGeneratorItem] = []
        self._overrides: dict[str, Any] | list[str] | None = None

    def add_env_config(self, env_config: EnvConfig, weight: float = 1.0) -> WeightedTaskSetGeneratorBuilder:
        """Add an environment configuration with weight (creates SingleTaskGenerator)."""
        single_config = SingleTaskGeneratorConfig(env_config=env_config)
        item = WeightedTaskSetGeneratorItem(task_generator_config=single_config, weight=weight)
        self._items.append(item)
        return self

    def add_task_generator_config(
        self, task_gen_config: TaskGeneratorConfig, weight: float = 1.0
    ) -> WeightedTaskSetGeneratorBuilder:
        """Add a nested task generator configuration with weight."""
        item = WeightedTaskSetGeneratorItem(task_generator_config=task_gen_config, weight=weight)
        self._items.append(item)
        return self

    def add_task_generator(
        self, task_gen_builder: WeightedTaskSetGeneratorBuilder | BucketedTaskGeneratorBuilder, weight: float = 1.0
    ) -> WeightedTaskSetGeneratorBuilder:
        """Add a nested task generator from another builder with weight."""
        return self.add_task_generator_config(task_gen_builder.build(), weight)

    def with_overrides(self, overrides: dict[str, Any] | list[str]) -> WeightedTaskSetGeneratorBuilder:
        """Add overrides to apply to generated configs."""
        self._overrides = overrides
        return self

    def with_dict_overrides(self, **kwargs: Any) -> WeightedTaskSetGeneratorBuilder:
        """Add overrides as keyword arguments."""
        if self._overrides is None:
            self._overrides = {}
        if isinstance(self._overrides, dict):
            self._overrides.update(kwargs)
        else:
            # Convert list to dict first
            self._overrides = self._parse_list_overrides(self._overrides)
            self._overrides.update(kwargs)
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

    def build(self) -> WeightedTaskSetGeneratorConfig:
        """Build the WeightedTaskSetGeneratorConfig."""
        # Convert overrides to dict if it's a list
        overrides = self._parse_overrides(self._overrides) if self._overrides else None
        return WeightedTaskSetGeneratorConfig(items=self._items, overrides=overrides)


class BucketedTaskGeneratorBuilder:
    """Builder for BucketedTaskGeneratorConfig."""

    def __init__(self, base_generator_config: TaskGeneratorConfig | None = None):
        self._base_generator_config = base_generator_config or SingleTaskGeneratorConfig(env_config=EnvConfig())
        self._buckets: dict[str, list[int | float | str | ValueRange]] = {}

    def with_base_generator(self, base_generator: TaskGeneratorConfig) -> BucketedTaskGeneratorBuilder:
        """Set the base task generator configuration."""
        self._base_generator_config = base_generator
        return self

    def with_base_env_config(self, base_config: EnvConfig) -> BucketedTaskGeneratorBuilder:
        """Set the base configuration (creates SingleTaskGenerator)."""
        self._base_generator_config = SingleTaskGeneratorConfig(env_config=base_config)
        return self

    def add_bucket(self, key: str, values: list[Any]) -> BucketedTaskGeneratorBuilder:
        """Add a bucket with list of values."""
        bucket_values = []
        for value in values:
            if isinstance(value, (list, tuple)) and len(value) == 2:
                bucket_values.append(ValueRange(range_min=value[0], range_max=value[1]))
            else:
                # Add primitive values directly to the bucket
                bucket_values.append(value)
        self._buckets[key] = bucket_values
        return self

    def add_value_bucket(self, key: str, values: list[Any]) -> BucketedTaskGeneratorBuilder:
        """Add a bucket with discrete values only."""
        # Primitive values can be added directly to the bucket
        self._buckets[key] = values
        return self

    def add_range_bucket(self, key: str, ranges: list[tuple[Any, Any]]) -> BucketedTaskGeneratorBuilder:
        """Add a bucket with ranges only."""
        bucket_values = [ValueRange(range_min=r[0], range_max=r[1]) for r in ranges]
        self._buckets[key] = bucket_values
        return self

    def add_single_range_bucket(self, key: str, min_val: Any, max_val: Any) -> BucketedTaskGeneratorBuilder:
        """Add a bucket with a single range."""
        bucket_values = [ValueRange(range_min=min_val, range_max=max_val)]
        self._buckets[key] = bucket_values
        return self

    def build(self) -> BucketedTaskGeneratorConfig:
        """Build the BucketedTaskGeneratorConfig."""
        return BucketedTaskGeneratorConfig(base_generator_config=self._base_generator_config, buckets=self._buckets)


class CurriculumBuilder:
    """Builder for Curriculum configurations."""

    @staticmethod
    def random(
        task_gen_config: TaskGeneratorConfig | WeightedTaskSetGeneratorBuilder | BucketedTaskGeneratorBuilder,
    ):  # -> RandomCurriculumBuilder:
        """Create a RandomCurriculum builder."""
        if isinstance(task_gen_config, (WeightedTaskSetGeneratorBuilder, BucketedTaskGeneratorBuilder)):
            task_gen_config = task_gen_config.build()
        raise NotImplementedError("RandomCurriculumBuilder is not available - RandomCurriculumConfig was deleted")
        # return RandomCurriculumBuilder(task_generator_config=task_gen_config)

    @staticmethod
    def learning_progress(
        task_gen_config: TaskGeneratorConfig | WeightedTaskSetGeneratorBuilder | BucketedTaskGeneratorBuilder,
    ) -> LearningProgressCurriculumBuilder:
        """Create a LearningProgressCurriculum builder."""
        if isinstance(task_gen_config, (WeightedTaskSetGeneratorBuilder, BucketedTaskGeneratorBuilder)):
            task_gen_config = task_gen_config.build()
        return LearningProgressCurriculumBuilder(task_generator_config=task_gen_config)


# class RandomCurriculumBuilder:
#     """Builder for RandomCurriculumConfig."""
#
#     def __init__(self, task_generator_config: TaskGeneratorConfig):
#         self._task_generator_config = task_generator_config
#
#     def build(self) -> RandomCurriculumConfig:
#         """Build the RandomCurriculumConfig."""
#         return RandomCurriculumConfig(task_generator_config=self._task_generator_config)


class LearningProgressCurriculumBuilder:
    """Builder for LearningProgressCurriculumConfig."""

    def __init__(self, task_generator_config: TaskGeneratorConfig):
        self._task_generator_config = task_generator_config
        self._n_tasks: int = 100
        self._ema_timescale: float = 0.001
        self._progress_smoothing: float = 0.05
        self._num_active_tasks: int = 16
        self._rand_task_rate: float = 0.25
        self._sample_threshold: int = 10
        self._memory: int = 25

    def with_n_tasks(self, n_tasks: int) -> LearningProgressCurriculumBuilder:
        """Set the number of tasks to generate."""
        self._n_tasks = n_tasks
        return self

    def with_ema_timescale(self, ema_timescale: float) -> LearningProgressCurriculumBuilder:
        """Set the EMA timescale for learning progress."""
        self._ema_timescale = ema_timescale
        return self

    def with_progress_smoothing(self, progress_smoothing: float) -> LearningProgressCurriculumBuilder:
        """Set the progress smoothing factor."""
        self._progress_smoothing = progress_smoothing
        return self

    def with_num_active_tasks(self, num_active_tasks: int) -> LearningProgressCurriculumBuilder:
        """Set the number of active tasks to maintain."""
        self._num_active_tasks = num_active_tasks
        return self

    def with_rand_task_rate(self, rand_task_rate: float) -> LearningProgressCurriculumBuilder:
        """Set the rate of random task selection."""
        self._rand_task_rate = rand_task_rate
        return self

    def with_sample_threshold(self, sample_threshold: int) -> LearningProgressCurriculumBuilder:
        """Set the minimum samples before task becomes active."""
        self._sample_threshold = sample_threshold
        return self

    def with_memory(self, memory: int) -> LearningProgressCurriculumBuilder:
        """Set the number of recent outcomes to remember per task."""
        self._memory = memory
        return self

    def build(self) -> LearningProgressCurriculumConfig:
        """Build the LearningProgressCurriculumConfig."""
        return LearningProgressCurriculumConfig(
            task_generator_config=self._task_generator_config,
            n_tasks=self._n_tasks,
            ema_timescale=self._ema_timescale,
            progress_smoothing=self._progress_smoothing,
            num_active_tasks=self._num_active_tasks,
            rand_task_rate=self._rand_task_rate,
            sample_threshold=self._sample_threshold,
            memory=self._memory,
        )
