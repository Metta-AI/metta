from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from omegaconf import OmegaConf
from pydantic import ConfigDict, Field, field_validator

from metta.common.util.typed_config import ConfigWithBuilder
from metta.rl.env_config import SystemConfig

logger = logging.getLogger(__name__)


class TaskGeneratorConfig(ConfigWithBuilder):
    """Base configuration for TaskGenerator."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )

    def create(self) -> "TaskGenerator":
        """Create a TaskGenerator from this configuration.

        Subclasses should override this method to create their specific generator type.
        """
        # Base TaskGeneratorConfig creates an empty WeightedTaskSetGenerator
        # This is only reached if someone instantiates the base class directly
        # Import here to avoid circular dependency
        from cogworks.curriculum.task.generator import WeightedTaskSetGenerator, WeightedTaskSetGeneratorConfig

        weighted_config = WeightedTaskSetGeneratorConfig(items=[])
        return WeightedTaskSetGenerator(weighted_config)


class TaskGenerator(ABC):
    """Base class for generating tasks with deterministic seeding.

    TaskGenerator supports .get_task(task_id) where task_id is used as the seed.
    It should always be constructed with a TaskGeneratorConfig.
    """

    def __init__(self, config: TaskGeneratorConfig):
        self._config = config

    def get_task(self, task_id: int) -> SystemConfig:
        """Generate a task (EnvConfig) using task_id as seed."""
        rng = self._init_rng(task_id)
        return self._generate_task(task_id, rng)

    @abstractmethod
    def _generate_task(self, task_id: int, rng: random.Random) -> SystemConfig:
        """Generate a task with the given task_id and RNG.

        This method should be overridden by subclasses to implement
        their specific task generation logic.

        Args:
            task_id: The task identifier used as the seed
            rng: A seeded random number generator

        Returns:
            An EnvConfig for the generated task
        """
        pass

    def _init_rng(self, task_id: int) -> random.Random:
        """Initialize and return a seeded random number generator."""
        rng = random.Random()
        rng.seed(task_id)
        return rng

    def _apply_overrides(self, env_config: SystemConfig, overrides: dict[str, Any]) -> SystemConfig:
        """Apply overrides to an EnvConfig using dot-separated keys."""
        if not overrides:
            return env_config

        # Convert to dict, apply overrides using OmegaConf
        config_dict = env_config.model_dump()
        config_dict = OmegaConf.create(config_dict)

        for key, value in overrides.items():
            OmegaConf.update(config_dict, key, value, merge=True)

        config_dict = OmegaConf.to_container(config_dict)

        return SystemConfig.model_validate(config_dict)


class SingleTaskGeneratorConfig(TaskGeneratorConfig):
    """Configuration for SingleTaskGenerator."""

    env_config: SystemConfig = Field(description="The environment configuration to always return")

    def create(self) -> "SingleTaskGenerator":
        """Create a SingleTaskGenerator from this configuration."""
        return SingleTaskGenerator(self)


class SingleTaskGenerator(TaskGenerator):
    """TaskGenerator that always returns the same EnvConfig."""

    def __init__(self, config: SingleTaskGeneratorConfig):
        super().__init__(config)
        self._config: SingleTaskGeneratorConfig = config

    def _generate_task(self, task_id: int, rng: random.Random) -> SystemConfig:
        """Always return the same EnvConfig."""
        return self._config.env_config


class WeightedTaskSetGeneratorItem(ConfigWithBuilder):
    """Configuration for an item in a WeightedTaskSetGenerator."""

    task_generator_config: TaskGeneratorConfig = Field(description="Nested task generator configuration")
    weight: float = Field(default=1.0, gt=0, description="Weight for sampling this item")


class WeightedTaskSetGeneratorConfig(TaskGeneratorConfig):
    """Configuration for WeightedTaskSetGenerator."""

    items: list[WeightedTaskSetGeneratorItem] = Field(min_length=1, description="Items to sample from")
    overrides: dict[str, Any] | None = Field(
        default=None, description="Overrides to apply as dict with dot-separated keys"
    )

    def create(self) -> "WeightedTaskSetGenerator":
        """Create a WeightedTaskSetGenerator from this configuration."""
        return WeightedTaskSetGenerator(self)


class WeightedTaskSetGenerator(TaskGenerator):
    """TaskGenerator that contains a list of TaskGenerators with weights.

    When get_task() is called, rng is initialized with seed, then we sample
    from the list by weight and return child.get_task().
    """

    def __init__(self, config: WeightedTaskSetGeneratorConfig):
        super().__init__(config)

        self._config: WeightedTaskSetGeneratorConfig = config

        self._sub_task_generators: list[TaskGenerator] = []
        self._weights: list[float] = []
        for item_config in self._config.items:
            task_gen = item_config.task_generator_config.create()
            self._sub_task_generators.append(task_gen)
            self._weights.append(item_config.weight)

        self._overrides = config.overrides or {}

    def _generate_task(self, task_id: int, rng: random.Random) -> SystemConfig:
        """Sample from items by weight and return EnvConfig."""
        if not self._sub_task_generators:
            raise ValueError("No items to sample from")

        # Sample by weight
        selected_generator = rng.choices(self._sub_task_generators, weights=self._weights)[0]

        task = selected_generator.get_task(task_id)
        return self._apply_overrides(task, self._overrides)


class ValueRange(ConfigWithBuilder):
    """A range of values with minimum and maximum bounds."""

    range_min: float | int = Field(description="Range minimum")
    range_max: float | int = Field(description="Range maximum")

    @field_validator("range_max")
    @classmethod
    def validate_range(cls, v, info):
        """Ensure range_min is less than range_max."""
        range_min = info.data.get("range_min")
        if range_min is not None and range_min >= v:
            raise ValueError("range_min must be less than range_max")
        return v


class BucketedTaskGeneratorConfig(TaskGeneratorConfig):
    """Configuration for BucketedTaskGenerator."""

    child_generator_config: TaskGeneratorConfig = Field(description="Child task generator configuration")
    buckets: dict[str, list[int | float | str | ValueRange]] = Field(
        default_factory=dict, min_length=1, description="Buckets for sampling, keys are config paths"
    )

    def create(self) -> "BucketedTaskGenerator":
        """Create a BucketedTaskGenerator from this configuration."""
        return BucketedTaskGenerator(self)


class BucketedTaskGenerator(TaskGenerator):
    """TaskGenerator that picks values from buckets and applies them as overrides to a child generator.

    When get_task() is called:
    1. Sample a value from each bucket
    2. Call the child TaskGenerator's get_task()
    3. Apply the sampled bucket values as overrides to the returned EnvConfig
    """

    def __init__(self, config: BucketedTaskGeneratorConfig):
        super().__init__(config)
        self._config: BucketedTaskGeneratorConfig = config
        self._child_generator = config.child_generator_config.create()

    def _get_bucket_value(self, bucket_values: list[int | float | str | ValueRange], rng: random.Random) -> Any:
        bucket_value = rng.choice(bucket_values)

        if isinstance(bucket_value, ValueRange):
            min_val, max_val = bucket_value.range_min, bucket_value.range_max
            if isinstance(min_val, int) and isinstance(max_val, int):
                bucket_value = rng.randint(min_val, max_val)
            elif isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                bucket_value = rng.uniform(min_val, max_val)
        return bucket_value

    def _generate_task(self, task_id: int, rng: random.Random) -> SystemConfig:
        """Generate task by calling child generator then applying bucket overrides."""
        # First, sample values from each bucket
        overrides = {}
        for key, bucket_values in self._config.buckets.items():
            sampled_value = self._get_bucket_value(bucket_values, rng)
            overrides[key] = sampled_value

        # Get task from the child generator
        env_config = self._child_generator.get_task(task_id)

        # Apply the sampled bucket values as overrides
        return self._apply_overrides(env_config, overrides)
