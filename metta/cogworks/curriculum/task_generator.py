from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from omegaconf import OmegaConf
from pydantic import ConfigDict, Field, field_validator

from metta.common.util.config import Config
from metta.mettagrid.mettagrid_config import EnvConfig

logger = logging.getLogger(__name__)


class TaskGeneratorConfig(Config):
    """Base configuration for TaskGenerator."""

    # pydantic configuration
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )

    overrides: dict[str, Any] = Field(
        default_factory=dict, description="Overrides to apply as dict with dot-separated keys"
    )

    def create(self) -> "TaskGenerator":
        """Create a TaskGenerator from this configuration.

        Subclasses should override this method to create their specific generator type.
        """
        raise NotImplementedError("TaskGeneratorConfig.create() must be overridden by subclasses")


class TaskGenerator(ABC):
    """Base class for generating tasks with deterministic seeding.

    TaskGenerator supports .get_task(task_id) where task_id is used as the seed.
    It should always be constructed with a TaskGeneratorConfig.
    """

    def __init__(self, config: TaskGeneratorConfig):
        self._config = config
        self._overrides = config.overrides

    def get_task(self, task_id: int) -> EnvConfig:
        """Generate a task (EnvConfig) using task_id as seed."""
        rng = random.Random()
        rng.seed(task_id)
        return self._apply_overrides(self._generate_task(task_id, rng), self._config.overrides)

    @abstractmethod
    def _generate_task(self, task_id: int, rng: random.Random) -> EnvConfig:
        """Generate a task with the given task_id and RNG.

        This method should be overridden by subclasses to implement
        their specific task generation logic.

        Args:
            task_id: The task identifier used as the seed
            rng: A seeded random number generator

        Returns:
            An EnvConfig for the generated task
        """
        raise NotImplementedError("TaskGenerator._generate_task() must be overridden by subclasses")

    def _apply_overrides(self, env_config: EnvConfig, overrides: dict[str, Any]) -> EnvConfig:
        """Apply overrides to an EnvConfig using dot-separated keys."""
        if not overrides:
            return env_config

        # Convert to dict, apply overrides using OmegaConf
        config_dict = OmegaConf.create(env_config.model_dump())

        for key, value in overrides.items():
            OmegaConf.update(config_dict, key, value, merge=True)

        config_dict = OmegaConf.to_container(config_dict)

        return EnvConfig.model_validate(config_dict)


################################################################################
# SingleTaskGenerator
################################################################################


class SingleTaskGeneratorConfig(TaskGeneratorConfig):
    """Configuration for SingleTaskGenerator."""

    env_config: EnvConfig = Field(description="The environment configuration to always return")

    def create(self) -> "SingleTaskGenerator":
        """Create a SingleTaskGenerator from this configuration."""
        return SingleTaskGenerator(self)


class SingleTaskGenerator(TaskGenerator):
    """TaskGenerator that always returns the same EnvConfig."""

    def __init__(self, config: SingleTaskGeneratorConfig):
        super().__init__(config)
        self._config: SingleTaskGeneratorConfig = config

    def _generate_task(self, task_id: int, rng: random.Random) -> EnvConfig:
        """Always return the same EnvConfig."""
        return self._config.env_config


################################################################################
# TaskGeneratorSet
################################################################################


class TaskGeneratorSetConfig(TaskGeneratorConfig):
    """Configuration for TaskGeneratorSet."""

    task_generator_configs: list[TaskGeneratorConfig] = Field(
        min_length=1, description="Task generator configurations to sample from"
    )
    weights: list[float] = Field(min_length=1, description="Weights for sampling each task generator")

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, v, info):
        """Ensure weights are positive."""
        if any(w <= 0 for w in v):
            raise ValueError("All weights must be positive")
        if len(v) != len(info.data.get("task_generator_configs", [])):
            raise ValueError("Number of weights must match number of task generator configs")
        return v

    def create(self) -> "TaskGeneratorSet":
        """Create a TaskGeneratorSet from this configuration."""
        return TaskGeneratorSet(self)


class TaskGeneratorSet(TaskGenerator):
    """TaskGenerator that contains a list of TaskGenerators with weights.

    When get_task() is called, rng is initialized with seed, then we sample
    from the list by weight and return child.get_task().
    """

    def __init__(self, config: TaskGeneratorSetConfig):
        super().__init__(config)

        self._config: TaskGeneratorSetConfig = config

        self._sub_task_generators = [config.create() for config in self._config.task_generator_configs]
        self._weights = self._config.weights

    def _generate_task(self, task_id: int, rng: random.Random) -> EnvConfig:
        return rng.choices(self._sub_task_generators, weights=self._weights)[0].get_task(task_id)


################################################################################
# BucketedTaskGenerator
################################################################################


class ValueRange(Config):
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

    @classmethod
    def vr(cls, range_min: float | int, range_max: float | int) -> "ValueRange":
        """Create a ValueRange from a range_min and range_max."""
        return cls(range_min=range_min, range_max=range_max)


class BucketedTaskGeneratorConfig(TaskGeneratorConfig):
    """Configuration for BucketedTaskGenerator."""

    child_generator_config: TaskGeneratorConfig = Field(description="Child task generator configuration")
    buckets: dict[str, list[int | float | str | ValueRange]] = Field(
        default_factory=dict, description="Buckets for sampling, keys are config paths"
    )

    def add_bucket(self, path: str, values: list[int | float | str | ValueRange]) -> "BucketedTaskGeneratorConfig":
        """Add a bucket of values for a specific configuration path."""
        assert path not in self.buckets, f"Bucket {path} already exists"
        self.buckets[path] = values
        return self

    def create(self) -> "BucketedTaskGenerator":
        """Create a BucketedTaskGenerator from this configuration."""
        return BucketedTaskGenerator(self)

    @classmethod
    def from_env_config(cls, env_config: EnvConfig) -> BucketedTaskGeneratorConfig:
        """Create a BucketedTaskGeneratorConfig from an EnvConfig."""
        return cls(child_generator_config=SingleTaskGeneratorConfig(env_config=env_config))


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
        assert config.buckets, "Buckets must be non-empty"
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

    def _generate_task(self, task_id: int, rng: random.Random) -> EnvConfig:
        """Generate task by calling child generator then applying bucket overrides."""
        # First, sample values from each bucket
        overrides = {}
        for key, bucket_values in self._config.buckets.items():
            overrides[key] = self._get_bucket_value(bucket_values, rng)

        # Get task from the child generator
        env_config = self._child_generator.get_task(task_id)

        # Apply the sampled bucket values as overrides
        return self._apply_overrides(env_config, overrides)
