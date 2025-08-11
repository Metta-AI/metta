"""Pydantic configuration classes for TaskSet and Curriculum."""

from __future__ import annotations

from typing import Any, ClassVar, Union

from pydantic import ConfigDict, Field, field_validator, model_validator

from metta.common.util.typed_config import ConfigWithBuilder
from metta.rl.env_config import EnvConfig


class TaskSetConfig(ConfigWithBuilder):
    """Base configuration for TaskSet."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )


class WeightedTaskSetItem(ConfigWithBuilder):
    """Configuration for an item in a WeightedTaskSet."""

    env_config: EnvConfig | None = Field(default=None, description="Environment configuration")
    task_set_config: TaskSetConfig | None = Field(default=None, description="Nested task set configuration")
    weight: float = Field(default=1.0, gt=0, description="Weight for sampling this item")

    @model_validator(mode="after")
    def validate_exactly_one_set(self) -> "WeightedTaskSetItem":
        """Ensure exactly one of env_config or task_set_config is set."""
        env_config_set = self.env_config is not None
        task_set_config_set = self.task_set_config is not None

        if not (env_config_set ^ task_set_config_set):  # XOR: exactly one should be True
            raise ValueError("Exactly one of env_config or task_set_config must be set")

        return self


class WeightedTaskSetConfig(TaskSetConfig):
    """Configuration for WeightedTaskSet."""

    items: list[WeightedTaskSetItem] = Field(default_factory=list, description="Items to sample from")
    overrides: dict[str, Any] | list[str] | None = Field(
        default=None, description="Overrides to apply as nested dict or list of 'key: value' strings"
    )


class BucketValue(ConfigWithBuilder):
    """A bucket value that can be a single value or a range."""

    value: Any | None = Field(default=None, description="Single value")
    range_min: float | int | None = Field(default=None, description="Range minimum")
    range_max: float | int | None = Field(default=None, description="Range maximum")

    @field_validator("value", "range_min", "range_max")
    @classmethod
    def validate_value_or_range(cls, v, info):
        """Ensure either value is set OR both range_min and range_max are set."""
        values = info.data
        value = values.get("value")
        range_min = values.get("range_min")
        range_max = values.get("range_max")

        # If all fields are being validated
        if all(k in values for k in ["value", "range_min", "range_max"]):
            has_value = value is not None
            has_range = range_min is not None and range_max is not None

            if not (has_value ^ has_range):  # XOR - exactly one should be true
                raise ValueError("Either 'value' must be set OR both 'range_min' and 'range_max' must be set")

            if has_range and range_min >= range_max:
                raise ValueError("range_min must be less than range_max")

        return v


class BuckettedTaskSetConfig(TaskSetConfig):
    """Configuration for BuckettedTaskSet."""

    base_config: EnvConfig = Field(description="Base environment configuration")
    buckets: dict[str, list[BucketValue]] = Field(
        default_factory=dict, description="Buckets for sampling, keys are config paths"
    )


class CurriculumConfig(ConfigWithBuilder):
    """Base configuration for Curriculum."""

    task_set_config: TaskSetConfig = Field(description="TaskSet configuration")

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )


class RandomCurriculumConfig(CurriculumConfig):
    """Configuration for RandomCurriculum."""


class LearningProgressCurriculumConfig(CurriculumConfig):
    """Configuration for LearningProgressCurriculum."""

    n_tasks: int = Field(default=100, gt=0, description="Number of tasks to generate")
    ema_timescale: float = Field(default=0.001, gt=0, le=1.0, description="EMA timescale for learning progress")
    progress_smoothing: float = Field(default=0.05, ge=0, le=1.0, description="Progress smoothing factor")
    num_active_tasks: int = Field(default=16, gt=0, description="Number of active tasks to maintain")
    rand_task_rate: float = Field(default=0.25, ge=0, le=1.0, description="Rate of random task selection")
    sample_threshold: int = Field(default=10, gt=0, description="Minimum samples before task becomes active")
    memory: int = Field(default=25, gt=0, description="Number of recent outcomes to remember per task")


# Union types for polymorphic configurations
TaskSetConfigUnion = Union[WeightedTaskSetConfig, BuckettedTaskSetConfig]
CurriculumConfigUnion = Union[RandomCurriculumConfig, LearningProgressCurriculumConfig]
