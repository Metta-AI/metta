from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from typing import Annotated, Any, ClassVar, Optional, Sequence, Type, TypeVar

from pydantic import (
    ConfigDict,
    Field,
    SerializeAsAny,
    WrapValidator,
    field_validator,
    model_serializer,
)
from typing_extensions import Generic

from mettagrid.config import Config
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.util.module import load_symbol

logger = logging.getLogger(__name__)

TTaskGenerator = TypeVar("TTaskGenerator", bound="TaskGenerator")


class TaskGeneratorConfig(Config, Generic[TTaskGenerator]):
    """Base configuration for TaskGenerator.

    Subclasses *optionally* know which TaskGenerator they build via `_generator_cls`
    (auto-filled when nested inside a TaskGenerator subclass).
    """

    _generator_cls: ClassVar[Optional[Type[TTaskGenerator]]] = None  # type: ignore[misc]

    # pydantic configuration
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )

    label: Optional[str] = Field(default=None, description="Label for the task generator")

    overrides: dict[str, Any] = Field(
        default_factory=dict, description="Overrides to apply as dict with dot-separated keys"
    )

    def create(self) -> TTaskGenerator:
        """Instantiate the bound TaskGenerator.

        Subclasses nested under a TaskGenerator automatically bind `_generator_cls`.
        If you define a standalone Config subclass, either set `_generator_cls`
        on the class or override `create()`.
        """
        return self.generator_cls()(self)  # type: ignore[call-arg]

    @classmethod
    def generator_cls(cls) -> Type[TTaskGenerator]:
        if cls._generator_cls is None:
            raise TypeError(
                f"{cls.__name__} is not bound to a TaskGenerator; "
                f"either define it nested under the generator or set `_generator_cls`."
            )
        return cls._generator_cls

    def to_curriculum(self, num_active_tasks: int = 1000, algorithm_config=None):
        """Create a CurriculumConfig from this TaskGeneratorConfig."""
        from metta.cogworks.curriculum.curriculum import CurriculumConfig
        from metta.cogworks.curriculum.learning_progress_algorithm import LearningProgressConfig

        if algorithm_config is None:
            algorithm_config = LearningProgressConfig()

        return CurriculumConfig(
            task_generator=self, num_active_tasks=num_active_tasks, algorithm_config=algorithm_config
        )

    @model_serializer(mode="wrap")
    def _serialize_with_type(self, handler):
        """Ensure YAML/JSON dumps always include a 'type' with a nice FQCN."""
        data = handler(self)  # dict of the model's fields
        typ_cls: Type[Any] = self._generator_cls or self.__class__
        # Prefer the *generator* class if known, fall back to the config class
        type_str = f"{typ_cls.__module__}.{typ_cls.__name__}"
        return {"type": type_str, **data}


class TaskGenerator(ABC):
    """Base class for generating tasks with deterministic seeding.

    TaskGenerator supports .get_task(task_id) where task_id is used as the seed.
    It should always be constructed with a TaskGeneratorConfig.

    If a subclass declares a nested class `Config` that inherits from TaskGeneratorConfig,
    it will be *automatically bound*.
    """

    Config: ClassVar[type[TaskGeneratorConfig[Any]]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Auto-bind nested Config class to this generator
        if hasattr(cls, "Config"):
            cls.Config._generator_cls = cls  # type: ignore[assignment]

    def __init__(self, config: TaskGeneratorConfig):
        self._config = config
        self._overrides = config.overrides

    def get_task(self, task_id: int) -> MettaGridConfig:
        """Generate a task (MettaGridConfig) using task_id as seed."""
        rng = random.Random()
        rng.seed(task_id)
        return self._apply_overrides(self._generate_task(task_id, rng), self._config.overrides)

    @abstractmethod
    def _generate_task(self, task_id: int, rng: random.Random) -> MettaGridConfig:
        """Generate a task with the given task_id and RNG.

        This method should be overridden by subclasses to implement
        their specific task generation logic.

        Args:
            task_id: The task identifier used as the seed
            rng: A seeded random number generator

        Returns:
            An MettaGridConfig for the generated task
        """
        raise NotImplementedError("TaskGenerator._generate_task() must be overridden by subclasses")

    def _apply_overrides(self, mg_config: MettaGridConfig, overrides: dict[str, Any]) -> MettaGridConfig:
        """Apply overrides to an MettaGridConfig using dot-separated keys."""
        if not overrides:
            return mg_config

        mg_config.update(overrides)
        return mg_config


################################################################################
# SingleTaskGenerator
################################################################################
class SingleTaskGenerator(TaskGenerator):
    """TaskGenerator that always returns the same MettaGridConfig."""

    class Config(TaskGeneratorConfig["SingleTaskGenerator"]):
        """Configuration for SingleTaskGenerator."""

        env: MettaGridConfig = Field(description="The environment configuration to always return")

    def __init__(self, config: "SingleTaskGenerator.Config"):
        super().__init__(config)
        self._config = config

    def _generate_task(self, task_id: int, rng: random.Random) -> MettaGridConfig:
        """Always return the same MettaGridConfig."""
        return self._config.env.model_copy(deep=True)


################################################################################
# TaskGeneratorSet
################################################################################
class TaskGeneratorSet(TaskGenerator):
    """TaskGenerator that contains a list of TaskGenerators with weights.

    When get_task() is called, rng is initialized with seed, then we sample
    from the list by weight and return child.get_task().
    """

    class Config(TaskGeneratorConfig["TaskGeneratorSet"]):
        """Configuration for TaskGeneratorSet."""

        task_generators: list[AnyTaskGeneratorConfig] = Field(
            default_factory=list, description="Task generator configurations to sample from"
        )
        weights: list[float] = Field(default_factory=list, description="Weights for sampling each task generator")

        @field_validator("weights")
        @classmethod
        def validate_weights(cls, v, info):
            """Ensure weights are positive."""
            if any(w <= 0 for w in v):
                raise ValueError("All weights must be positive")
            task_gens = info.data.get("task_generators", [])
            if v and len(v) != len(task_gens):
                raise ValueError("Number of weights must match number of task generator configs")
            return v

        def add(self, task_generator: AnyTaskGeneratorConfig, weight: float = 1.0) -> "TaskGeneratorSet.Config":
            """Add a task generator to the set with a weight."""
            self.task_generators.append(task_generator)
            self.weights.append(weight)
            return self

    def __init__(self, config: "TaskGeneratorSet.Config"):
        super().__init__(config)
        self._config = config
        self._sub_task_generators = [gen_config.create() for gen_config in self._config.task_generators]
        self._weights = self._config.weights if self._config.weights else [1.0] * len(self._sub_task_generators)

    def _generate_task(self, task_id: int, rng: random.Random) -> MettaGridConfig:
        chosen_generator = rng.choices(self._sub_task_generators, weights=self._weights)[0]
        result = chosen_generator.get_task(task_id)

        # Propagate bucket values if the chosen generator has them
        if hasattr(chosen_generator, "_last_bucket_values"):
            self._last_bucket_values = chosen_generator._last_bucket_values.copy()
        else:
            self._last_bucket_values = {}

        return result


################################################################################
# BucketedTaskGenerator
################################################################################
class Span(Config):
    """A range of values with minimum and maximum bounds."""

    range_min: float | int = Field(description="Range minimum")
    range_max: float | int = Field(description="Range maximum")

    def __init__(self, range_min: float | int | None = None, range_max: float | int | None = None, **kwargs):
        """Initialize Span with positional arguments or keyword arguments."""
        if range_min is not None and range_max is not None:
            # Called with positional arguments
            super().__init__(range_min=range_min, range_max=range_max, **kwargs)
        else:
            # Called with keyword arguments (normal Pydantic behavior)
            super().__init__(**kwargs)

    @field_validator("range_max")
    @classmethod
    def validate_range(cls, v, info):
        """Ensure range_min is less than range_max."""
        range_min = info.data.get("range_min")
        if range_min is not None and range_min >= v:
            raise ValueError("range_min must be less than range_max")
        return v

    def __str__(self) -> str:
        return f"{self.range_min}-{self.range_max}"


class BucketedTaskGenerator(TaskGenerator):
    """TaskGenerator that picks values from buckets and applies them as overrides to a child generator.

    When get_task() is called:
    1. Sample a value from each bucket
    2. Call the child TaskGenerator's get_task()
    3. Apply the sampled bucket values as overrides to the returned MettaGridConfig
    """

    class Config(TaskGeneratorConfig["BucketedTaskGenerator"]):
        """Configuration for BucketedTaskGenerator."""

        child_generator_config: AnyTaskGeneratorConfig = Field(description="Child task generator configuration")
        buckets: dict[str, Sequence[int | float | str | Span]] = Field(
            default_factory=dict, description="Buckets for sampling, keys are config paths"
        )

        def add_bucket(self, path: str, values: Sequence[int | float | str | Span]) -> "BucketedTaskGenerator.Config":
            """Add a bucket of values for a specific configuration path."""
            assert path not in self.buckets, f"Bucket {path} already exists"
            self.buckets[path] = values
            return self

        @classmethod
        def from_mg(cls, mg_config: MettaGridConfig) -> "BucketedTaskGenerator.Config":
            """Create a BucketedTaskGenerator.Config from an MettaGridConfig."""
            return cls(child_generator_config=SingleTaskGenerator.Config(env=mg_config))

    def __init__(self, config: "BucketedTaskGenerator.Config"):
        super().__init__(config)
        self._config = config
        assert config.buckets, "Buckets must be non-empty"
        self._child_generator = config.child_generator_config.create()

    def _get_bucket_value(self, bucket_values: Sequence[int | float | str | Span], rng: random.Random) -> Any:
        bucket_value = rng.choice(bucket_values)

        if isinstance(bucket_value, Span):
            min_val, max_val = bucket_value.range_min, bucket_value.range_max
            if isinstance(min_val, int) and isinstance(max_val, int):
                bucket_value = rng.randint(min_val, max_val)
            elif isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                bucket_value = rng.uniform(min_val, max_val)
        return bucket_value

    def _generate_task(self, task_id: int, rng: random.Random) -> MettaGridConfig:
        """Generate task by calling child generator then applying bucket overrides."""
        # First, sample values from each bucket
        overrides = {}
        for key, bucket_values in self._config.buckets.items():
            overrides[key] = self._get_bucket_value(bucket_values, rng)

        # Store the bucket values for the curriculum to access
        self._last_bucket_values = overrides.copy()

        # Get task from the child generator
        mg_config = self._child_generator.get_task(task_id)
        if self._config.label is not None:
            mg_config.label += "|" + self._config.label

        # Apply the sampled bucket values as overrides
        return self._apply_overrides(mg_config, overrides)


def _validate_open_task_generator(v: Any, handler):
    """Accepts any of:
    - a TaskGeneratorConfig instance (already specific)
    - a dict with {"type": "<FQCN-of-TaskGenerator-or-Config>", ...params...}
    - anything else -> let the default handler try (will error if invalid)
    """
    if isinstance(v, TaskGeneratorConfig):
        return v

    if isinstance(v, dict):
        t = v.get("type")
        if t is None:
            # try default handler first (e.g., if the default type is already implied)
            return handler(v)

        # Import the symbol named in 'type'
        target = load_symbol(t) if isinstance(t, str) else t

        # If it's a Generator, use its nested Config
        if isinstance(target, type) and issubclass(target, TaskGenerator):
            # Special handling for known task generators
            if target is SingleTaskGenerator:
                data = {k: v for k, v in v.items() if k != "type"}
                return SingleTaskGenerator.Config.model_validate(data)
            elif target is TaskGeneratorSet:
                data = {k: v for k, v in v.items() if k != "type"}
                return TaskGeneratorSet.Config.model_validate(data)
            elif target is BucketedTaskGenerator:
                data = {k: v for k, v in v.items() if k != "type"}
                return BucketedTaskGenerator.Config.model_validate(data)
            else:
                # Generic handling for unknown task generators
                cfg_model = getattr(target, "Config", None)
                if not (isinstance(cfg_model, type) and issubclass(cfg_model, TaskGeneratorConfig)):
                    raise TypeError(f"{target.__name__} must define a nested class Config(TaskGeneratorConfig).")
                data = {k: v for k, v in v.items() if k != "type"}
                return cfg_model.model_validate(data)

        # If it's already a Config subclass, validate with it directly
        if isinstance(target, type) and issubclass(target, TaskGeneratorConfig):
            data = {k: v for k, v in v.items() if k != "type"}
            return target.model_validate(data)

        raise TypeError(
            f"'type' must point to a TaskGenerator subclass or a TaskGeneratorConfig subclass; got {target!r}"
        )

    # Fallback to the normal validator (will raise a decent error)
    return handler(v)


AnyTaskGeneratorConfig = SerializeAsAny[
    Annotated[TaskGeneratorConfig[Any], WrapValidator(_validate_open_task_generator)]
]
