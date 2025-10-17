"""Base classes and protocols for task generation."""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_serializer

from agora.config import TConfig

logger = logging.getLogger(__name__)

# Type variable for TaskGenerator subclasses
TTaskGenerator = TypeVar("TTaskGenerator", bound="TaskGenerator")


class Span:
    """A range of values with minimum and maximum bounds for parameter variation."""

    def __init__(self, range_min: float | int, range_max: float | int):
        """Initialize Span with min and max values.

        Args:
            range_min: Minimum value in the range
            range_max: Maximum value in the range

        Raises:
            ValueError: If range_min >= range_max
        """
        if range_min >= range_max:
            raise ValueError(f"range_min ({range_min}) must be less than range_max ({range_max})")
        self.range_min = range_min
        self.range_max = range_max

    def sample(self, rng: random.Random) -> float:
        """Sample a random value from this range.

        Args:
            rng: Random number generator to use

        Returns:
            Random value in [range_min, range_max]
        """
        return rng.uniform(self.range_min, self.range_max)

    def __repr__(self) -> str:
        return f"Span({self.range_min}, {self.range_max})"


class TaskGeneratorConfig(BaseModel, ABC, Generic[TTaskGenerator]):
    """Base configuration for TaskGenerator.

    Subclasses optionally know which TaskGenerator they build via `_generator_cls`
    (auto-filled when nested inside a TaskGenerator subclass).
    """

    _generator_cls: ClassVar[type[TTaskGenerator] | None] = None

    # Pydantic configuration
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    label: str | None = Field(default=None, description="Label for the task generator")

    overrides: dict[str, Any] = Field(
        default_factory=dict, description="Overrides to apply as dict with dot-separated keys"
    )

    def create(self) -> TTaskGenerator:
        """Instantiate the bound TaskGenerator.

        Subclasses nested under a TaskGenerator automatically bind `_generator_cls`.
        If you define a standalone Config subclass, either set `_generator_cls`
        on the class or override `create()`.

        Returns:
            Instance of the TaskGenerator

        Raises:
            TypeError: If _generator_cls is not set
        """
        return self.generator_cls()(self)  # type: ignore[call-arg]

    def to_curriculum(self, **kwargs: Any) -> Any:
        """Create a CurriculumConfig from this task generator config.

        Backward compatibility method.

        Args:
            **kwargs: Additional arguments to pass to CurriculumConfig

        Returns:
            CurriculumConfig instance
        """
        from agora.curriculum import CurriculumConfig

        return CurriculumConfig(task_generator=self, **kwargs)

    @classmethod
    def generator_cls(cls) -> type[TTaskGenerator]:
        """Get the TaskGenerator class this config is bound to.

        Returns:
            The TaskGenerator class

        Raises:
            TypeError: If _generator_cls is not set
        """
        if cls._generator_cls is None:
            raise TypeError(
                f"{cls.__name__} is not bound to a TaskGenerator; "
                f"either define it nested under the generator or set `_generator_cls`."
            )
        return cls._generator_cls

    @model_serializer(mode="wrap")
    def _serialize_with_type(self, handler: Any) -> dict[str, Any]:
        """Ensure YAML/JSON dumps always include a 'type' with a nice FQCN."""
        data = handler(self)  # dict of the model's fields
        typ_cls: type[Any] = self._generator_cls or self.__class__
        # Prefer the *generator* class if known, fall back to the config class
        type_str = f"{typ_cls.__module__}.{typ_cls.__name__}"
        return {"type": type_str, **data}


class TaskGenerator(ABC, Generic[TConfig]):
    """Base class for generating tasks with deterministic seeding.

    TaskGenerator supports .get_task(task_id) where task_id is used as the seed.
    It should always be constructed with a TaskGeneratorConfig.

    If a subclass declares a nested class `Config` that inherits from TaskGeneratorConfig,
    it will be *automatically bound*.

    Type parameter TConfig should be a concrete config type that implements TaskConfig protocol.
    """

    Config: ClassVar[type[TaskGeneratorConfig[Any]]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-bind nested Config class to this generator."""
        super().__init_subclass__(**kwargs)
        # Auto-bind nested Config class to this generator
        if hasattr(cls, "Config"):
            cls.Config._generator_cls = cls  # type: ignore[assignment]

    def __init__(self, config: TaskGeneratorConfig[Any]):
        """Initialize TaskGenerator with configuration.

        Args:
            config: TaskGeneratorConfig instance with generation parameters
        """
        self._config = config
        self._overrides = config.overrides
        self._reference_task: TConfig | None = None

    def get_task(self, task_id: int) -> TConfig:
        """Generate a task configuration using task_id as seed.

        Args:
            task_id: Unique identifier for the task, used as RNG seed

        Returns:
            Task configuration (any type implementing TaskConfig protocol)
        """
        rng = random.Random()
        rng.seed(task_id)
        task_config = self._apply_overrides(self._generate_task(task_id, rng), self._config.overrides)

        # Validate invariants across all generated tasks
        self._validate_task_invariants(task_config, task_id)

        return task_config

    def _validate_task_invariants(self, task_config: TConfig, task_id: int) -> None:
        """Ensure critical environment parameters don't change across tasks.

        This validates that the number of resources, actions, and agents remain
        consistent across all tasks generated by this TaskGenerator. This prevents
        issues where evaluation uses different environment configurations than training.

        Args:
            task_config: The generated task configuration to validate
            task_id: The task ID being generated (for error messages)

        Raises:
            AssertionError: If task configuration doesn't match reference invariants
        """
        if self._reference_task is None:
            # First task - establish reference invariants
            self._reference_task = task_config
            return

        ref = self._reference_task

        # Try to validate if this looks like a MettaGridConfig
        # For other config types, skip validation
        if not hasattr(task_config, "game"):
            return

        # Validate action count consistency
        try:
            current_action_count = sum(
                1 for action in task_config.game.actions.model_dump().values() if action.get("enabled", True)
            )
            ref_action_count = sum(
                1 for action in ref.game.actions.model_dump().values() if action.get("enabled", True)
            )

            assert current_action_count == ref_action_count, (
                f"TaskGenerator produced inconsistent action count for task {task_id}: "
                f"expected {ref_action_count}, got {current_action_count}. "
                f"Actions must remain constant across all curriculum tasks."
            )
        except AttributeError:
            # No actions attribute, skip this check
            pass

        # Validate resource count consistency
        try:
            assert len(task_config.game.resource_names) == len(ref.game.resource_names), (
                f"TaskGenerator produced inconsistent resource count for task {task_id}: "
                f"expected {len(ref.game.resource_names)}, got {len(task_config.game.resource_names)}. "
                f"Resources must remain constant across all curriculum tasks."
            )
        except AttributeError:
            # No resource_names attribute, skip this check
            pass

        # Validate num_agents consistency
        try:
            assert task_config.game.num_agents == ref.game.num_agents, (
                f"TaskGenerator produced inconsistent agent count for task {task_id}: "
                f"expected {ref.game.num_agents}, got {task_config.game.num_agents}. "
                f"Number of agents must remain constant across all curriculum tasks."
            )
        except AttributeError:
            # No num_agents attribute, skip this check
            pass

    @abstractmethod
    def _generate_task(self, task_id: int, rng: random.Random) -> TConfig:
        """Generate a task with the given task_id and RNG.

        This method should be overridden by subclasses to implement
        their specific task generation logic.

        Args:
            task_id: The task identifier used as the seed
            rng: A seeded random number generator

        Returns:
            A task configuration of type TConfig
        """
        raise NotImplementedError("TaskGenerator._generate_task() must be overridden by subclasses")

    def _apply_overrides(self, task_config: TConfig, overrides: dict[str, Any]) -> TConfig:
        """Apply overrides to a task configuration using dot-separated keys.

        Args:
            task_config: Task configuration to modify
            overrides: Dictionary of overrides with dot-separated keys

        Returns:
            Modified task configuration
        """
        if not overrides:
            return task_config

        # Assume task_config has an update method (Pydantic models do)
        if hasattr(task_config, "update"):
            task_config.update(overrides)  # type: ignore[attr-defined]

        return task_config


__all__ = [
    "Span",
    "TaskGeneratorConfig",
    "TaskGenerator",
    "TTaskGenerator",
]
