"""Bucketed task generator for curriculum learning with parameter variation."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any, ClassVar

from pydantic import Field, SerializeAsAny

from agora.config import TConfig
from agora.generators.base import Span, TaskGenerator, TaskGeneratorConfig


class BucketedTaskGenerator(TaskGenerator[TConfig]):
    """TaskGenerator that picks values from buckets and applies them as overrides to a child generator.

    When get_task() is called:
    1. Sample a value from each bucket
    2. Call the child TaskGenerator's get_task()
    3. Apply the sampled bucket values as overrides to the returned config

    This is useful for curriculum learning where you want to vary specific parameters
    (e.g., difficulty, number of obstacles) across tasks.

    Example:
        >>> from agora.generators import SingleTaskGenerator, BucketedTaskGenerator, Span
        >>> base_config = MyConfig(difficulty=1, num_enemies=5)
        >>> child = SingleTaskGenerator.Config(env=base_config)
        >>> bucketed_config = BucketedTaskGenerator.Config(
        ...     child_generator_config=child,
        ...     buckets={
        ...         "difficulty": [1, 2, 3, 4, 5],
        ...         "num_enemies": [Span(5, 10), Span(10, 20)],
        ...     }
        ... )
        >>> generator = BucketedTaskGenerator(bucketed_config)
        >>> task = generator.get_task(task_id=0)  # Random difficulty and num_enemies
    """

    class Config(TaskGeneratorConfig["BucketedTaskGenerator"]):
        """Configuration for BucketedTaskGenerator."""

        child_generator_config: SerializeAsAny[TaskGeneratorConfig[Any]] = Field(
            description="Child task generator configuration"
        )
        buckets: dict[str, Sequence[int | float | str | Span]] = Field(
            default_factory=dict, description="Buckets for sampling, keys are config paths"
        )

        def add_bucket(self, path: str, values: Sequence[int | float | str | Span]) -> BucketedTaskGenerator.Config:
            """Add a bucket of values for a specific configuration path.

            Args:
                path: Dot-separated path to the config field (e.g., "game.difficulty")
                values: Sequence of values or Spans to sample from

            Returns:
                Self for chaining

            Raises:
                AssertionError: If bucket already exists for this path
            """
            assert path not in self.buckets, f"Bucket {path} already exists"
            self.buckets[path] = values
            return self

        @classmethod
        def from_base(cls, base_config: TConfig) -> BucketedTaskGenerator.Config:
            """Create a BucketedTaskGenerator.Config from a base configuration.

            Args:
                base_config: Base task configuration to wrap

            Returns:
                BucketedTaskGenerator config with the base as child
            """
            # Import here to avoid circular dependency
            from agora.generators.single import SingleTaskGenerator

            # Use model_construct to bypass Pydantic validation issues with generic types
            single_config = SingleTaskGenerator.Config.model_construct(env=base_config)
            return cls.model_construct(child_generator_config=single_config, buckets={})

        @classmethod
        def from_mg(cls, mg_config: Any) -> BucketedTaskGenerator.Config:
            """Create from MettaGridConfig (backward compatibility).

            Args:
                mg_config: MettaGridConfig instance

            Returns:
                BucketedTaskGenerator config
            """
            return cls.from_base(mg_config)

    Config: ClassVar[type[Config]]  # type: ignore[misc]

    def __init__(self, config: BucketedTaskGenerator.Config):
        """Initialize BucketedTaskGenerator.

        Args:
            config: Configuration with child generator and buckets

        Raises:
            AssertionError: If buckets is empty
        """
        super().__init__(config)
        self._config: BucketedTaskGenerator.Config = config
        assert config.buckets, "Buckets must be non-empty"
        self._child_generator = config.child_generator_config.create()
        self._last_bucket_values: dict[str, Any] = {}

    def _get_bucket_value(
        self, bucket_values: Sequence[int | float | str | Span], rng: random.Random
    ) -> int | float | str:
        """Sample a value from a bucket.

        Args:
            bucket_values: Sequence of values or Spans to sample from
            rng: Random number generator

        Returns:
            Sampled value (int, float, or str)
        """
        bucket_value: int | float | str | Span = rng.choice(bucket_values)  # type: ignore[arg-type]

        if isinstance(bucket_value, Span):
            min_val, max_val = bucket_value.range_min, bucket_value.range_max
            if isinstance(min_val, int) and isinstance(max_val, int):
                return rng.randint(min_val, max_val)
            elif isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                return rng.uniform(min_val, max_val)

        return bucket_value  # type: ignore[return-value]

    def _generate_task(self, task_id: int, rng: random.Random) -> TConfig:
        """Generate task by calling child generator then applying bucket overrides.

        Args:
            task_id: Task identifier to pass to child generator
            rng: Random number generator for sampling bucket values

        Returns:
            Task configuration with bucket overrides applied
        """
        # First, sample values from each bucket
        overrides = {}
        for key, bucket_values in self._config.buckets.items():
            overrides[key] = self._get_bucket_value(bucket_values, rng)

        # Store the bucket values for the curriculum to access
        self._last_bucket_values = overrides.copy()

        # Get task from the child generator
        task_config = self._child_generator.get_task(task_id)

        # Apply label if specified
        if self._config.label is not None and hasattr(task_config, "label"):
            current_label = getattr(task_config, "label", "")
            if current_label:
                setattr(task_config, "label", f"{current_label}|{self._config.label}")
            else:
                setattr(task_config, "label", self._config.label)

        # Apply the sampled bucket values as overrides
        return self._apply_overrides(task_config, overrides)


__all__ = ["BucketedTaskGenerator"]
