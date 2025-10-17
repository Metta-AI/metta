"""Composite generator that combines multiple task generators with weighted sampling."""

from __future__ import annotations

import random
from typing import Any, ClassVar

from pydantic import Field, SerializeAsAny, field_validator

from agora.config import TConfig
from agora.generators.base import TaskGenerator, TaskGeneratorConfig


class TaskGeneratorSet(TaskGenerator[TConfig]):
    """TaskGenerator that contains a list of TaskGenerators with weights.

    When get_task() is called, rng is initialized with seed, then we sample
    from the list by weight and return child.get_task().

    Example:
        >>> gen1_config = SingleTaskGenerator.Config(env=config1)
        >>> gen2_config = SingleTaskGenerator.Config(env=config2)
        >>> set_config = TaskGeneratorSet.Config(
        ...     task_generators=[gen1_config, gen2_config],
        ...     weights=[0.7, 0.3]
        ... )
        >>> generator = TaskGeneratorSet(set_config)
        >>> task = generator.get_task(task_id=0)  # 70% chance gen1, 30% gen2
    """

    class Config(TaskGeneratorConfig["TaskGeneratorSet"]):
        """Configuration for TaskGeneratorSet."""

        task_generators: list[SerializeAsAny[TaskGeneratorConfig[Any]]] = Field(
            default_factory=list, description="Task generator configurations to sample from"
        )
        weights: list[float] = Field(default_factory=list, description="Weights for sampling each task generator")

        @field_validator("weights")
        @classmethod
        def validate_weights(cls, v: list[float], info: Any) -> list[float]:
            """Ensure weights are positive and match number of generators.

            Args:
                v: List of weights to validate
                info: Validation info containing other field values

            Returns:
                Validated weights

            Raises:
                ValueError: If weights are invalid
            """
            if any(w <= 0 for w in v):
                raise ValueError("All weights must be positive")
            task_gens = info.data.get("task_generators", [])
            if v and len(v) != len(task_gens):
                raise ValueError("Number of weights must match number of task generator configs")
            return v

        def add(self, task_generator: TaskGeneratorConfig[Any], weight: float = 1.0) -> TaskGeneratorSet.Config:
            """Add a task generator to the set with a weight.

            Args:
                task_generator: Generator configuration to add
                weight: Sampling weight (must be positive)

            Returns:
                Self for chaining
            """
            self.task_generators.append(task_generator)  # type: ignore[arg-type]
            self.weights.append(weight)
            return self

    Config: ClassVar[type[Config]]  # type: ignore[misc]

    def __init__(self, config: TaskGeneratorSet.Config):
        """Initialize TaskGeneratorSet.

        Args:
            config: Configuration with list of generators and weights
        """
        super().__init__(config)
        self._config: TaskGeneratorSet.Config = config
        self._sub_task_generators = [gen_config.create() for gen_config in self._config.task_generators]
        self._weights = self._config.weights if self._config.weights else [1.0] * len(self._sub_task_generators)
        self._last_bucket_values: dict[str, Any] = {}

    def _generate_task(self, task_id: int, rng: random.Random) -> TConfig:
        """Generate task by sampling a sub-generator and calling it.

        Args:
            task_id: Task identifier to pass to chosen generator
            rng: Random number generator for choosing generator

        Returns:
            Task configuration from chosen generator
        """
        chosen_generator = rng.choices(self._sub_task_generators, weights=self._weights)[0]
        result = chosen_generator.get_task(task_id)

        # Propagate bucket values if the chosen generator has them
        if hasattr(chosen_generator, "_last_bucket_values"):
            self._last_bucket_values = chosen_generator._last_bucket_values.copy()  # type: ignore[attr-defined]
        else:
            self._last_bucket_values = {}

        return result


__all__ = ["TaskGeneratorSet"]
