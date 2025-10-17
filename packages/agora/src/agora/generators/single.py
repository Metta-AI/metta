"""Single-task generator for fixed task configurations."""

from __future__ import annotations

import random
from typing import Any, ClassVar

from pydantic import ConfigDict, Field

from agora.config import TConfig
from agora.generators.base import TaskGenerator, TaskGeneratorConfig


class SingleTaskGenerator(TaskGenerator[TConfig]):
    """TaskGenerator that always returns the same task configuration.

    Useful for non-curriculum training or as a baseline/child generator.

    Example:
        >>> from pydantic import BaseModel
        >>> class MyConfig(BaseModel):
        ...     difficulty: int = 1
        >>> config = SingleTaskGenerator.Config(env=MyConfig(difficulty=5))
        >>> generator = SingleTaskGenerator(config)
        >>> task = generator.get_task(task_id=0)  # Always returns same config
    """

    class Config(TaskGeneratorConfig["SingleTaskGenerator"]):
        """Configuration for SingleTaskGenerator."""

        model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

        env: Any = Field(description="The task configuration to always return")

    Config: ClassVar[type[Config]]  # type: ignore[misc]

    def __init__(self, config: SingleTaskGenerator.Config):
        """Initialize SingleTaskGenerator.

        Args:
            config: Configuration with the fixed task to return
        """
        super().__init__(config)
        self._config: SingleTaskGenerator.Config = config

    def _generate_task(self, task_id: int, rng: random.Random) -> TConfig:
        """Always return the same task configuration.

        Args:
            task_id: Task identifier (unused for single task)
            rng: Random number generator (unused for single task)

        Returns:
            Deep copy of the configured task
        """
        return self._config.env.model_copy(deep=True)


__all__ = ["SingleTaskGenerator"]
