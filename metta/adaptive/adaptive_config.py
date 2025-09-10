"""Simplified configuration for adaptive experiments."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AdaptiveConfig:
    """
    Simple, focused configuration for adaptive experiments.

    Contains only the essential settings - no nested hierarchies or unused fields.
    """

    # Core experiment limits
    max_trials: int = 100
    max_parallel: int = 10

    # Execution settings
    monitoring_interval: int = 60
    train_recipe: str = "experiments.recipes.arena.train"
    eval_recipe: str = "experiments.recipes.arena.evaluate"
    resume: bool = False # Whether we are resuming from an existing experiment
    #TODO: In future, this check should be automatic.

    # Optional settings
    base_overrides: dict[str, Any] = field(default_factory=dict)
    experiment_tags: list[str] = field(default_factory=list)

    def validate(self) -> None:
        """Validate configuration values"""
        if self.max_trials <= 0:
            raise ValueError("max_trials must be positive")
        if self.max_parallel <= 0:
            raise ValueError("max_parallel must be positive")
        if self.monitoring_interval <= 0:
            raise ValueError("monitoring_interval must be positive")
