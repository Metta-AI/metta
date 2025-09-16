"""Simplified configuration for adaptive experiments."""

from dataclasses import dataclass, field


@dataclass
class AdaptiveConfig:
    """
    Simple, focused configuration for adaptive experiments.

    Contains only the essential settings - no nested hierarchies or unused fields.
    """

    # Core experiment limits
    max_trials: int = 100
    max_parallel: int = 1

    # Execution settings
    monitoring_interval: int = 60
    resume: bool = False  # Whether we are resuming from an existing experiment
    # TODO: In future, this check should be automatic.

    # Optional settings
    experiment_tags: list[str] = field(default_factory=list)

    def validate(self) -> None:
        """Validate configuration values"""
        if self.max_trials <= 0:
            raise ValueError("max_trials must be positive")
        if self.max_parallel <= 0:
            raise ValueError("max_parallel must be positive")
        if self.monitoring_interval <= 0:
            raise ValueError("monitoring_interval must be positive")
