"""Simplified configuration for adaptive experiments."""

from pydantic import BaseModel, Field


class AdaptiveConfig(BaseModel):
    # Core resource limits
    max_parallel: int = Field(default=1, gt=0)

    # Execution settings
    monitoring_interval: int = Field(default=60, gt=0)
    resume: bool = False  # Whether we are resuming from an existing experiment
    # TODO: In future, this check should be automatic.

    # Optional settings
    experiment_tags: list[str] = Field(default_factory=list)
