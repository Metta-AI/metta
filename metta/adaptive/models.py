"""Data models for adaptive experiment orchestration."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum, auto
from typing import Any


class JobTypes(StrEnum):
    LAUNCH_TRAINING = auto()
    LAUNCH_EVAL = auto()


@dataclass
class JobDefinition:
    run_id: str
    cmd: str  # e.g., "recipes.experiment.arena.train_shaped" or "recipes.experiment.arena.evaluate"
    gpus: int = 1
    nodes: int = 1
    # Single source for recipe arguments (serialized as --args key=value)
    args: dict[str, Any] = field(default_factory=dict)
    # Single source for config overrides (serialized as --overrides key=value)
    overrides: dict[str, Any] = field(default_factory=dict)
    type: JobTypes = JobTypes.LAUNCH_TRAINING  # JobTypes enum value
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


class JobStatus(StrEnum):
    """Run lifecycle phases - computed by RunPhaseManager, not stored on RunInfo."""

    PENDING = "PENDING"  # Initialized but not started
    IN_TRAINING = "IN TRAINING"
    TRAINING_DONE_NO_EVAL = "TRAINING DONE (NO EVAL)"
    IN_EVAL = "IN EVAL"
    COMPLETED = "COMPLETED"
    STALE = "STALE"
    FAILED = "FAILED"  # Job failed during training or evaluation


@dataclass
class RunInfo:
    """Standardized run information returned by Store.

    Note: The lifecycle phase (JobStatus) is computed by RunPhaseManager,
    not stored directly on RunInfo. Use phase_manager.get_phase(run) to get
    the current phase.
    """

    run_id: str
    group: str | None = None
    tags: list | None = None

    # Timestamps
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    last_updated_at: datetime | None = None

    # Configuration and results
    summary: dict | None = None
    has_started_training: bool = False
    has_completed_training: bool = False
    has_failed: bool = False
    cost: float = 0
    runtime: float = 0

    # Training progress tracking
    total_timesteps: int | None = None  # Target timesteps from config
    current_steps: int | None = None  # Current agent_step from metrics
