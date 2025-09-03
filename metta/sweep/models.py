"""Data models for sweep orchestration."""

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
    cmd: str  # e.g., "experiments.recipes.arena.train_shaped" or "experiments.recipes.arena.evaluate"
    gpus: int = 1
    nodes: int = 1
    args: list[str] = field(default_factory=list)  # positional arguments
    overrides: dict[str, Any] = field(default_factory=dict)  # key=value overrides for the tool
    config: dict[str, Any] = field(default_factory=dict)  # additional config from optimizer
    type: JobTypes = JobTypes.LAUNCH_TRAINING  # JobTypes enum value
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


class JobStatus(StrEnum):
    PENDING = auto()  # Initialized but not started
    IN_TRAINING = auto()
    TRAINING_DONE_NO_EVAL = auto()
    IN_EVAL = auto()
    EVAL_DONE_NOT_COMPLETED = auto()
    COMPLETED = auto()
    FAILED = auto()  # Job failed during training or evaluation


class SweepStatus(StrEnum):
    CREATED = auto()
    RESUMED = auto()


@dataclass
class Observation:
    score: float
    cost: float
    suggestion: dict


@dataclass
class RunInfo:
    """Standardized run information returned by Store"""

    run_id: str
    group: str | None = None
    tags: list | None = None

    # Timestamps
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    last_heartbeat_at: datetime | None = None

    # Configuration and results
    summary: dict | None = None
    has_started_training: bool = False
    has_completed_training: bool = False
    has_started_eval: bool = False
    has_been_evaluated: bool = False
    has_failed: bool = False
    cost: float = 0
    runtime: float = 0

    # Training progress tracking
    total_timesteps: int | None = None  # Target timesteps from config
    current_steps: int | None = None  # Current agent_step from metrics

    # Sweep specific
    observation: Observation | None = None

    @property
    def status(self) -> JobStatus:
        if self.has_failed:
            return JobStatus.FAILED
        if not self.has_started_training:
            return JobStatus.PENDING
        if self.has_started_training and not self.has_completed_training:
            return JobStatus.IN_TRAINING
        if self.has_completed_training and not self.has_started_eval:
            return JobStatus.TRAINING_DONE_NO_EVAL
        if self.has_started_eval and not self.has_been_evaluated:
            return JobStatus.IN_EVAL
        if self.has_been_evaluated:
            if self.observation is not None:
                return JobStatus.COMPLETED
            return JobStatus.EVAL_DONE_NOT_COMPLETED
        return JobStatus.COMPLETED

    # Dispatch info
    # dispatch_id: str | None = None
    # dispatch_type: DispatchType | None = None


@dataclass
class SweepMetadata:
    """
    Metadata about a sweep stored in the Store.
    This is the persistent state that survives controller restarts.
    """

    sweep_id: str
    start_time: datetime = field(default_factory=datetime.now)
    last_scheduling: datetime = field(default_factory=datetime.now)
    runs_created: int = 0
    runs_pending: int = 0
    runs_in_progress: int = 0
    runs_completed: int = 0

    def to_metrics_dict(self) -> dict[str, Any]:
        """Convert to metrics dictionary for logging"""
        return {
            "runs_created": self.runs_created,
            "runs_completed": self.runs_completed,
            "runs_in_progress": self.runs_in_progress,
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
        }
