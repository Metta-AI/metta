"""Data models for sweep orchestration."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum, auto
from typing import Any


@dataclass
class Observation:
    score: float
    cost: float
    suggestion: dict


class JobTypes(StrEnum):
    LAUNCH_TRAINING = auto()
    LAUNCH_EVAL = auto()


@dataclass
class JobDefinition:
    run_id: str
    cmd: str  # e.g., "experiments.recipes.arena.train_shaped" or "experiments.recipes.arena.evaluate"
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
    PENDING = auto()  # Initialized but not started
    IN_TRAINING = auto()
    TRAINING_DONE_NO_EVAL = auto()
    IN_EVAL = auto()
    COMPLETED = auto()
    FAILED = auto()  # Job failed during training or evaluation


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

    # Evaluation results
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
        return JobStatus.COMPLETED

    # Dispatch info
    # dispatch_id: str | None = None
    # dispatch_type: DispatchType | None = None
