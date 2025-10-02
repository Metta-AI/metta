"""Data models for stable release validation.

Types for tracking release state, validation results, and outcomes.
"""

from datetime import datetime
from enum import StrEnum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Location(StrEnum):
    """Job execution location."""

    LOCAL = "local"
    REMOTE = "remote"


class Lifecycle(StrEnum):
    """Validation lifecycle state."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"


class Outcome(StrEnum):
    """Validation outcome (separate from lifecycle)."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    INCONCLUSIVE = "inconclusive"


class Artifact(BaseModel):
    """Artifact produced by a validation run."""

    name: str
    uri: str  # file path or URL
    type: Literal["log", "html", "json", "image", "table", "other"] = "log"
    mime: Optional[str] = None


class ThresholdCheck(BaseModel):
    """A single threshold check with operator and expected value."""

    key: str
    op: Literal[">=", ">", "<=", "<", "==", "!="] = ">="
    expected: float
    actual: Optional[float] = None
    passed: Optional[bool] = None
    note: Optional[str] = None


class GateResult(BaseModel):
    """Result of running a gate (bug check, workflow validation, etc)."""

    name: str  # e.g., "workflow", "bug"
    outcome: Outcome
    started_at: datetime | None = None
    ended_at: datetime | None = None
    notes: Optional[str] = None
    failed_checks: list[ThresholdCheck] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)


class RunResult(BaseModel):
    """Result of a single validation run."""

    name: str
    location: Location
    lifecycle: Lifecycle
    outcome: Optional[Outcome] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    exit_code: int = 0
    metrics: dict[str, float] = Field(default_factory=dict)
    artifacts: list[Artifact] = Field(default_factory=list)
    logs_path: Optional[str] = None
    external_id: Optional[str] = None  # e.g., SkyPilot job_id
    notes: Optional[str] = None
    error: Optional[str] = None

    def mark_started(self) -> "RunResult":
        """Mark this run as started."""
        self.lifecycle = Lifecycle.RUNNING
        self.started_at = datetime.utcnow()
        return self

    def mark_completed(self, outcome: Outcome, exit_code: int = 0, notes: Optional[str] = None) -> "RunResult":
        """Mark this run as completed with outcome."""
        self.lifecycle = Lifecycle.COMPLETED
        self.outcome = outcome
        self.exit_code = exit_code
        self.ended_at = datetime.utcnow()
        if notes:
            self.notes = notes
        return self

    def mark_failed(self, error: str, exit_code: int = 1) -> "RunResult":
        """Mark this run as failed."""
        self.lifecycle = Lifecycle.COMPLETED
        self.outcome = Outcome.FAILED
        self.exit_code = exit_code
        self.ended_at = datetime.utcnow()
        self.error = error
        return self

    @property
    def duration_seconds(self) -> Optional[float]:
        """Compute duration if both timestamps exist."""
        if self.started_at and self.ended_at:
            return (self.ended_at - self.started_at).total_seconds()
        return None


class Validation(BaseModel):
    """Configuration for a single validation run."""

    name: str
    module: str
    location: Location
    args: list[str] = Field(default_factory=list)
    timeout_s: int = 900
    acceptance: list[ThresholdCheck] = Field(default_factory=list)  # inline criteria


class ReleaseState(BaseModel):
    """State of a release validation run."""

    version: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    repo_root: str
    commit_sha: Optional[str] = None

    gates: list[GateResult] = Field(default_factory=list)
    validations: dict[str, RunResult] = Field(default_factory=dict)  # validation_name -> result

    def add_validation_result(self, result: RunResult) -> None:
        """Add or update a validation result."""
        self.validations[result.name] = result

    def add_gate_result(self, gate: GateResult) -> None:
        """Add a gate result."""
        self.gates.append(gate)

    def get_validation_result(self, name: str) -> Optional[RunResult]:
        """Get result for a specific validation."""
        return self.validations.get(name)

    @property
    def all_validations_passed(self) -> bool:
        """Check if all validations passed."""
        if not self.validations:
            return False
        return all(v.outcome == Outcome.PASSED for v in self.validations.values() if v.lifecycle == Lifecycle.COMPLETED)

    @property
    def validation_summary(self) -> dict[str, int]:
        """Get summary counts of validation outcomes."""
        summary = {"passed": 0, "failed": 0, "running": 0, "pending": 0, "skipped": 0}
        for result in self.validations.values():
            if result.outcome:
                summary[result.outcome.value] += 1
            elif result.lifecycle == Lifecycle.RUNNING:
                summary["running"] += 1
            else:
                summary["pending"] += 1
        return summary
