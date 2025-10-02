"""Domain models for stable release validation.

Core types for tracking release state, validation outcomes, and lifecycle.
"""

from datetime import datetime
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field


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


class CheckResult(BaseModel):
    """Result of a single check/validation."""

    name: str
    lifecycle: Lifecycle
    outcome: Optional[Outcome] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    metrics: dict[str, float] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)  # name -> URL/path
    logs_path: Optional[str] = None
    notes: Optional[str] = None
    error: Optional[str] = None

    def mark_started(self) -> "CheckResult":
        """Mark this check as started."""
        self.lifecycle = Lifecycle.RUNNING
        self.started_at = datetime.utcnow()
        return self

    def mark_completed(self, outcome: Outcome, notes: Optional[str] = None) -> "CheckResult":
        """Mark this check as completed with outcome."""
        self.lifecycle = Lifecycle.COMPLETED
        self.outcome = outcome
        self.ended_at = datetime.utcnow()
        if notes:
            self.notes = notes
        return self

    def mark_failed(self, error: str) -> "CheckResult":
        """Mark this check as failed."""
        self.lifecycle = Lifecycle.COMPLETED
        self.outcome = Outcome.FAILED
        self.ended_at = datetime.utcnow()
        self.error = error
        return self

    @property
    def duration_seconds(self) -> Optional[float]:
        """Compute duration if both timestamps exist."""
        if self.started_at and self.ended_at:
            return (self.ended_at - self.started_at).total_seconds()
        return None


class ValidationConfig(BaseModel):
    """Configuration for a single validation."""

    name: str
    module: str
    location: str  # "local" or "remote"
    args: list[str] = Field(default_factory=list)
    acceptance: dict[str, float] = Field(default_factory=dict)  # e.g., {"sps_min": 40000}
    timeout_seconds: int = 300


class ReleaseState(BaseModel):
    """State of a release validation run."""

    version: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    repo_root: str
    commit_sha: Optional[str] = None

    # Step results
    prepare_branch: Optional[CheckResult] = None
    bug_check: Optional[CheckResult] = None
    validations: dict[str, CheckResult] = Field(default_factory=dict)  # validation_name -> result

    def add_validation_result(self, result: CheckResult) -> None:
        """Add or update a validation result."""
        self.validations[result.name] = result

    def get_validation_result(self, name: str) -> Optional[CheckResult]:
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
