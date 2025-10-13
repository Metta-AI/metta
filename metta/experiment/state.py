"""Experiment state management with JSON persistence."""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from metta.jobs.models import JobSpec

ExperimentStatus = Literal["pending", "running", "completed", "partial", "failed", "cancelled"]
JobStatus = Literal["pending", "running", "completed", "failed", "cancelled"]


@dataclass
class JobState:
    """State of a single job within an experiment."""

    name: str
    spec: JobSpec  # Full job specification

    # Runtime state
    status: JobStatus = "pending"
    job_id: Optional[str] = None  # Skypilot job ID for remote, PID for local
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    # Results
    exit_code: Optional[int] = None
    logs_path: Optional[str] = None

    # Extracted artifacts
    wandb_url: Optional[str] = None
    wandb_run_id: Optional[str] = None
    checkpoint_uri: Optional[str] = None

    # Metrics (for acceptance criteria)
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "name": self.name,
            "spec": self.spec.to_dict(),
            "status": self.status,
            "job_id": self.job_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "exit_code": self.exit_code,
            "logs_path": self.logs_path,
            "wandb_url": self.wandb_url,
            "wandb_run_id": self.wandb_run_id,
            "checkpoint_uri": self.checkpoint_uri,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobState":
        """Deserialize from JSON-compatible dict."""
        # Reconstruct JobSpec
        data["spec"] = JobSpec.from_dict(data["spec"])
        return cls(**data)


@dataclass
class ExperimentState:
    """Complete state of an experiment instance."""

    # Identity
    experiment_id: str  # e.g., "lr_comparison_20251013_1430"
    recipe: str  # e.g., "experiments.user.lr_comparison.my_experiment"

    # Timestamps
    created_at: str
    updated_at: str

    # Overall status
    status: ExperimentStatus

    # Job states
    jobs: dict[str, JobState] = field(default_factory=dict)  # job_name -> JobState

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "experiment_id": self.experiment_id,
            "recipe": self.recipe,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "jobs": {name: job.to_dict() for name, job in self.jobs.items()},
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentState":
        """Deserialize from JSON-compatible dict."""
        data["jobs"] = {name: JobState.from_dict(job_data) for name, job_data in data["jobs"].items()}
        return cls(**data)

    def save(self, base_dir: Path = Path("experiments/state")) -> None:
        """Save state to JSON file atomically."""
        base_dir.mkdir(parents=True, exist_ok=True)
        path = base_dir / f"{self.experiment_id}.json"

        self.updated_at = datetime.utcnow().isoformat(timespec="seconds")

        # Atomic write: write to temp file then replace
        tmp_path = path.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        # Atomic replace (POSIX)
        os.replace(tmp_path, path)

    @classmethod
    def load(cls, instance_id: str, base_dir: Path = Path("experiments/state")) -> Optional["ExperimentState"]:
        """Load state from JSON file."""
        path = base_dir / f"{instance_id}.json"
        if not path.exists():
            return None

        with open(path) as f:
            data = json.load(f)

        return cls.from_dict(data)

    def update_job_status(self, job_name: str, **updates: Any) -> None:
        """Update a job's state and save.

        Args:
            job_name: Name of job to update
            **updates: Field updates (e.g., status="running", job_id="12345")
        """
        if job_name not in self.jobs:
            raise ValueError(f"Job {job_name} not found in experiment")

        job_state = self.jobs[job_name]
        for key, value in updates.items():
            setattr(job_state, key, value)

        # Update overall experiment status
        self._update_overall_status()

        self.save()

    def _update_overall_status(self) -> None:
        """Compute overall experiment status from job states."""
        if not self.jobs:
            self.status = "pending"
            return

        statuses = [job.status for job in self.jobs.values()]

        if all(s == "completed" for s in statuses):
            self.status = "completed"
        elif any(s == "failed" for s in statuses):
            self.status = "failed"
        elif any(s == "cancelled" for s in statuses):
            self.status = "cancelled"
        elif any(s == "running" for s in statuses):
            self.status = "running"
        elif all(s in ("completed", "pending") for s in statuses):
            self.status = "partial"
        else:
            self.status = "pending"
