"""Experiment state management with JSON persistence."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from metta.jobs.state import JobState

ExperimentStatus = Literal["pending", "running", "completed", "partial", "failed", "cancelled"]


class ExperimentState(BaseModel):
    """Complete state of an experiment instance.

    Manages the lifecycle and status of a multi-job experiment,
    with atomic JSON persistence to disk.
    """

    # Identity
    experiment_id: str  # e.g., "lr_comparison_20251013_1430"
    recipe: str  # e.g., "experiments.user.lr_comparison.my_experiment"

    # Timestamps
    created_at: str
    updated_at: str

    # Overall status
    status: ExperimentStatus

    # Job states
    jobs: dict[str, JobState] = Field(default_factory=dict)  # job_name -> JobState

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    def save(self, base_dir: Path = Path("experiments/state")) -> None:
        """Save state to JSON file atomically."""
        base_dir.mkdir(parents=True, exist_ok=True)
        path = base_dir / f"{self.experiment_id}.json"

        self.updated_at = datetime.utcnow().isoformat(timespec="seconds")

        # Atomic write: write to temp file then replace
        tmp_path = path.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)

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

        return cls.model_validate(data)

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
