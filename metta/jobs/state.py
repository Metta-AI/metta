"""Job state models for tracking job execution status."""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from metta.jobs.models import JobConfig

JobStatus = Literal["pending", "running", "completed", "failed", "cancelled"]


class JobState(BaseModel):
    """State of a single job within an experiment.

    This tracks the runtime status, results, and extracted artifacts
    from a job execution.
    """

    name: str
    config: JobConfig  # Full job configuration

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
    metrics: dict[str, float] = Field(default_factory=dict)
