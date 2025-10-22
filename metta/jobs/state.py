"""Job state models for tracking job execution status."""

import json
from typing import Any, Literal, Optional

from sqlalchemy import Text
from sqlmodel import Column, Field, SQLModel

from metta.jobs.models import JobConfig

JobStatus = Literal["pending", "running", "completed", "failed", "cancelled"]


class JobState(SQLModel, table=True):
    """State of a single job.

    This tracks the runtime status, results, and extracted artifacts
    from a job execution. Persisted to SQLite via SQLModel.

    Composite primary key: (batch_id, name) allows multiple batches
    to share the same database.
    """

    # Composite primary key
    batch_id: str = Field(primary_key=True)
    name: str = Field(primary_key=True)

    # Job configuration (stored as JSON TEXT, exposed as 'config' property)
    config_json: str = Field(default="", sa_column=Column("config", Text), exclude=True)

    def __init__(self, **data: Any):
        """Initialize JobState, handling JobConfig serialization."""
        if "config" in data and isinstance(data["config"], JobConfig):
            data["config_json"] = json.dumps(data["config"].model_dump())
            del data["config"]
        super().__init__(**data)

    @property
    def config(self) -> JobConfig:
        """Get JobConfig from JSON."""
        if not self.config_json:
            raise ValueError("Config not set")
        return JobConfig(**json.loads(self.config_json))

    @config.setter
    def config(self, value: JobConfig) -> None:
        """Set JobConfig as JSON."""
        self.config_json = json.dumps(value.model_dump())

    # Runtime state
    status: str = Field(default="pending")
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

    # Metrics (stored as JSON TEXT)
    metrics_json: str = Field(default="{}", sa_column=Column("metrics", Text), exclude=True)

    @property
    def metrics(self) -> dict[str, float]:
        """Get metrics from JSON."""
        return json.loads(self.metrics_json)

    @metrics.setter
    def metrics(self, value: dict[str, float]) -> None:
        """Set metrics as JSON."""
        self.metrics_json = json.dumps(value)
