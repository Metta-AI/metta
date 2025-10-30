"""Job state models for tracking job execution status."""

import json
from typing import Any, Literal, Optional

from sqlalchemy import Text
from sqlmodel import Column, Field, SQLModel

from metta.jobs.job_config import JobConfig

JobStatus = Literal["pending", "running", "completed"]


class JobState(SQLModel, table=True):
    """Tracks job execution state and results.

    Persisted to SQLite with name as primary key.
    Config stored as JSON, exposed via property for type safety.
    """

    name: str = Field(primary_key=True)
    config_json: str = Field(default="", sa_column=Column("config", Text), exclude=True)

    def __init__(self, **data: Any):
        if "config" in data and isinstance(data["config"], JobConfig):
            data["config_json"] = json.dumps(data["config"].model_dump())
            del data["config"]
        super().__init__(**data)

    @property
    def config(self) -> JobConfig:
        if not self.config_json:
            raise ValueError("Config not set")
        return JobConfig(**json.loads(self.config_json))

    @config.setter
    def config(self, value: JobConfig) -> None:
        self.config_json = json.dumps(value.model_dump())

    # Runtime state
    status: str = Field(default="pending")
    request_id: Optional[str] = None  # SkyPilot request ID (remote jobs only)
    job_id: Optional[str] = None  # SkyPilot job ID or local PID
    skypilot_status: Optional[str] = None  # SkyPilot job status (PENDING, RUNNING, SUCCEEDED, etc.)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    # Results
    exit_code: Optional[int] = None
    logs_path: Optional[str] = None
    acceptance_passed: Optional[bool] = None  # None = not evaluated, True/False = result

    # Extracted artifacts
    wandb_url: Optional[str] = None
    wandb_run_id: Optional[str] = None
    checkpoint_uri: Optional[str] = None

    # Metrics
    metrics_json: str = Field(default="{}", sa_column=Column("metrics", Text), exclude=True)

    @property
    def metrics(self) -> dict[str, float]:
        return json.loads(self.metrics_json)

    @metrics.setter
    def metrics(self, value: dict[str, float]) -> None:
        self.metrics_json = json.dumps(value)
