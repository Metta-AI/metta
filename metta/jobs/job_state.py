"""Job state models for tracking job execution status."""

import json
import logging
from datetime import datetime
from typing import Any, Literal, Optional

from sqlalchemy import Text
from sqlmodel import Column, Field, SQLModel

from metta.jobs.job_config import JobConfig

logger = logging.getLogger(__name__)

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
    status: str = Field(default="pending")  # One of: "pending", "running", "completed" (see JobStatus)
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

    def evaluate_acceptance(self) -> bool:
        """Evaluate acceptance criteria against job metrics.

        Returns True if all criteria pass, False if any fail.
        Returns True if no criteria defined (vacuous truth).
        Returns False if criteria defined but metrics missing.
        """
        if not self.config.acceptance_criteria:
            return True  # No criteria = pass

        if not self.metrics:
            logger.warning(f"Cannot evaluate acceptance for {self.name}: criteria defined but no metrics available")
            return False

        failures = []
        for criterion in self.config.acceptance_criteria:
            metric_value = self.metrics.get(criterion.metric)
            if metric_value is None:
                failures.append(f"{criterion.metric}: metric missing")
                continue

            # Use criterion's evaluate method
            if not criterion.evaluate(metric_value):
                failures.append(
                    f"{criterion.metric}: expected {criterion.operator} {criterion.threshold}, got {metric_value:.4f}"
                )

        if failures:
            logger.info(f"Acceptance criteria failed for {self.name}: {'; '.join(failures)}")
            return False

        logger.info(f"Acceptance criteria passed for {self.name}")
        return True

    @property
    def duration_s(self) -> float | None:
        """Compute duration from started_at and completed_at timestamps."""
        if not self.started_at or not self.completed_at:
            return None

        start = datetime.fromisoformat(self.started_at)
        end = datetime.fromisoformat(self.completed_at)
        return (end - start).total_seconds()

    @property
    def is_successful(self) -> bool:
        """Job succeeded if exit_code=0 and acceptance criteria passed (or not evaluated)."""
        return self.exit_code == 0 and self.acceptance_passed is not False

    @property
    def is_failed(self) -> bool:
        """Job failed if exit_code!=0 or acceptance criteria failed."""
        return self.exit_code != 0 or self.acceptance_passed is False

    @property
    def progress_pct(self) -> float | None:
        """Compute progress percentage if _progress metadata exists in metrics."""
        if "_progress" not in self.metrics:
            return None

        progress = self.metrics["_progress"]
        if isinstance(progress, dict) and "current_step" in progress and "total_steps" in progress:
            total = progress["total_steps"]
            if total > 0:
                return (progress["current_step"] / total) * 100.0

        return None
