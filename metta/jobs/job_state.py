"""Job state models for tracking job execution status."""

import json
import logging
from datetime import datetime
from enum import StrEnum
from typing import Any, Optional

from sqlalchemy import Text
from sqlmodel import Column, Field, SQLModel

from metta.jobs.job_config import JobConfig
from metta.jobs.job_metrics import fetch_job_metrics, parse_total_timesteps
from metta.jobs.job_runner import Job

logger = logging.getLogger(__name__)


class JobStatus(StrEnum):
    """Job execution status values."""

    PENDING = "pending"  # Queued, waiting to start
    RUNNING = "running"  # Currently executing
    COMPLETED = "completed"  # Finished (success or failure determined by exit_code)


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
    status: str = Field(default=JobStatus.PENDING)
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

    def update_from_spawned_job(self, job: "Job") -> None:
        """Update job state with metadata from a newly spawned job.

        Extracts job_id, logs_path, and WandB info (for training jobs) from the job.
        This method mutates the JobState instance.
        Caller is responsible for persisting changes (session.add + commit).

        Args:
            job: Job instance (LocalJob or RemoteJob) that was just spawned
        """
        # Set job_id if available
        if job.job_id:
            self.job_id = job.job_id

        # Set logs path so monitor can tail logs during execution
        self.logs_path = job.log_path

        # Capture job metadata (remote jobs return values, local jobs return None)
        if job.request_id:
            self.request_id = job.request_id

        # Set WandB info for training jobs (remote training jobs generate run names)
        if job.run_name and self.config.is_training_job:
            self.wandb_run_id = job.run_name
            self.wandb_url = (
                f"https://wandb.ai/{self.config.wandb_entity}/{self.config.wandb_project}/runs/{job.run_name}"
            )

    def fetch_and_update_metrics(self) -> None:
        """Fetch metrics from WandB and update self.metrics.

        This method mutates the JobState instance by updating metrics.
        Caller is responsible for persisting changes (session.add + commit).
        """
        if not self.config.metrics_to_track:
            return

        if not self.wandb_run_id:
            logger.warning(f"Cannot fetch metrics for {self.name}: no wandb_run_id set")
            return

        total_timesteps = parse_total_timesteps(self.config.args)

        try:
            metrics = fetch_job_metrics(
                entity=self.config.wandb_entity,
                project=self.config.wandb_project,
                run_name=self.wandb_run_id,
                metric_keys=self.config.metrics_to_track,
                total_timesteps=total_timesteps,
            )

            if metrics:
                self.metrics = metrics
                logger.info(f"Fetched metrics for {self.name}: {metrics}")
            else:
                logger.warning(f"No metrics returned for {self.name} (run may not have data yet)")
        except Exception as e:
            logger.warning(f"Failed to fetch metrics for {self.name}: {e}")

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
