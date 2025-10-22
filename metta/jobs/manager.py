"""Job manager with worker pool, queue, and persistence."""

import time
from datetime import datetime
from pathlib import Path
from typing import Literal

from sqlmodel import Session, SQLModel, create_engine, select

from metta.jobs.metrics import (
    extract_checkpoint_path,
    extract_final_metrics,
    extract_skypilot_job_id,
    extract_wandb_info,
)
from metta.jobs.models import JobConfig
from metta.jobs.runner import LocalJob, RemoteJob
from metta.jobs.state import JobState, JobStatus


class JobManager:
    """Manages job execution with concurrency control and persistence.

    Responsibilities:
    - Worker pool (max_local_jobs, max_remote_jobs)
    - Queue management (pending -> running based on worker slots)
    - State persistence (SQLite with multi-batch support)

    NOT responsible for:
    - Dependencies (caller's job)
    - Acceptance criteria (caller's job)
    - Result evaluation (caller's job)
    """

    def __init__(
        self,
        db_path: Path,
        max_local_jobs: int = 1,
        max_remote_jobs: int = 10,
    ):
        """Create job manager with worker limits.

        Args:
            db_path: SQLite database for state persistence
            max_local_jobs: Max concurrent local jobs (default: 1)
            max_remote_jobs: Max concurrent remote jobs (default: 10)
        """
        self.db_path = Path(db_path)
        self.max_local_jobs = max_local_jobs
        self.max_remote_jobs = max_remote_jobs

        # Worker tracking (across ALL batches)
        # (batch_id, name) -> Job instance
        self._active_jobs: dict[tuple[str, str], LocalJob | RemoteJob] = {}

        # Initialize database
        self._init_db()

    def _init_db(self) -> None:
        """Create database and tables if they don't exist."""
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create engine and tables
        self._engine = create_engine(f"sqlite:///{self.db_path}")
        SQLModel.metadata.create_all(self._engine)

    # ---- Submit ----

    def submit(self, batch_id: str, config: JobConfig) -> None:
        """Submit a job to the queue.

        Creates JobState in DB:
        - If worker slot available: spawns Job instance immediately, marks 'running'
        - If no slots available: stays in 'pending', will auto-start when slot frees

        JobManager handles all rate limiting - caller just submits!

        Args:
            batch_id: Batch identifier (e.g., version)
            config: Job configuration
        """
        with Session(self._engine) as session:
            # Create job state
            job_state = JobState(
                batch_id=batch_id,
                name=config.name,
                config=config,
                status="pending",
            )
            session.add(job_state)
            session.commit()

        # Try to start immediately if slot available
        self._try_start_job(batch_id, config.name)

    def _try_start_job(self, batch_id: str, name: str) -> bool:
        """Try to start a pending job if worker slot available.

        Returns:
            True if job was started, False if no slots available
        """
        with Session(self._engine) as session:
            # Get job state
            job_state = session.get(JobState, (batch_id, name))
            if not job_state or job_state.status != "pending":
                return False

            # Check if slot available
            if not self._has_available_slot(job_state.config.execution):
                return False

            # Start job
            job = self._spawn_job(job_state)
            self._active_jobs[(batch_id, name)] = job

            # Update state to running
            job_state.status = "running"
            job_state.started_at = datetime.utcnow().isoformat(timespec="seconds")
            session.add(job_state)
            session.commit()

            return True

    def _has_available_slot(self, execution: Literal["local", "remote"]) -> bool:
        """Check if worker slot available for execution type."""
        active_count = sum(
            1
            for (batch_id, name), job in self._active_jobs.items()
            if (execution == "local" and isinstance(job, LocalJob))
            or (execution == "remote" and isinstance(job, RemoteJob))
        )

        if execution == "local":
            return active_count < self.max_local_jobs
        else:
            return active_count < self.max_remote_jobs

    def _spawn_job(self, job_state: JobState) -> LocalJob | RemoteJob:
        """Spawn Job instance from JobState."""
        config = job_state.config

        # Determine log directory (keep stable releases together)
        from metta.common.util.fs import get_repo_root

        if job_state.batch_id.startswith("release_v"):
            log_dir = str(get_repo_root() / "devops" / "stable" / "logs" / job_state.batch_id)
        else:
            log_dir = f"./logs/{job_state.batch_id}"

        if config.execution == "local":
            args = config.to_local_job_args(log_dir)
            return LocalJob(**args)
        else:
            args = config.to_remote_job_args(log_dir)
            # Check if we're resuming an existing job
            if job_state.job_id:
                args["job_id"] = int(job_state.job_id)
            return RemoteJob(**args)

    # ---- Polling ----

    def poll(self) -> list[tuple[str, str]]:
        """Update ALL jobs across all batches and process queue.

        This is where JobManager does its work:

        1. Check each running job:
           - If completed: update state, free worker slot
           - If still running: continue

        2. Check pending jobs:
           - Count available local/remote slots
           - Start pending jobs up to limits (FIFO order)

        Returns:
            List of (batch_id, name) for newly completed jobs
        """
        completed = []

        # 1. Check running jobs for completion
        for (batch_id, name), job in list(self._active_jobs.items()):
            if job.is_complete():
                # Get job result
                job_result = job.wait(stream_output=False)

                # Update state
                with Session(self._engine) as session:
                    job_state = session.get(JobState, (batch_id, name))
                    if job_state:
                        job_state.status = "completed"
                        job_state.completed_at = datetime.utcnow().isoformat(timespec="seconds")
                        job_state.exit_code = job_result.exit_code
                        job_state.logs_path = job_result.logs_path
                        job_state.job_id = str(job_result.job_id) if job_result.job_id else None

                        # Extract artifacts from logs
                        if job_state.logs_path:
                            try:
                                logs = Path(job_state.logs_path).read_text(errors="ignore")
                            except Exception:
                                logs = ""

                            if logs:
                                # Extract WandB info
                                if not job_state.wandb_url:
                                    wandb_info = extract_wandb_info(logs)
                                    if wandb_info:
                                        job_state.wandb_run_id = wandb_info.run_id
                                        job_state.wandb_url = wandb_info.url

                                # Extract checkpoint URI
                                if not job_state.checkpoint_uri:
                                    checkpoint = extract_checkpoint_path(logs)
                                    if checkpoint:
                                        job_state.checkpoint_uri = checkpoint
                                    elif job_state.wandb_run_id:
                                        # Default checkpoint URI from wandb
                                        job_state.checkpoint_uri = f"wandb://run/{job_state.wandb_run_id}"

                                # Extract SkyPilot job ID if not already set
                                if not job_state.job_id:
                                    skypilot_id = extract_skypilot_job_id(logs)
                                    if skypilot_id:
                                        job_state.job_id = skypilot_id

                                # Extract and merge metrics
                                parsed_metrics = extract_final_metrics(logs)
                                if parsed_metrics:
                                    # Merge with existing metrics (don't clobber)
                                    current_metrics = job_state.metrics
                                    job_state.metrics = {**current_metrics, **parsed_metrics}

                        session.add(job_state)
                        session.commit()

                # Free worker slot
                del self._active_jobs[(batch_id, name)]
                completed.append((batch_id, name))

        # 2. Try to start pending jobs
        with Session(self._engine) as session:
            # Get all pending jobs (FIFO order by creation)
            pending_jobs = session.exec(select(JobState).where(JobState.status == "pending")).all()

            for job_state in pending_jobs:
                self._try_start_job(job_state.batch_id, job_state.name)

        return completed

    def wait_for_job(self, batch_id: str, name: str, poll_interval_s: float = 1.0) -> JobState:
        """Poll until job completes and return JobState.

        Args:
            batch_id: Batch identifier
            name: Job name
            poll_interval_s: Seconds between poll attempts (default: 1.0)

        Returns:
            Completed JobState

        Raises:
            RuntimeError: If job not found after completion
        """
        while True:
            completed = self.poll()
            if (batch_id, name) in completed:
                break
            time.sleep(poll_interval_s)

        job_state = self.get_job_state(batch_id, name)
        if not job_state:
            raise RuntimeError(f"Job {name} not found after completion")
        return job_state

    # ---- Query ----

    def get_status(self, batch_id: str, name: str) -> JobStatus | None:
        """Get job status.

        Args:
            batch_id: Batch identifier
            name: Job name (unique within batch)

        Returns:
            'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
            None if job doesn't exist
        """
        with Session(self._engine) as session:
            job_state = session.get(JobState, (batch_id, name))
            return job_state.status if job_state else None

    def get_job_state(self, batch_id: str, name: str) -> JobState | None:
        """Get full job state (or None if doesn't exist)."""
        with Session(self._engine) as session:
            job_state = session.get(JobState, (batch_id, name))
            if job_state:
                # Detach from session so it can be used outside
                session.expunge(job_state)
            return job_state

    def get_batch_jobs(self, batch_id: str) -> dict[str, JobState]:
        """Get all jobs in batch (name -> JobState)."""
        with Session(self._engine) as session:
            jobs = session.exec(select(JobState).where(JobState.batch_id == batch_id)).all()
            # Detach from session
            for job in jobs:
                session.expunge(job)
            return {job.name: job for job in jobs}

    # ---- Batch Operations ----

    def cancel_batch(self, batch_id: str) -> int:
        """Cancel all jobs in batch.

        Returns:
            Number of jobs cancelled
        """
        count = 0
        with Session(self._engine) as session:
            jobs = session.exec(select(JobState).where(JobState.batch_id == batch_id)).all()

            for job_state in jobs:
                if job_state.status in ("pending", "running"):
                    # Cancel running job
                    if (batch_id, job_state.name) in self._active_jobs:
                        job = self._active_jobs[(batch_id, job_state.name)]
                        job.cancel()
                        del self._active_jobs[(batch_id, job_state.name)]

                    # Update state
                    job_state.status = "cancelled"
                    session.add(job_state)
                    count += 1

            session.commit()

        return count
