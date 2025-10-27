"""Job manager with worker pool, queue, and persistence."""

import time
from datetime import datetime
from pathlib import Path

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
    - State persistence (SQLite)

    NOT responsible for:
    - Dependencies (caller's job)
    - Acceptance criteria (caller's job)
    - Result evaluation (caller's job)
    """

    def __init__(
        self,
        base_dir: Path,
        max_local_jobs: int = 1,
        max_remote_jobs: int = 10,
    ):
        """Create job manager with worker limits.

        Args:
            base_dir: Base directory for state and logs (creates jobs.sqlite and logs/ subdirectory)
            max_local_jobs: Max concurrent local jobs (default: 1)
            max_remote_jobs: Max concurrent remote jobs (default: 10)
        """
        self.base_dir = Path(base_dir)
        self.db_path = self.base_dir / "jobs.sqlite"
        self.log_dir = self.base_dir / "logs"
        self.max_local_jobs = max_local_jobs
        self.max_remote_jobs = max_remote_jobs

        # Worker tracking (across ALL groups)
        # name -> Job instance
        self._active_jobs: dict[str, LocalJob | RemoteJob] = {}

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

    def submit(self, config: JobConfig) -> None:
        """Submit a job to the queue.

        Creates JobState in DB:
        - If worker slot available: spawns Job instance immediately, marks 'running'
        - If no slots available: stays in 'pending', will auto-start when slot frees

        JobManager handles all rate limiting - caller just submits!

        Raises:
            ValueError: If job with same name already exists
        """
        with Session(self._engine) as session:
            # Check if job already exists - ERROR if it does
            existing = session.get(JobState, config.name)
            if existing:
                raise ValueError(
                    f"Job '{config.name}' already exists with status '{existing.status}'. "
                    f"Use get_job_state() to check status before submitting."
                )

            # Create new job state
            job_state = JobState(
                name=config.name,
                config=config,
                status="pending",
            )
            session.add(job_state)
            session.commit()

        # Try to start immediately if slot available
        self._try_start_job(config.name)

    def _try_start_job(self, name: str) -> bool:
        """Try to start a pending job if worker slot available, returning if it was started."""
        with Session(self._engine) as session:
            # Get job state
            job_state = session.get(JobState, name)
            if not job_state or job_state.status != "pending":
                return False

            # Check if slot available based on remote config
            is_remote = job_state.config.remote is not None
            if not self._has_available_slot(is_remote):
                return False

            # Start job
            job = self._spawn_job(job_state)
            self._active_jobs[name] = job

            # Update state to running
            job_state.status = "running"
            job_state.started_at = datetime.utcnow().isoformat(timespec="seconds")
            session.add(job_state)
            session.commit()

            return True

    def _has_available_slot(self, is_remote: bool) -> bool:
        """Check if worker slot available for execution type."""
        active_count = sum(
            1
            for job in self._active_jobs.values()
            if (not is_remote and isinstance(job, LocalJob)) or (is_remote and isinstance(job, RemoteJob))
        )

        if not is_remote:
            return active_count < self.max_local_jobs
        else:
            return active_count < self.max_remote_jobs

    def _spawn_job(self, job_state: JobState) -> LocalJob | RemoteJob:
        """Spawn Job instance from JobState."""
        config = job_state.config

        # Use manager's log_dir for all jobs
        log_dir = str(self.log_dir)

        # Check config.remote to determine execution location
        if config.remote is None:
            # Local execution
            return LocalJob(config, log_dir)
        else:
            # Remote execution
            job_id = int(job_state.job_id) if job_state.job_id else None
            return RemoteJob(config, log_dir, job_id=job_id)

    # ---- Polling ----

    def poll(self) -> list[str]:
        """Update ALL jobs and process queue.

        This is where JobManager does its work:

        1. Check each running job:
           - If completed: update state, free worker slot
           - If still running: continue

        2. Check pending jobs:
           - Count available local/remote slots
           - Start pending jobs up to limits (FIFO order)

        Returns:
            List of job names for newly completed jobs
        """
        completed = []

        # 1. Check running jobs for completion
        for name, job in list(self._active_jobs.items()):
            if job.is_complete():
                # Get job result
                job_result = job.wait(stream_output=False)

                # Update state
                with Session(self._engine) as session:
                    job_state = session.get(JobState, name)
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
                del self._active_jobs[name]
                completed.append(name)

        # 2. Try to start pending jobs
        with Session(self._engine) as session:
            # Get all pending jobs (FIFO order by creation)
            pending_jobs = session.exec(select(JobState).where(JobState.status == "pending")).all()

            for job_state in pending_jobs:
                self._try_start_job(job_state.name)

        return completed

    def wait_for_job(self, name: str, poll_interval_s: float = 1.0) -> JobState:
        """Poll until job completes and return JobState.

        Args:
            name: Job name
            poll_interval_s: Seconds between poll attempts (default: 1.0)

        Returns:
            Completed JobState

        Raises:
            RuntimeError: If job not found after completion
        """
        while True:
            completed = self.poll()
            if name in completed:
                break
            time.sleep(poll_interval_s)

        job_state = self.get_job_state(name)
        if not job_state:
            raise RuntimeError(f"Job {name} not found after completion")
        return job_state

    # ---- Query ----

    def get_status(self, name: str) -> JobStatus | None:
        """Get job status, or None if job doesn't exist."""
        with Session(self._engine) as session:
            job_state = session.get(JobState, name)
            return job_state.status if job_state else None

    def get_job_state(self, name: str) -> JobState | None:
        """Get full job state (or None if doesn't exist)."""
        with Session(self._engine) as session:
            job_state = session.get(JobState, name)
            if job_state:
                # Detach from session so it can be used outside
                session.expunge(job_state)
            return job_state

    def get_group_jobs(self, group: str) -> dict[str, JobState]:
        """Get all jobs in group. Returns a dictionary mapping job name to JobState for all jobs in group."""
        with Session(self._engine) as session:
            # Get all jobs and filter by group in config
            all_jobs = session.exec(select(JobState)).all()
            group_jobs = [job for job in all_jobs if job.config.group == group]

            # Detach from session
            for job in group_jobs:
                session.expunge(job)
            return {job.name: job for job in group_jobs}

    # ---- Group Operations ----

    def cancel_group(self, group: str) -> int:
        """Cancel all jobs in group."""
        count = 0
        with Session(self._engine) as session:
            # Get all jobs and filter by group
            all_jobs = session.exec(select(JobState)).all()
            group_jobs = [job for job in all_jobs if job.config.group == group]

            for job_state in group_jobs:
                if job_state.status in ("pending", "running"):
                    # Cancel running job
                    if job_state.name in self._active_jobs:
                        job = self._active_jobs[job_state.name]
                        job.cancel()
                        del self._active_jobs[job_state.name]

                    # Update state
                    job_state.status = "cancelled"
                    session.add(job_state)
                    count += 1

            session.commit()

        return count
