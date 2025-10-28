"""Job manager with worker pool, queue, and persistence."""

import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine, select

from devops.skypilot.utils.job_helpers import check_job_statuses
from metta.jobs.job_config import JobConfig
from metta.jobs.job_metrics import (
    extract_checkpoint_path,
    extract_final_metrics,
    extract_skypilot_job_id,
    extract_wandb_info,
)
from metta.jobs.job_runner import LocalJob, RemoteJob
from metta.jobs.job_state import JobState, JobStatus

logger = logging.getLogger(__name__)


class JobManager:
    """Manages job execution with concurrency control and persistence.

    Architecture:
    - Maintains worker pools (max_local_jobs, max_remote_jobs)
    - SQLite database stores job state (jobs.sqlite)
    - Job instances (LocalJob/RemoteJob) handle execution
    - Artifacts extracted from logs after completion

    Job Lifecycle:
    1. submit() -> Creates JobState in DB, starts if slot available
    2. poll() -> Checks running jobs, extracts artifacts, starts pending jobs
    3. Query methods -> get_status(), get_job_state(), get_group_jobs()

    Separation of concerns:
    - JobManager: Worker pools, queue, persistence
    - Job (LocalJob/RemoteJob): Execution, log streaming
    - Caller (TaskRunner): Dependencies, acceptance criteria, evaluation
    """

    def __init__(
        self,
        base_dir: Path,
        max_local_jobs: int = 1,
        max_remote_jobs: int = 10,
        remote_poll_interval_s: float = 5.0,
    ):
        self.base_dir = Path(base_dir)
        self.db_path = self.base_dir / "jobs.sqlite"
        self.log_dir = self.base_dir / "logs"
        self.max_local_jobs = max_local_jobs
        self.max_remote_jobs = max_remote_jobs
        self.remote_poll_interval_s = remote_poll_interval_s  # How often to poll SkyPilot for remote job status
        self._active_jobs: dict[str, LocalJob | RemoteJob] = {}
        self._monitor_threads: dict[str, threading.Thread] = {}  # Remote job monitoring threads
        self._init_db()
        self._validate_job_states()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(f"sqlite:///{self.db_path}")
        SQLModel.metadata.create_all(self._engine)

    def _validate_job_states(self) -> None:
        """Validate and update status of all running jobs on startup.

        Synchronously validates local jobs (fast PID checks).
        Asynchronously validates remote jobs (slow SkyPilot API calls).

        Local jobs: Check if PID exists, mark stale ones as completed.
        Remote jobs: Query SkyPilot for actual status in background thread.
        """
        with Session(self._engine) as session:
            running_jobs = session.exec(select(JobState).where(JobState.status == "running")).all()

            local_stale_count = 0
            remote_jobs = []

            for job_state in running_jobs:
                is_remote = job_state.config.remote is not None

                if not is_remote:
                    # Validate local jobs synchronously (fast)
                    if job_state.job_id:
                        try:
                            pid = int(job_state.job_id)
                            # Send signal 0 to check if process exists
                            os.kill(pid, 0)
                            # Process exists, leave it alone
                            continue
                        except (ValueError, OSError, ProcessLookupError):
                            # Process doesn't exist - mark as stale
                            pass

                    # Mark stale local job as completed with abnormal termination
                    job_state.status = "completed"
                    job_state.exit_code = -1  # Abnormal termination
                    job_state.completed_at = datetime.now().isoformat(timespec="seconds")
                    session.add(job_state)
                    local_stale_count += 1
                else:
                    # Collect remote jobs for async validation
                    remote_jobs.append((job_state.name, job_state.job_id))

            session.commit()

            if local_stale_count > 0:
                logger.info(f"Cleaned up {local_stale_count} stale local job(s)")

        # Start background validation for remote jobs
        if remote_jobs:
            logger.info(f"Validating {len(remote_jobs)} remote job(s) in background...")
            thread = threading.Thread(
                target=self._validate_remote_jobs_async,
                args=(remote_jobs,),
                daemon=True,
            )
            thread.start()

    def _start_remote_monitor(self, job_name: str, job_id: int) -> None:
        """Start background monitoring thread for a remote job.

        The thread polls SkyPilot API at configured interval to update job status in database.
        Exits when job reaches terminal state or thread is interrupted.

        Args:
            job_name: Name of job to monitor
            job_id: SkyPilot job ID
        """

        def monitor_loop():
            try:
                while True:
                    try:
                        # Query SkyPilot for current status
                        statuses = check_job_statuses([job_id])
                        status = statuses.get(job_id, {}).get("status")

                        if not status:
                            # Job info not available, exit thread
                            break

                        # Update database with current status
                        with Session(self._engine) as session:
                            job_state = session.get(JobState, job_name)
                            if job_state:
                                job_state.skypilot_status = status
                                session.add(job_state)
                                session.commit()

                        # Exit if terminal state reached
                        if status in (
                            "SUCCEEDED",
                            "FAILED",
                            "FAILED_SETUP",
                            "FAILED_DRIVER",
                            "CANCELLED",
                            "UNKNOWN",
                            "ERROR",
                        ):
                            break

                        # Poll at configured interval
                        time.sleep(self.remote_poll_interval_s)

                    except Exception as e:
                        logger.warning(f"Remote monitor thread for {job_name} failed: {e}")
                        time.sleep(self.remote_poll_interval_s)  # Continue trying

            finally:
                # Clean up thread reference when done
                if job_name in self._monitor_threads:
                    del self._monitor_threads[job_name]

        # Start daemon thread (will exit when main program exits)
        thread = threading.Thread(target=monitor_loop, daemon=True, name=f"monitor-{job_name}")
        thread.start()
        self._monitor_threads[job_name] = thread
        logger.debug(f"Started monitoring thread for remote job {job_name} (Job ID: {job_id})")

    def _validate_remote_jobs_async(self, remote_jobs: list[tuple[str, str | None]]) -> None:
        """Validate remote job statuses via SkyPilot API (runs in background thread).

        Queries SkyPilot for actual job status and updates DB accordingly.
        This is slow (API calls) so we run it asynchronously to not block startup.

        Args:
            remote_jobs: List of (job_name, job_id) tuples to validate
        """
        # Extract job IDs (skip jobs without IDs)
        job_ids = []
        job_id_to_name = {}
        for job_name, job_id in remote_jobs:
            if job_id:
                try:
                    job_id_int = int(job_id)
                    job_ids.append(job_id_int)
                    job_id_to_name[job_id_int] = job_name
                except ValueError:
                    pass

        if not job_ids:
            return

        try:
            # Query SkyPilot for job statuses (can be slow)
            statuses = check_job_statuses(job_ids)

            # Update DB with actual statuses
            with Session(self._engine) as session:
                for job_id, status in statuses.items():
                    job_name = job_id_to_name.get(job_id)
                    if not job_name:
                        continue

                    job_state = session.get(JobState, job_name)
                    if not job_state or job_state.status != "running":
                        continue

                    # Store SkyPilot status for monitoring
                    job_state.skypilot_status = status

                    # Update based on SkyPilot status
                    if status in ("SUCCEEDED", "FAILED", "CANCELLED"):
                        job_state.status = "completed"
                        if status == "SUCCEEDED":
                            job_state.exit_code = 0
                        elif status == "CANCELLED":
                            job_state.exit_code = 130  # User cancelled
                        else:
                            job_state.exit_code = 1  # Failed
                        job_state.completed_at = datetime.now().isoformat(timespec="seconds")
                        session.add(job_state)
                        logger.info(f"Remote job {job_name} updated: {status}")
                    else:
                        # RUNNING or PENDING - just update the status field
                        session.add(job_state)

                session.commit()

        except Exception as e:
            # Don't crash on validation errors - just log
            logger.warning(f"Remote job validation failed: {e}")

    def submit(self, config: JobConfig) -> None:
        """Submit job to queue, starting immediately if worker slot available."""
        with Session(self._engine) as session:
            existing = session.get(JobState, config.name)
            if existing:
                raise ValueError(
                    f"Job '{config.name}' already exists with status '{existing.status}'. "
                    f"Use get_job_state() to check status before submitting."
                )
            job_state = JobState(name=config.name, config=config, status="pending")
            session.add(job_state)
            session.commit()

        self._try_start_job(config.name)

    def _try_start_job(self, name: str) -> bool:
        with Session(self._engine) as session:
            job_state = session.get(JobState, name)
            if not job_state or job_state.status != "pending":
                return False

            is_remote = job_state.config.remote is not None
            if not self._has_available_slot(is_remote):
                return False

            job = self._spawn_job(job_state)
            self._active_jobs[name] = job
            job_state.status = "running"
            job_state.started_at = datetime.now().isoformat(timespec="seconds")
            session.add(job_state)
            session.commit()

            # Start background monitoring thread for remote jobs
            if is_remote and job_state.job_id:
                try:
                    job_id_int = int(job_state.job_id)
                    self._start_remote_monitor(name, job_id_int)
                except ValueError:
                    pass  # Job ID not available yet, will be monitored when it becomes available

            return True

    def _has_available_slot(self, is_remote: bool) -> bool:
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
        config = job_state.config
        log_dir = str(self.log_dir)
        if config.remote is None:
            job = LocalJob(config, log_dir)
            # Submit local job and capture PID
            job.submit()
            if job.job_id:
                job_state.job_id = job.job_id
            # Set logs_path immediately so monitor can tail logs during execution
            job_state.logs_path = job.log_path
            return job
        else:
            job_id = int(job_state.job_id) if job_state.job_id else None
            job = RemoteJob(config, log_dir, job_id=job_id)
            # Submit remote job and capture request_id
            job.submit()
            if isinstance(job, RemoteJob) and job.request_id:
                job_state.request_id = job.request_id
            if job.job_id:
                job_state.job_id = job.job_id
            # Set logs_path immediately so monitor can tail logs during execution
            job_state.logs_path = job.log_path
            return job

    def poll(self) -> list[str]:
        """Check running jobs for completion, start pending jobs, return completed names."""
        completed = []

        # Update job_id and skypilot_status for remote jobs as they become available
        with Session(self._engine) as session:
            for name, job in list(self._active_jobs.items()):
                if isinstance(job, RemoteJob):
                    job_state = session.get(JobState, name)
                    if job_state:
                        # Update job_id if available and start monitoring thread
                        if job.job_id and not job_state.job_id:
                            job_state.job_id = job.job_id
                            session.add(job_state)
                            session.commit()  # Commit before starting thread

                            # Start monitoring thread if not already running
                            if name not in self._monitor_threads:
                                try:
                                    job_id_int = int(job.job_id)
                                    self._start_remote_monitor(name, job_id_int)
                                except ValueError:
                                    pass

                        # Update SkyPilot status if available (from RemoteJob's own polling)
                        if hasattr(job, "_job_status") and job._job_status:
                            job_state.skypilot_status = job._job_status
                            session.add(job_state)
            session.commit()

        for name, job in list(self._active_jobs.items()):
            if job.is_complete():
                # Fetch logs first (critical for RemoteJob - populates log file)
                try:
                    job.get_logs()
                except Exception:
                    # Log fetch failed, but continue - we'll try to extract what we can
                    pass

                job_result = job.get_result()
                if not job_result:
                    continue

                with Session(self._engine) as session:
                    job_state = session.get(JobState, name)
                    if job_state:
                        job_state.status = "completed"
                        job_state.completed_at = datetime.now().isoformat(timespec="seconds")
                        job_state.exit_code = job_result.exit_code
                        job_state.logs_path = job_result.logs_path
                        job_state.job_id = str(job_result.job_id) if job_result.job_id else None

                        # Extract artifacts from completed job logs for downstream tasks
                        # Priority order: wandb info -> checkpoint URI -> job ID -> metrics
                        if job_state.logs_path:
                            try:
                                logs = Path(job_state.logs_path).read_text(errors="ignore")
                            except Exception:
                                logs = ""

                            if logs:
                                # WandB run info (needed for metrics and default checkpoint URI)
                                if not job_state.wandb_url:
                                    wandb_info = extract_wandb_info(logs)
                                    if wandb_info:
                                        job_state.wandb_run_id = wandb_info.run_id
                                        job_state.wandb_url = wandb_info.url

                                if not job_state.checkpoint_uri:
                                    checkpoint = extract_checkpoint_path(logs)
                                    if checkpoint:
                                        job_state.checkpoint_uri = checkpoint
                                    elif job_state.wandb_run_id:
                                        job_state.checkpoint_uri = f"wandb://run/{job_state.wandb_run_id}"

                                if not job_state.job_id:
                                    skypilot_id = extract_skypilot_job_id(logs)
                                    if skypilot_id:
                                        job_state.job_id = skypilot_id

                                parsed_metrics = extract_final_metrics(logs)
                                if parsed_metrics:
                                    current_metrics = job_state.metrics
                                    job_state.metrics = {**current_metrics, **parsed_metrics}

                        session.add(job_state)
                        session.commit()

                del self._active_jobs[name]
                completed.append(name)

        with Session(self._engine) as session:
            pending_jobs = session.exec(select(JobState).where(JobState.status == "pending")).all()
            for job_state in pending_jobs:
                self._try_start_job(job_state.name)

        return completed

    def wait_for_job(self, name: str, poll_interval_s: float = 1.0) -> JobState:
        while True:
            completed = self.poll()
            if name in completed:
                break
            time.sleep(poll_interval_s)

        job_state = self.get_job_state(name)
        if not job_state:
            raise RuntimeError(f"Job {name} not found after completion")
        return job_state

    def get_status(self, name: str) -> JobStatus | None:
        with Session(self._engine) as session:
            job_state = session.get(JobState, name)
            return job_state.status if job_state else None

    def get_job_state(self, name: str) -> JobState | None:
        with Session(self._engine) as session:
            job_state = session.get(JobState, name)
            if job_state:
                session.expunge(job_state)
            return job_state

    def get_group_jobs(self, group: str) -> dict[str, JobState]:
        with Session(self._engine) as session:
            all_jobs = session.exec(select(JobState)).all()
            group_jobs = [job for job in all_jobs if job.config.group == group]
            for job in group_jobs:
                session.expunge(job)
            return {job.name: job for job in group_jobs}

    def get_all_jobs(self) -> dict[str, JobState]:
        with Session(self._engine) as session:
            all_jobs = session.exec(select(JobState)).all()
            for job in all_jobs:
                session.expunge(job)
            return {job.name: job for job in all_jobs}

    def cancel_group(self, group: str, local_only: bool = False) -> int:
        """Cancel jobs in a group.

        This actually calls job.cancel() to stop jobs:
        - Local jobs: Kills process via SIGTERM/SIGKILL
        - Remote jobs: Cancels job on SkyPilot cluster (if local_only=False)

        Cancelled jobs are marked as "completed" with exit_code=130 (SIGINT).

        Args:
            group: Group name to cancel
            local_only: If True, only cancel local jobs (leave remote jobs running)

        Returns:
            Number of jobs cancelled
        """
        count = 0
        with Session(self._engine) as session:
            all_jobs = session.exec(select(JobState)).all()
            group_jobs = [job for job in all_jobs if job.config.group == group]

            for job_state in group_jobs:
                if job_state.status in ("pending", "running"):
                    is_remote = job_state.config.remote is not None

                    # Skip remote jobs if local_only
                    if local_only and is_remote:
                        continue

                    # Cancel the job if it's active
                    if job_state.name in self._active_jobs:
                        job = self._active_jobs[job_state.name]
                        job.cancel()
                        del self._active_jobs[job_state.name]

                    # Mark as completed with SIGINT exit code
                    job_state.status = "completed"
                    job_state.exit_code = 130  # Standard exit code for SIGINT/user cancel
                    job_state.completed_at = datetime.now().isoformat(timespec="seconds")

                    session.add(job_state)
                    count += 1

            session.commit()
        return count
