"""Job manager with worker pool, queue, and persistence."""

import logging
import threading
import time
from datetime import datetime
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine, select

from devops.skypilot.utils.job_helpers import check_job_statuses
from metta.common.util.constants import SOFTMAX_S3_POLICY_PREFIX
from metta.jobs.job_config import JobConfig
from metta.jobs.job_metrics import fetch_wandb_metrics
from metta.jobs.job_runner import LocalJob, RemoteJob
from metta.jobs.job_state import JobState

logger = logging.getLogger(__name__)

# Exit codes
EXIT_CODE_SKIPPED = -2  # Job skipped due to failed dependency

# Job state statuses (internal to JobManager)
JOB_STATUS_PENDING = "pending"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_COMPLETED = "completed"

# Active job statuses (jobs that can be cancelled or monitored)
ACTIVE_JOB_STATUSES = frozenset({JOB_STATUS_PENDING, JOB_STATUS_RUNNING})

# SkyPilot job statuses (terminal states that indicate job completion)
SKYPILOT_TERMINAL_STATUSES = frozenset(
    {
        "SUCCEEDED",
        "FAILED",
        "FAILED_SETUP",
        "FAILED_DRIVER",
        "CANCELLED",
        "UNKNOWN",
        "ERROR",
    }
)


class JobManager:
    """Manages job execution with concurrency control and persistence.

    Architecture:
    - Maintains worker pools (max_local_jobs, max_remote_jobs)
    - SQLite database stores job state (jobs.sqlite)
    - Job instances (LocalJob/RemoteJob) handle execution
    - Independent monitoring thread per job handles status checks, log fetching, metrics

    Job Lifecycle:
    1. submit() -> Creates JobState in DB, starts job + monitoring thread if slot available
    2. Monitoring thread -> Polls status, fetches logs/metrics, marks complete in DB
    3. poll() -> Returns newly completed jobs, starts pending jobs
    4. Query methods -> get_status(), get_job_state(), get_group_jobs()

    Separation of concerns:
    - JobManager: Worker pools, queue, persistence, monitoring thread coordination
    - Monitoring threads: Job-specific status checks, log fetching, metrics fetching
    - Job (LocalJob/RemoteJob): Execution, log streaming
    - Caller (TaskRunner): Dependencies, acceptance criteria, evaluation
    """

    def __init__(
        self,
        base_dir: Path,
        max_local_jobs: int = 1,
        max_remote_jobs: int = 10,
        remote_poll_interval_s: float = 5.0,
        metrics_fetch_interval_s: float = 300.0,  # Fetch metrics every 5 minutes
    ):
        self.base_dir = Path(base_dir)
        self.db_path = self.base_dir / "jobs.sqlite"
        self.log_dir = self.base_dir / "logs"
        self.max_local_jobs = max_local_jobs
        self.max_remote_jobs = max_remote_jobs
        self.remote_poll_interval_s = remote_poll_interval_s  # How often to poll SkyPilot for remote job status
        self.metrics_fetch_interval_s = metrics_fetch_interval_s  # How often to fetch WandB metrics
        self._active_jobs: dict[str, LocalJob | RemoteJob] = {}
        self._monitor_threads: dict[str, threading.Thread] = {}  # Per-job monitoring threads
        self._jobs_lock = threading.Lock()  # Protects _active_jobs and _monitor_threads from concurrent access
        self._shared_status_monitor: threading.Thread | None = None  # Shared thread for batch status checks
        self._shared_status_monitor_stop = threading.Event()  # Signal to stop shared monitor
        self._init_db()
        self._validate_job_states()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(f"sqlite:///{self.db_path}")
        SQLModel.metadata.create_all(self._engine)

    def _validate_job_states(self) -> None:
        """Validate and reattach to running jobs on startup.

        Local jobs: Mark all as stale (can't reattach to subprocesses).
        Remote jobs: Batch check status, mark completed if finished, or reattach if still running.
        """
        with Session(self._engine) as session:
            running_jobs = session.exec(select(JobState).where(JobState.status == JOB_STATUS_RUNNING)).all()

            local_stale = []
            remote_job_map: dict[int, tuple[str, JobState]] = {}  # job_id -> (job_name, job_state)

            # First pass: categorize jobs
            for job_state in running_jobs:
                is_remote = job_state.config.remote is not None

                if not is_remote:
                    local_stale.append(job_state)
                elif job_state.job_id:
                    try:
                        job_id_int = int(job_state.job_id)
                        remote_job_map[job_id_int] = (job_state.name, job_state)
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid job_id for {job_state.name}: {job_state.job_id}")

            # Batch query all remote job statuses at once (single API call!)
            if remote_job_map:
                job_ids = list(remote_job_map.keys())
                logger.info(f"Batch checking status for {len(job_ids)} remote job(s)")
                try:
                    statuses = check_job_statuses(job_ids)
                except Exception as e:
                    logger.error(f"Failed to batch check job statuses: {e}")
                    statuses = {}

                # Process each remote job based on its status
                reattached_count = 0
                completed_count = 0
                for job_id, (job_name, job_state) in remote_job_map.items():
                    status_info = statuses.get(job_id, {})
                    status = status_info.get("status")

                    if status in SKYPILOT_TERMINAL_STATUSES:
                        # Job finished while we were down - mark complete immediately
                        job_state.status = JOB_STATUS_COMPLETED
                        job_state.skypilot_status = status
                        job_state.exit_code = self._map_skypilot_status_to_exit_code(status)
                        job_state.completed_at = datetime.now().isoformat(timespec="seconds")
                        session.add(job_state)
                        completed_count += 1
                        logger.info(f"Job {job_name} finished while down: {status}")
                    else:
                        # Still running or status unknown - reattach and monitor
                        try:
                            job = RemoteJob(job_state.config, str(self.log_dir), job_id=job_id)
                            with self._jobs_lock:
                                self._active_jobs[job_name] = job
                            # Fetch metrics immediately for reattached jobs
                            self._start_remote_monitor(job_name, job_id, fetch_immediately=True)
                            reattached_count += 1
                            logger.debug(f"Reattached to running job: {job_name} (status: {status or 'unknown'})")
                        except Exception as e:
                            logger.error(f"Failed to reattach to {job_name}: {e}")

                if completed_count > 0:
                    logger.info(f"Marked {completed_count} remote job(s) as completed (finished while down)")
                if reattached_count > 0:
                    logger.info(f"Reattached to {reattached_count} running remote job(s)")

            # Mark local jobs as stale (can't reattach to subprocesses)
            for job_state in local_stale:
                job_state.status = JOB_STATUS_COMPLETED
                job_state.exit_code = -1  # Abnormal termination
                job_state.completed_at = datetime.now().isoformat(timespec="seconds")
                session.add(job_state)

            session.commit()

            if local_stale:
                logger.info(f"Marked {len(local_stale)} local job(s) as stale (cannot reattach to subprocesses)")

    @staticmethod
    def _map_skypilot_status_to_exit_code(status: str) -> int:
        """Map SkyPilot job status to exit code."""
        if status == "SUCCEEDED":
            return 0
        elif status == "CANCELLED":
            return 130  # Standard exit code for SIGINT
        else:
            return 1  # Generic failure

    def _dependencies_satisfied(self, job_state: JobState, session: Session) -> bool:
        """Check if all dependencies are completed successfully.

        A job can start if all its dependencies have:
        - status == JOB_STATUS_COMPLETED
        - exit_code == 0
        - acceptance_passed != False (True or None)

        Args:
            job_state: Job to check dependencies for
            session: Active database session

        Returns:
            True if all dependencies are satisfied (or no dependencies), False otherwise
        """
        if not job_state.config.dependency_names:
            return True

        for dep_name in job_state.config.dependency_names:
            dep_state = session.get(JobState, dep_name)
            if not dep_state:
                logger.debug(f"Job {job_state.name}: dependency {dep_name} not found in database yet")
                return False

            if dep_state.status != "completed":
                logger.debug(f"Job {job_state.name}: waiting for dependency {dep_name} (status: {dep_state.status})")
                return False

            # Check if dependency passed (exit_code 0 + acceptance)
            if dep_state.exit_code != 0:
                logger.warning(
                    f"Job {job_state.name}: dependency {dep_name} failed with exit_code={dep_state.exit_code}"
                )
                return False

            if dep_state.acceptance_passed is False:
                logger.warning(f"Job {job_state.name}: dependency {dep_name} failed acceptance criteria")
                return False

        return True

    def _get_total_timesteps(self, job_config: JobConfig) -> int | None:
        """Extract total_timesteps from job config args.

        Parses args list looking for 'trainer.total_timesteps=X' format.
        """
        for arg in job_config.args:
            if arg.startswith("trainer.total_timesteps="):
                try:
                    value = arg.split("=", 1)[1]
                    return int(value)
                except (ValueError, IndexError):
                    continue
        return None

    def _fetch_and_update_metrics(self, job_state: JobState) -> None:
        """Fetch metrics from WandB and update job_state.metrics (doesn't commit to DB).

        This is a helper that doesn't require a session - caller is responsible for
        adding job_state to session and committing.
        """
        if not job_state.config.metrics_to_track:
            return

        if not job_state.wandb_run_id:
            logger.warning(f"Cannot fetch metrics for {job_state.name}: no wandb_run_id set")
            return

        total_timesteps = self._get_total_timesteps(job_state.config)

        try:
            metrics_data, current_step = fetch_wandb_metrics(
                entity=job_state.config.wandb_entity,
                project=job_state.config.wandb_project,
                run_name=job_state.wandb_run_id,
                metric_keys=job_state.config.metrics_to_track,
            )

            if metrics_data:
                # Extract values and add progress
                metrics_values = {key: data["value"] for key, data in metrics_data.items()}

                if current_step is not None and total_timesteps is not None:
                    metrics_values["_progress"] = {
                        "current_step": current_step,
                        "total_steps": total_timesteps,
                    }

                job_state.metrics = metrics_values

                # Log with value, count, and progress
                metrics_info = ", ".join(
                    f"{key}={data['value']:.2f} (n={int(data['count'])})" for key, data in metrics_data.items()
                )
                progress_info = ""
                if current_step is not None and total_timesteps is not None:
                    progress_pct = (current_step / total_timesteps) * 100
                    progress_info = f" | progress={current_step}/{total_timesteps} ({progress_pct:.1f}%)"

                logger.info(f"Fetched metrics for {job_state.name}: {metrics_info}{progress_info}")
            else:
                logger.warning(f"No metrics returned for {job_state.name} (run may not have data yet)")
        except Exception as e:
            logger.warning(f"Failed to fetch metrics for {job_state.name}: {e}")

    def _fetch_metrics(self, job_name: str, job_state: JobState, session: Session) -> None:
        """Fetch and update metrics for a training job.

        NOTE: This method updates job_state.metrics but does NOT commit the session.
        The caller is responsible for committing the session.
        """
        self._fetch_and_update_metrics(job_state)
        if job_state.metrics:
            session.add(job_state)

    def _handle_local_job_completion(self, job_name: str, job: LocalJob) -> bool:
        """Handle completion of a local job.

        Returns True if job was marked complete, False otherwise.
        """
        # Fetch final logs
        try:
            job.get_logs()
        except Exception:
            pass

        job_result = job.get_result()
        if not job_result:
            return False

        with Session(self._engine) as session:
            job_state = session.get(JobState, job_name)
            if not job_state:
                return False

            job_state.status = JOB_STATUS_COMPLETED
            job_state.completed_at = datetime.now().isoformat(timespec="seconds")
            job_state.exit_code = job_result.exit_code
            job_state.logs_path = job_result.logs_path

            logger.info(f"Job completed: {job_name} (exit_code={job_result.exit_code}, logs={job_result.logs_path})")

            # Fetch final metrics
            if job_state.config.metrics_to_track and job_state.wandb_run_id:
                logger.debug(f"Fetching final metrics for {job_name}")
                self._fetch_and_update_metrics(job_state)

            # Evaluate acceptance criteria
            if job_state.config.acceptance_criteria:
                job_state.acceptance_passed = job_state.evaluate_acceptance()
            else:
                job_state.acceptance_passed = None

            session.add(job_state)
            session.commit()

        # Clean up active job
        with self._jobs_lock:
            if job_name in self._active_jobs:
                del self._active_jobs[job_name]

        return True

    def _handle_remote_job_completion(self, job_name: str, status: str, job_id: int) -> bool:
        """Handle completion of a remote job.

        Returns True if job was marked complete, False otherwise.
        """
        # Fetch final logs
        with self._jobs_lock:
            job = self._active_jobs.get(job_name)
        if job:
            try:
                job.get_logs()
            except Exception:
                pass

        with Session(self._engine) as session:
            job_state = session.get(JobState, job_name)
            if not job_state or job_state.status == JOB_STATUS_COMPLETED:
                return False

            # Determine exit code from SkyPilot status
            exit_code = self._map_skypilot_status_to_exit_code(status)

            job_state.status = JOB_STATUS_COMPLETED
            job_state.completed_at = datetime.now().isoformat(timespec="seconds")
            job_state.exit_code = exit_code
            if not job_state.job_id:
                job_state.job_id = str(job_id)

            logger.info(f"Job completed: {job_name} (skypilot_status={status}, exit_code={exit_code}, job_id={job_id})")

            # Fetch final metrics
            if job_state.config.metrics_to_track and job_state.wandb_run_id:
                logger.debug(f"Fetching final metrics for {job_name}")
                self._fetch_and_update_metrics(job_state)

            # Evaluate acceptance criteria
            if job_state.config.acceptance_criteria:
                job_state.acceptance_passed = job_state.evaluate_acceptance()
            else:
                job_state.acceptance_passed = None

            session.add(job_state)
            session.commit()
            logger.info(
                f"Committed completion for {job_name}: "
                f"status={job_state.status}, exit_code={job_state.exit_code}, "
                f"acceptance={job_state.acceptance_passed}"
            )

        # Clean up active job
        with self._jobs_lock:
            if job_name in self._active_jobs:
                del self._active_jobs[job_name]

        return True

    def _start_local_monitor(self, job_name: str, fetch_immediately: bool = False) -> None:
        """Start background monitoring thread for a local job.

        The thread periodically checks if process is still running and fetches metrics.
        Marks job complete when process exits.

        Args:
            job_name: Name of job to monitor
            fetch_immediately: If True, fetch metrics on first check (for reattached jobs)
        """

        def monitor_loop():
            # For reattached jobs, fetch metrics immediately; for new jobs, wait full interval
            last_metrics_fetch = -self.metrics_fetch_interval_s if fetch_immediately else 0.0

            try:
                while True:
                    # Get job from active_jobs (protected by lock)
                    with self._jobs_lock:
                        job = self._active_jobs.get(job_name)
                    if not job:
                        break

                    # Check if job completed
                    if job.is_complete():
                        self._handle_local_job_completion(job_name, job)
                        break

                    # Fetch metrics periodically while running
                    now = time.time()
                    if now - last_metrics_fetch >= self.metrics_fetch_interval_s:
                        with Session(self._engine) as session:
                            job_state = session.get(JobState, job_name)
                            if job_state and job_state.status == JOB_STATUS_RUNNING:
                                self._fetch_metrics(job_name, job_state, session)
                                session.commit()
                        last_metrics_fetch = now

                    time.sleep(1.0)  # Check every second

            except Exception as e:
                logger.warning(f"Local monitor thread for {job_name} failed: {e}")
            finally:
                # Clean up monitor thread (protected by lock)
                with self._jobs_lock:
                    if job_name in self._monitor_threads:
                        del self._monitor_threads[job_name]

        thread = threading.Thread(target=monitor_loop, daemon=True, name=f"monitor-{job_name}")
        thread.start()
        with self._jobs_lock:
            self._monitor_threads[job_name] = thread
        logger.info(f"Started monitoring thread for local job: {job_name}")

    def _ensure_shared_status_monitor_running(self) -> None:
        """Start shared status monitor if not already running.

        The shared monitor batch-checks all remote job statuses in a single API call
        and updates the database. This is much more efficient than per-job status checks.
        """
        with self._jobs_lock:
            if self._shared_status_monitor is not None and self._shared_status_monitor.is_alive():
                return  # Already running

        def monitor_loop():
            """Batch check all remote job statuses and update database."""
            logger.info("Starting shared remote status monitor")

            # Exponential backoff for transient failures
            retry_delay = 5.0
            max_retry_delay = 1800.0  # Max 30 minutes

            while not self._shared_status_monitor_stop.is_set():
                try:
                    # Get all active remote jobs
                    with self._jobs_lock:
                        remote_jobs = {}
                        for name, job in self._active_jobs.items():
                            if isinstance(job, RemoteJob) and job.job_id:
                                try:
                                    remote_jobs[int(job.job_id)] = name
                                except (ValueError, TypeError):
                                    continue

                    if remote_jobs:
                        # Batch query all at once
                        job_ids = list(remote_jobs.keys())
                        logger.debug(f"Batch checking {len(job_ids)} remote job(s)")

                        try:
                            statuses = check_job_statuses(job_ids)
                            # Reset retry delay on success
                            retry_delay = 5.0

                            # Update database with statuses
                            with Session(self._engine) as session:
                                for job_id, status_info in statuses.items():
                                    job_name = remote_jobs.get(job_id)
                                    if not job_name:
                                        continue

                                    status = status_info.get("status")
                                    if status:
                                        job_state = session.get(JobState, job_name)
                                        if job_state:
                                            job_state.skypilot_status = status
                                            session.add(job_state)

                                session.commit()

                        except Exception as e:
                            logger.warning(f"Shared status monitor failed (will retry in {retry_delay:.0f}s): {e}")
                            retry_delay = min(retry_delay * 2, max_retry_delay)

                    # Wait for next check or stop signal
                    if self._shared_status_monitor_stop.wait(timeout=self.remote_poll_interval_s):
                        break  # Stop signal received

                except Exception as e:
                    logger.error(f"Shared status monitor error: {e}")
                    if self._shared_status_monitor_stop.wait(timeout=self.remote_poll_interval_s):
                        break

            logger.info("Shared remote status monitor stopped")

        self._shared_status_monitor_stop.clear()
        thread = threading.Thread(target=monitor_loop, daemon=True, name="shared-status-monitor")
        thread.start()
        with self._jobs_lock:
            self._shared_status_monitor = thread
        logger.info("Shared status monitor started")

    def _start_remote_monitor(self, job_name: str, job_id: int, fetch_immediately: bool = False) -> None:
        """Start background monitoring thread for a remote job.

        This thread fetches logs and metrics. Status is checked by the shared monitor thread
        which batch-queries all remote jobs in a single API call (more efficient).

        Args:
            job_name: Name of job to monitor
            job_id: SkyPilot job ID
            fetch_immediately: If True, fetch metrics on first check (for reattached jobs)
        """
        # Ensure shared status monitor is running (starts it if needed)
        self._ensure_shared_status_monitor_running()

        def monitor_loop():
            # For reattached jobs, fetch metrics immediately; for new jobs, wait full interval
            last_metrics_fetch = -self.metrics_fetch_interval_s if fetch_immediately else 0.0

            try:
                while True:
                    # Read status from database (written by shared status monitor)
                    with Session(self._engine) as session:
                        job_state = session.get(JobState, job_name)
                        if not job_state:
                            logger.warning(f"Job {job_name} not found in database")
                            break
                        status = job_state.skypilot_status

                    if not status:
                        # Wait for shared monitor to populate status
                        time.sleep(1.0)
                        continue

                    # Check if terminal state reached
                    if status in SKYPILOT_TERMINAL_STATUSES:
                        self._handle_remote_job_completion(job_name, status, job_id)
                        break

                    # While running: fetch logs and metrics periodically
                    if status == "RUNNING":
                        # Fetch logs
                        if job_name in self._active_jobs:
                            job = self._active_jobs[job_name]
                            if isinstance(job, RemoteJob):
                                try:
                                    job.get_logs()
                                except Exception:
                                    pass

                        # Fetch metrics periodically
                        now = time.time()
                        if now - last_metrics_fetch >= self.metrics_fetch_interval_s:
                            with Session(self._engine) as session:
                                job_state = session.get(JobState, job_name)
                                if job_state:
                                    self._fetch_metrics(job_name, job_state, session)
                                    session.commit()
                            last_metrics_fetch = now

                    time.sleep(1.0)  # Check status frequently

            except Exception as e:
                logger.warning(f"Remote monitor thread for {job_name} failed: {e}")
                time.sleep(1.0)

            finally:
                # Clean up monitor thread (protected by lock)
                with self._jobs_lock:
                    if job_name in self._monitor_threads:
                        del self._monitor_threads[job_name]

        thread = threading.Thread(target=monitor_loop, daemon=True, name=f"monitor-{job_name}")
        thread.start()
        with self._jobs_lock:
            self._monitor_threads[job_name] = thread
        logger.info(f"Started monitoring thread for remote job: {job_name} (job_id={job_id})")

    def submit(self, config: JobConfig) -> None:
        """Submit job to queue, starting immediately if worker slot available.

        Note: Job is identified by config.name. Callers are responsible for ensuring
        unique names across their use case (e.g., release_stable.py prefixes names
        with version like "v0.1.0_train_arena" to avoid collisions across releases).
        """

        with Session(self._engine) as session:
            existing = session.get(JobState, config.name)
            if existing:
                raise ValueError(
                    f"Job '{config.name}' already exists with status '{existing.status}'. "
                    f"Use get_job_state() to check status before submitting."
                )
            job_state = JobState(name=config.name, config=config, status="pending")

            # Set checkpoint URI (known at submission time)
            # WandB URL will be extracted from logs once job starts running
            # Uses the same S3 prefix that CheckpointManager uses for policy storage
            job_state.checkpoint_uri = f"{SOFTMAX_S3_POLICY_PREFIX}/{config.name}"

            session.add(job_state)
            session.commit()

        job_type = "remote" if config.remote else "local"
        logger.info(
            f"Job submitted: {config.name} | type={job_type} | module={config.module} | "
            f"is_training={config.is_training_job} | metrics={config.metrics_to_track}"
        )

        self._try_start_job(config.name)

    def _try_start_job(self, name: str) -> bool:
        with Session(self._engine) as session:
            job_state = session.get(JobState, name)
            if not job_state or job_state.status != "pending":
                return False

            # Check dependencies first
            if not self._dependencies_satisfied(job_state, session):
                return False

            is_remote = job_state.config.remote is not None
            if not self._has_available_slot(is_remote):
                return False

            # Spawn job and update state with job metadata
            job = self._spawn_job(job_state.config, existing_job_id=job_state.job_id)
            self._update_job_state_from_spawned_job(job_state, job)

            # Add to active jobs (protected by lock)
            with self._jobs_lock:
                self._active_jobs[name] = job

            # Check if remote job failed to launch (job_id will be None)
            if is_remote and not job_state.job_id:
                # Launch failed - mark as completed with exit code from job
                job_state.status = JOB_STATUS_COMPLETED
                job_state.started_at = datetime.now().isoformat(timespec="seconds")
                job_state.completed_at = datetime.now().isoformat(timespec="seconds")
                job_state.exit_code = getattr(job, "exit_code", 1)  # Default to 1 if not set
                session.add(job_state)
                session.commit()

                logger.error(f"Job {name} failed to launch (no job_id)")

                # Clean up from active jobs
                with self._jobs_lock:
                    if name in self._active_jobs:
                        del self._active_jobs[name]

                return False

            job_state.status = JOB_STATUS_RUNNING
            job_state.started_at = datetime.now().isoformat(timespec="seconds")
            session.add(job_state)

            # Get job_id before session closes (to avoid DetachedInstanceError)
            job_id_value = job_state.job_id

            session.commit()

            job_type = "remote" if is_remote else "local"
            logger.info(f"Job started: {name} (type={job_type})")

        # Start monitoring thread outside the lock to avoid holding lock during thread creation
        if is_remote:
            # Remote job: start monitor once we have job_id
            if job_id_value:
                try:
                    job_id_int = int(job_id_value)
                    self._start_remote_monitor(name, job_id_int)
                except ValueError:
                    pass  # Job ID not available yet
        else:
            # Local job: start monitor immediately
            self._start_local_monitor(name)

        return True

    def _has_available_slot(self, is_remote: bool) -> bool:
        with self._jobs_lock:
            active_count = sum(
                1
                for job in self._active_jobs.values()
                if (not is_remote and isinstance(job, LocalJob)) or (is_remote and isinstance(job, RemoteJob))
            )

        if not is_remote:
            return active_count < self.max_local_jobs
        else:
            return active_count < self.max_remote_jobs

    def _spawn_job(self, config: JobConfig, existing_job_id: str | None = None) -> LocalJob | RemoteJob:
        """Create and submit a job (local or remote).

        Args:
            config: Job configuration
            existing_job_id: For remote jobs, reattach to this job_id if provided

        Returns:
            The created and submitted job
        """
        log_dir = str(self.log_dir)

        if config.remote is None:
            # Local job
            job = LocalJob(config, log_dir)
            job.submit()
            return job
        else:
            # Remote job
            job_id = int(existing_job_id) if existing_job_id else None
            job = RemoteJob(config, log_dir, job_id=job_id)
            job.submit()
            return job

    def _update_job_state_from_spawned_job(self, job_state: JobState, job: LocalJob | RemoteJob) -> None:
        """Update job_state with metadata from a newly spawned job.

        Extracts job_id, logs_path, and WandB info (for training jobs) from the job.
        """
        config = job_state.config

        # Set job_id if available
        if job.job_id:
            job_state.job_id = job.job_id

        # Set logs path so monitor can tail logs during execution
        job_state.logs_path = job.log_path

        # For remote jobs, capture additional metadata
        if isinstance(job, RemoteJob):
            if job.request_id:
                job_state.request_id = job.request_id

            # Only set WandB info for training jobs (they use run= param and log to WandB)
            if job.run_name and config.is_training_job:
                job_state.wandb_run_id = job.run_name
                job_state.wandb_url = (
                    f"https://wandb.ai/{config.wandb_entity}/{config.wandb_project}/runs/{job.run_name}"
                )

    def poll(self) -> list[str]:
        """Start pending jobs and return recently completed job names.

        Note: Job monitoring (status checks, log fetching, metrics) happens in
        independent background threads per job. This just coordinates lifecycle transitions.
        """
        completed = []

        # Update job_id for remote jobs once available, and start their monitoring threads
        with Session(self._engine) as session:
            # Get snapshot of active jobs (protected by lock)
            with self._jobs_lock:
                active_jobs_snapshot = list(self._active_jobs.items())

            for name, job in active_jobs_snapshot:
                if isinstance(job, RemoteJob):
                    job_state = session.get(JobState, name)
                    if job_state and job.job_id and not job_state.job_id:
                        job_state.job_id = job.job_id
                        session.add(job_state)
                        session.commit()

                        logger.info(f"Remote job ID available: {name} (job_id={job.job_id})")

                        # Start monitoring thread if not already running
                        with self._jobs_lock:
                            monitor_exists = name in self._monitor_threads
                        if not monitor_exists:
                            try:
                                job_id_int = int(job.job_id)
                                self._start_remote_monitor(name, job_id_int)
                            except ValueError:
                                logger.warning(f"Invalid job ID for {name}: {job.job_id}")

        # Check for jobs that monitoring threads marked as completed
        with Session(self._engine) as session:
            for name in list(self._active_jobs.keys()):
                job_state = session.get(JobState, name)
                if job_state and job_state.status == JOB_STATUS_COMPLETED:
                    # Monitoring thread finished this job
                    completed.append(name)
                    # Note: _active_jobs cleanup happens in monitoring thread

        # Try to start pending jobs, or skip if dependencies failed
        with Session(self._engine) as session:
            pending_jobs = session.exec(select(JobState).where(JobState.status == JOB_STATUS_PENDING)).all()
            for job_state in pending_jobs:
                # Check if any dependency failed
                if job_state.config.dependency_names and not self._dependencies_satisfied(job_state, session):
                    # Check if dependency failed (vs just not complete yet)
                    has_failed_dep = False
                    for dep_name in job_state.config.dependency_names:
                        dep_state = session.get(JobState, dep_name)
                        if dep_state and dep_state.status == JOB_STATUS_COMPLETED:
                            if dep_state.exit_code != 0 or dep_state.acceptance_passed is False:
                                has_failed_dep = True
                                break

                    if has_failed_dep:
                        # Mark as completed with special exit code for skipped
                        job_state.status = JOB_STATUS_COMPLETED
                        job_state.exit_code = EXIT_CODE_SKIPPED
                        job_state.completed_at = datetime.now().isoformat(timespec="seconds")
                        session.add(job_state)
                        session.commit()
                        logger.info(f"Job {job_state.name} skipped due to failed dependency")
                        completed.append(job_state.name)
                        continue

                # Try to start if dependencies satisfied
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

    def get_status(self, name: str) -> str | None:
        """Get job status. Returns one of: 'pending', 'running', 'completed', or None if not found."""
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
                if job_state.status in ACTIVE_JOB_STATUSES:
                    is_remote = job_state.config.remote is not None

                    # Skip remote jobs if local_only
                    if local_only and is_remote:
                        continue

                    # Cancel the job if it's active (protected by lock)
                    job_to_cancel = None
                    with self._jobs_lock:
                        if job_state.name in self._active_jobs:
                            job_to_cancel = self._active_jobs[job_state.name]
                            del self._active_jobs[job_state.name]
                    # Cancel outside lock to avoid blocking
                    if job_to_cancel:
                        job_to_cancel.cancel()

                    # Mark as completed with SIGINT exit code
                    job_state.status = JOB_STATUS_COMPLETED
                    job_state.exit_code = 130  # Standard exit code for SIGINT/user cancel
                    job_state.completed_at = datetime.now().isoformat(timespec="seconds")

                    session.add(job_state)
                    count += 1

            session.commit()
        return count

    def _reset_dependent_jobs(self, job_name: str, session: Session) -> None:
        """Reset all jobs that depend on this job (transitively) from skipped back to pending.

        This allows retrying a failed job to also retry all jobs that were skipped due to
        the failure. Works transitively - if A depends on B and B depends on C, resetting C
        will reset both B and A.

        Args:
            job_name: Name of job being reset/retried
            session: Active database session
        """
        # Find all jobs that directly or transitively depend on this one
        all_jobs = session.exec(select(JobState)).all()

        # BFS to find transitive dependents
        dependent_names = set()
        to_check = [job_name]

        while to_check:
            current = to_check.pop(0)
            for job_state in all_jobs:
                if current in job_state.config.dependency_names and job_state.name not in dependent_names:
                    dependent_names.add(job_state.name)
                    to_check.append(job_state.name)

        # Reset skipped jobs back to pending
        for dep_name in dependent_names:
            dep_state = session.get(JobState, dep_name)
            if dep_state and dep_state.status == JOB_STATUS_COMPLETED and dep_state.exit_code == EXIT_CODE_SKIPPED:
                dep_state.status = JOB_STATUS_PENDING
                dep_state.exit_code = None
                dep_state.completed_at = None
                session.add(dep_state)
                logger.info(f"Reset skipped job {dep_name} back to pending (dependency {job_name} being retried)")

    def delete_job(self, name: str) -> bool:
        """Delete a job from the database and clean up its log files.

        Useful for retrying failed jobs - deletes the old state so a new job can be submitted.
        Can delete pending jobs (not started yet) or completed jobs.
        Also resets any dependent jobs that were skipped due to this job's failure.

        Args:
            name: Name of job to delete

        Returns:
            True if job was deleted, False if job didn't exist
        """
        with Session(self._engine) as session:
            job_state = session.get(JobState, name)
            if not job_state:
                return False

            # Don't allow deleting running jobs - they need to be cancelled first
            if job_state.status == JOB_STATUS_RUNNING:
                raise ValueError(f"Cannot delete job '{name}' with status 'running'. Cancel it first.")

            # Reset dependent jobs that were skipped before deleting
            self._reset_dependent_jobs(name, session)

            # Clean up log file if it exists
            if job_state.logs_path:
                try:
                    log_path = Path(job_state.logs_path)
                    if log_path.exists():
                        log_path.unlink()
                        logger.debug(f"Deleted log file: {job_state.logs_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete log file {job_state.logs_path}: {e}")

            session.delete(job_state)
            session.commit()
            logger.info(f"Deleted job: {name}")
            return True
