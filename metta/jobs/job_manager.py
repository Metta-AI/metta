"""Job manager with worker pool, queue, and persistence."""

import json
import logging
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine, select

from devops.skypilot.utils.job_helpers import check_job_statuses
from metta.common.util.constants import METTA_WANDB_ENTITY, METTA_WANDB_PROJECT
from metta.jobs.job_metrics import _strip_ansi_codes, parse_cogames_eval_results, parse_cogames_stats_from_logs
from metta.jobs.job_config import JobConfig, MetricsSource
from metta.jobs.job_metrics import extract_skypilot_job_id, fetch_wandb_metrics
from metta.jobs.job_runner import LocalJob, RemoteJob
from metta.jobs.job_state import JobState, JobStatus

logger = logging.getLogger(__name__)


class ExitCode:
    """Special exit codes."""

    SKIPPED = -2  # Job skipped due to failed dependency


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
        self._monitor_threads: dict[str, threading.Thread] = {}  # Job monitoring threads
        self._jobs_lock = threading.Lock()  # Protects _active_jobs and _monitor_threads from concurrent access
        self._init_db()
        self._validate_job_states()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(f"sqlite:///{self.db_path}")
        SQLModel.metadata.create_all(self._engine)

    def _validate_job_states(self) -> None:
        """Validate and reattach to running jobs on startup.

        Local jobs: Mark all as stale (can't reattach to subprocesses).
        Remote jobs: Reattach and start monitoring threads.
        """
        with Session(self._engine) as session:
            running_jobs = session.exec(select(JobState).where(JobState.status == "running")).all()

            local_stale_count = 0
            remote_jobs = []

            for job_state in running_jobs:
                is_remote = job_state.config.remote is not None

                if not is_remote:
                    # Mark all local running jobs as stale on startup (can't reattach to subprocesses)
                    job_state.status = "completed"
                    job_state.exit_code = ExitCode.ABNORMAL
                    job_state.completed_at = datetime.now().isoformat(timespec="seconds")
                    session.add(job_state)
                    local_stale_count += 1
                else:
                    # Reattach remote jobs - monitoring thread will validate status
                    if job_state.job_id:
                        try:
                            job_id_int = int(job_state.job_id)
                            job = RemoteJob(job_state.config, str(self.log_dir), job_id=job_id_int)
                            self._active_jobs[job_state.name] = job
                            # Fetch metrics immediately for reattached jobs (they've been running for a while)
                            self._start_remote_monitor(job_state.name, job_id_int, fetch_immediately=True)
                            remote_jobs.append(job_state.name)
                        except (ValueError, TypeError):
                            # Invalid job_id, will be cleaned up by validation
                            remote_jobs.append((job_state.name, job_state.job_id))

            session.commit()

            if local_stale_count > 0:
                logger.info(f"Marked {local_stale_count} local job(s) as stale (cannot reattach to subprocesses)")
            if remote_jobs:
                logger.info(f"Reattached to {len(remote_jobs)} remote job(s)")

    def _get_total_timesteps(self, job_config: JobConfig) -> int | None:
        """Extract total_timesteps from job config overrides."""
        if "trainer.total_timesteps" in job_config.overrides:
            try:
                return int(job_config.overrides["trainer.total_timesteps"])
            except (ValueError, TypeError):
                return None
        return None

    def _fetch_metrics(self, job_name: str, job_state: JobState) -> None:
        """Fetch and update metrics for a job based on its metrics_source.

        Routes to the appropriate metrics handler:
        - MetricsSource.WANDB: Fetch from WandB API
        - MetricsSource.COGAMES_LOG: Parse from cogames log output
        - MetricsSource.NONE: No-op
        """
        if not job_state.config.metrics_to_track:
            logger.debug(f"Skipping metrics fetch for {job_name}: no metrics_to_track configured")
            return

        # Route to appropriate metrics handler based on source
        if job_state.config.metrics_source == MetricsSource.COGAMES_LOG:
            self._fetch_cogames_metrics(job_name, job_state)
        elif job_state.config.metrics_source == MetricsSource.WANDB:
            self._fetch_wandb_metrics(job_name, job_state)
        else:
            logger.debug(f"Skipping metrics fetch for {job_name}: metrics_source={job_state.config.metrics_source}")

    def _fetch_cogames_metrics(self, job_name: str, job_state: JobState) -> None:
        """Parse cogames stats from log output.

        Parses both eval results (for metrics like heart.gained) and training logs (for metrics like SPS).
        This allows mixing eval-based and training-based metrics in the same task.
        """
        if not job_state.logs_path:
            logger.debug(f"Cannot parse cogames metrics for {job_name}: no logs_path set")
            return

        try:
            # Read log file
            log_path = Path(job_state.logs_path)
            if not log_path.exists():
                logger.debug(f"Cannot parse cogames metrics for {job_name}: log file not found at {log_path}")
                return

            log_text = log_path.read_text(errors="ignore")

            # Parse eval results AND training logs to support mixed metrics

            eval_metrics = parse_cogames_eval_results(
                log_text=log_text,
                metric_keys=job_state.config.metrics_to_track,
            )

            train_metrics = parse_cogames_stats_from_logs(
                log_text=log_text,
                metric_keys=job_state.config.metrics_to_track,
                last_n_percent=0.25,
            )

            # Merge both sources (eval takes precedence if same key exists)
            metrics_data = {**train_metrics, **eval_metrics}

            if metrics_data:
                # Extract just the values for storage (backward compatible with display)
                metrics_values = {key: data["value"] for key, data in metrics_data.items()}

                with Session(self._engine) as session:
                    job_state_fresh = session.get(JobState, job_name)
                    if job_state_fresh:
                        job_state_fresh.metrics = metrics_values
                        session.add(job_state_fresh)
                        session.commit()

                        # Log with value and count
                        metrics_info = ", ".join(
                            f"{key}={data['value']:.2f} (n={int(data['count'])})" for key, data in metrics_data.items()
                        )
                        logger.info(f"Parsed cogames metrics for {job_name}: {metrics_info}")
            else:
                logger.debug(f"No cogames metrics found in logs for {job_name}")

        except Exception as e:
            logger.warning(f"Failed to parse cogames metrics for {job_name}: {e}")

    def _fetch_wandb_metrics(self, job_name: str, job_state: JobState) -> None:
        """Fetch metrics from WandB API for tool-based training jobs."""
        if not job_state.wandb_run_id:
            logger.warning(f"Cannot fetch metrics for {job_name}: no wandb_run_id set")
            return

        # Get total timesteps for progress tracking
        total_timesteps = self._get_total_timesteps(job_state.config)

        logger.debug(
            f"Fetching metrics for {job_name}: run={job_state.wandb_run_id}, "
            f"metrics={job_state.config.metrics_to_track}, total_timesteps={total_timesteps}"
        )

        try:
            metrics_data, current_step = fetch_wandb_metrics(
                entity=METTA_WANDB_ENTITY,
                project=METTA_WANDB_PROJECT,
                run_name=job_state.wandb_run_id,
                metric_keys=job_state.config.metrics_to_track,
            )
            if metrics_data:
                # Extract just the values for storage (backward compatible with display)
                metrics_values = {key: data["value"] for key, data in metrics_data.items()}

                # Add progress tracking if we have both current and total steps
                if current_step is not None and total_timesteps is not None:
                    metrics_values["_progress"] = {
                        "current_step": current_step,
                        "total_steps": total_timesteps,
                    }

                with Session(self._engine) as session:
                    job_state_fresh = session.get(JobState, job_name)
                    if job_state_fresh:
                        job_state_fresh.metrics = metrics_values
                        session.add(job_state_fresh)
                        session.commit()

                        # Log with value, count, and progress
                        metrics_info = ", ".join(
                            f"{key}={data['value']:.2f} (n={int(data['count'])})" for key, data in metrics_data.items()
                        )
                        progress_info = ""
                        if current_step is not None and total_timesteps is not None:
                            progress_pct = (current_step / total_timesteps) * 100
                            progress_info = f" | progress={current_step}/{total_timesteps} ({progress_pct:.1f}%)"

                        logger.info(f"Fetched metrics for {job_name}: {metrics_info}{progress_info}")
            else:
                logger.warning(f"No metrics returned for {job_name} (run may not have data yet)")
        except Exception as e:
            logger.warning(f"Failed to fetch metrics for {job_name}: {e}")

    def _fetch_artifacts(self, job_name: str, job_state: JobState) -> None:
        """Download and parse artifacts from S3.

        For each declared artifact:
        1. Build S3 path using convention: s3://softmax-public/stable/jobs/{job_name}/{artifact_name}
        2. Download artifact to temp location
        3. Parse content based on artifact type
        4. Extract metrics and merge into job_state.metrics
        """
        if not job_state.config.artifacts:
            logger.debug(f"Skipping artifact fetch for {job_name}: no artifacts declared")
            return

        try:
            for artifact_name in job_state.config.artifacts:
                # Build S3 path using convention
                s3_path = f"s3://softmax-public/stable/jobs/{job_name}/{artifact_name}"

                # Download artifact to temp file
                temp_file = Path(f"/tmp/{job_name}_{artifact_name}")
                download_cmd = ["aws", "s3", "cp", s3_path, str(temp_file)]

                result = subprocess.run(download_cmd, capture_output=True, text=True, timeout=30)
                if result.returncode != 0:
                    logger.warning(f"Failed to download artifact {artifact_name} for {job_name}: {result.stderr}")
                    continue

                # Parse artifact based on type
                if artifact_name == "eval_results.json":
                    # Parse eval results JSON
                    raw_content = temp_file.read_text()
                    clean_content = _strip_ansi_codes(raw_content)

                    # Find JSON start (skip any leading text)
                    json_start = clean_content.find("{")
                    if json_start == -1:
                        logger.warning(f"No JSON found in artifact {artifact_name} for {job_name}")
                        continue

                    json_str = clean_content[json_start:]
                    eval_data = json.loads(json_str)

                    # Extract metrics from eval JSON
                    # Structure: {"missions": [{"mission_name": "...",
                    #              "policy_summaries": [{"avg_agent_metrics": {...}}]}]}
                    if "missions" in eval_data and eval_data["missions"]:
                        mission = eval_data["missions"][0]
                        if "policy_summaries" in mission and mission["policy_summaries"]:
                            policy_summary = mission["policy_summaries"][0]

                            # Extract all avg_agent_metrics
                            if "avg_agent_metrics" in policy_summary:
                                avg_metrics = policy_summary["avg_agent_metrics"]

                                # Only extract metrics that are tracked
                                artifact_metrics = {}
                                for metric_key in job_state.config.metrics_to_track:
                                    # Handle nested keys like "avg_agent_metrics.energy.gained"
                                    if metric_key.startswith("avg_agent_metrics."):
                                        metric_name = metric_key.replace("avg_agent_metrics.", "")
                                        if metric_name in avg_metrics:
                                            artifact_metrics[metric_key] = avg_metrics[metric_name]

                                if artifact_metrics:
                                    # Merge with existing metrics
                                    with Session(self._engine) as session:
                                        job_state_fresh = session.get(JobState, job_name)
                                        if job_state_fresh:
                                            existing_metrics = job_state_fresh.metrics or {}
                                            existing_metrics.update(artifact_metrics)
                                            job_state_fresh.metrics = existing_metrics
                                            session.add(job_state_fresh)
                                            session.commit()

                                            metrics_info = ", ".join(
                                                f"{key}={value:.2f}" for key, value in artifact_metrics.items()
                                            )
                                            logger.info(f"Parsed artifact metrics for {job_name}: {metrics_info}")

                # Clean up temp file
                if temp_file.exists():
                    temp_file.unlink()

        except Exception as e:
            logger.warning(f"Failed to fetch artifacts for {job_name}: {e}")

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
                    job = self._active_jobs.get(job_name)
                    if not job:
                        break

                    # Check if job completed
                    if job.is_complete():
                        # Fetch logs and mark complete
                        try:
                            job.get_logs()
                        except Exception:
                            pass

                        job_result = job.get_result()
                        if job_result:
                            with Session(self._engine) as session:
                                job_state = session.get(JobState, job_name)
                                if job_state:
                                    job_state.status = "completed"
                                    job_state.completed_at = datetime.now().isoformat(timespec="seconds")
                                    job_state.exit_code = job_result.exit_code
                                    job_state.logs_path = job_result.logs_path

                                    logger.info(
                                        f"Job completed: {job_name} (exit_code={job_result.exit_code}, "
                                        f"logs={job_result.logs_path})"
                                    )

                                    # Fetch final metrics
                                    if job_state.config.metrics_to_track:
                                        logger.debug(f"Fetching final metrics for {job_name}")
                                        self._fetch_metrics(job_name, job_state)

                                    # Fetch artifacts if declared
                                    if job_state.config.artifacts:
                                        logger.debug(f"Fetching artifacts for {job_name}")
                                        self._fetch_artifacts(job_name, job_state)

                                    session.add(job_state)
                                    session.commit()

                            del self._active_jobs[job_name]
                        break

                    # Fetch metrics periodically while running
                    now = time.time()
                    if now - last_metrics_fetch >= self.metrics_fetch_interval_s:
                        with Session(self._engine) as session:
                            job_state = session.get(JobState, job_name)
                            if job_state and job_state.status == "running":
                                self._fetch_metrics(job_name, job_state)
                        last_metrics_fetch = now

                    time.sleep(1.0)  # Check every second

            except Exception as e:
                logger.warning(f"Local monitor thread for {job_name} failed: {e}")
            finally:
                if job_name in self._monitor_threads:
                    del self._monitor_threads[job_name]

        thread = threading.Thread(target=monitor_loop, daemon=True, name=f"monitor-{job_name}")
        thread.start()
        self._monitor_threads[job_name] = thread
        logger.info(f"Started monitoring thread for local job: {job_name}")

    def _start_remote_monitor(self, job_name: str, job_id: int, fetch_immediately: bool = False) -> None:
        """Start background monitoring thread for a remote job.

        The thread polls SkyPilot API, fetches logs, fetches metrics, and marks complete.

        Args:
            job_name: Name of job to monitor
            job_id: SkyPilot job ID
            fetch_immediately: If True, fetch metrics on first check (for reattached jobs)
        """

        def monitor_loop():
            # For reattached jobs, fetch metrics immediately; for new jobs, wait full interval
            last_metrics_fetch = -self.metrics_fetch_interval_s if fetch_immediately else 0.0

            try:
                while True:
                    try:
                        # Query SkyPilot for current status
                        statuses = check_job_statuses([job_id])
                        status = statuses.get(job_id, {}).get("status")

                        if not status:
                            break

                        # Update database with current status
                        with Session(self._engine) as session:
                            job_state = session.get(JobState, job_name)
                            if job_state:
                                job_state.skypilot_status = status
                                session.add(job_state)
                                session.commit()

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
                                        self._fetch_metrics(job_name, job_state)
                                last_metrics_fetch = now

                        # Check if terminal state reached
                        if status in (
                            "SUCCEEDED",
                            "FAILED",
                            "FAILED_SETUP",
                            "FAILED_DRIVER",
                            "CANCELLED",
                            "UNKNOWN",
                            "ERROR",
                        ):
                            # Mark job complete
                            if job_name in self._active_jobs:
                                job = self._active_jobs[job_name]
                                try:
                                    job.get_logs()
                                except Exception:
                                    pass

                                job_result = job.get_result()
                                if job_result:
                                    with Session(self._engine) as session:
                                        job_state = session.get(JobState, job_name)
                                        if job_state:
                                            job_state.status = "completed"
                                            job_state.completed_at = datetime.now().isoformat(timespec="seconds")
                                            job_state.exit_code = job_result.exit_code
                                            job_state.logs_path = job_result.logs_path
                                            job_state.job_id = str(job_result.job_id) if job_result.job_id else None

                                            logger.info(
                                                f"Job completed: {job_name} (skypilot_status={status}, "
                                                f"exit_code={job_result.exit_code}, job_id={job_id})"
                                            )

                                            # Extract job ID from logs if not set
                                            if job_state.logs_path and not job_state.job_id:
                                                try:
                                                    logs = Path(job_state.logs_path).read_text(errors="ignore")
                                                    skypilot_id = extract_skypilot_job_id(logs)
                                                    if skypilot_id:
                                                        job_state.job_id = skypilot_id
                                                except Exception:
                                                    pass

                                            # Fetch final metrics
                                            if job_state.config.metrics_to_track:
                                                logger.debug(f"Fetching final metrics for {job_name}")
                                                self._fetch_metrics(job_name, job_state)

                                            # Fetch artifacts if declared
                                            if job_state.config.artifacts:
                                                logger.debug(f"Fetching artifacts for {job_name}")
                                                self._fetch_artifacts(job_name, job_state)

                                            session.add(job_state)
                                            session.commit()

                                    del self._active_jobs[job_name]
                            break

                        time.sleep(self.remote_poll_interval_s)

                    except Exception as e:
                        logger.warning(f"Remote monitor thread for {job_name} failed: {e}")
                        time.sleep(self.remote_poll_interval_s)

            finally:
                if job_name in self._monitor_threads:
                    del self._monitor_threads[job_name]

        thread = threading.Thread(target=monitor_loop, daemon=True, name=f"monitor-{job_name}")
        thread.start()
        self._monitor_threads[job_name] = thread
        logger.info(f"Started monitoring thread for remote job: {job_name} (job_id={job_id})")

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

            # Set checkpoint URI (known at submission time)
            # WandB URL will be extracted from logs once job starts running
            job_state.checkpoint_uri = f"s3://softmax-public/policies/{config.name}"

            session.add(job_state)
            session.commit()

        job_type = "remote" if config.remote else "local"
        task_spec = config.tool if config.tool else config.cmd
        logger.info(
            f"Job submitted: {config.name} | type={job_type} | task={task_spec} | "
            f"metrics_source={config.metrics_source.value} | metrics={config.metrics_to_track}"
        )

        self._try_start_job(config.name)

    def _dependencies_satisfied(self, job_state: JobState, session: Session) -> bool:
        """Check if all dependencies are completed successfully.

        A job can start if all its dependencies have:
        - status == "completed"
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
                logger.debug(f"Dependency {dep_name} not found for {job_state.name}")
                return False

            if dep_state.status != "completed":
                logger.debug(f"Dependency {dep_name} not completed for {job_state.name} (status={dep_state.status})")
                return False

            if dep_state.exit_code != 0:
                logger.debug(f"Dependency {dep_name} failed for {job_state.name} (exit_code={dep_state.exit_code})")
                return False

            if dep_state.acceptance_passed is False:
                logger.debug(f"Dependency {dep_name} failed acceptance for {job_state.name}")
                return False

        return True

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

            job = self._spawn_job(job_state)

            # Check if remote job failed to launch (no request_id means launch failed)
            if is_remote and isinstance(job, RemoteJob) and not job.request_id:
                # Launch failed - mark as completed with exit code from job
                job_state.status = "completed"
                job_state.started_at = datetime.now().isoformat(timespec="seconds")
                job_state.completed_at = datetime.now().isoformat(timespec="seconds")
                job_state.exit_code = getattr(job, "_exit_code", None) or 1  # Default to 1 if not set
                session.add(job_state)
                session.commit()

                logger.error(f"Job {name} failed to launch (no request_id)")

                # Clean up from active jobs
                with self._jobs_lock:
                    if name in self._active_jobs:
                        del self._active_jobs[name]

                return False

            with self._jobs_lock:
                self._active_jobs[name] = job
            job_state.status = "running"
            job_state.started_at = datetime.now().isoformat(timespec="seconds")
            session.add(job_state)
            session.commit()

            job_type = "remote" if is_remote else "local"
            logger.info(f"Job started: {name} (type={job_type})")

            # Start background monitoring thread
            if is_remote:
                # Remote job: start monitor once we have job_id
                if job_state.job_id:
                    try:
                        job_id_int = int(job_state.job_id)
                        self._start_remote_monitor(name, job_id_int)
                    except ValueError:
                        pass  # Job ID not available yet, will start monitor when it becomes available
            else:
                # Local job: start monitor immediately
                self._start_local_monitor(name)

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
            # Submit remote job and capture request_id and run_name
            job.submit()
            if isinstance(job, RemoteJob):
                if job.request_id:
                    job_state.request_id = job.request_id
                # Only set WandB info for WandB-tracked jobs
                if job.run_name and config.metrics_source == MetricsSource.WANDB:
                    job_state.wandb_run_id = job.run_name
                    job_state.wandb_url = (
                        f"https://wandb.ai/{METTA_WANDB_ENTITY}/{METTA_WANDB_PROJECT}/runs/{job.run_name}"
                    )
            if job.job_id:
                job_state.job_id = job.job_id
            # Set logs_path immediately so monitor can tail logs during execution
            job_state.logs_path = job.log_path
            return job

    def poll(self) -> list[str]:
        """Start pending jobs and return recently completed job names.

        Note: Job monitoring (status checks, log fetching, metrics) happens in
        independent background threads per job. This just coordinates lifecycle transitions.
        """
        completed = []

        # Update job_id for remote jobs once available, and start their monitoring threads
        with Session(self._engine) as session:
            for name, job in list(self._active_jobs.items()):
                if isinstance(job, RemoteJob):
                    job_state = session.get(JobState, name)
                    if job_state and job.job_id and not job_state.job_id:
                        job_state.job_id = job.job_id
                        session.add(job_state)
                        session.commit()

                        logger.info(f"Remote job ID available: {name} (job_id={job.job_id})")

                        # Start monitoring thread if not already running
                        if name not in self._monitor_threads:
                            try:
                                job_id_int = int(job.job_id)
                                self._start_remote_monitor(name, job_id_int)
                            except ValueError:
                                logger.warning(f"Invalid job ID for {name}: {job.job_id}")

        # Check for jobs that monitoring threads marked as completed
        with Session(self._engine) as session:
            for name in list(self._active_jobs.keys()):
                job_state = session.get(JobState, name)
                if job_state and job_state.status == "completed":
                    # Monitoring thread finished this job
                    completed.append(name)
                    # Note: _active_jobs cleanup happens in monitoring thread

        # Try to start pending jobs, or skip if dependencies failed
        with Session(self._engine) as session:
            pending_jobs = session.exec(select(JobState).where(JobState.status == "pending")).all()
            for job_state in pending_jobs:
                # Check if any dependency failed
                if job_state.config.dependency_names and not self._dependencies_satisfied(job_state, session):
                    # Check if dependency failed (vs just not complete yet)
                    has_failed_dep = False
                    for dep_name in job_state.config.dependency_names:
                        dep_state = session.get(JobState, dep_name)
                        if dep_state and dep_state.status == "completed":
                            if dep_state.exit_code != 0 or dep_state.acceptance_passed is False:
                                has_failed_dep = True
                                break

                    if has_failed_dep:
                        # Mark as completed with special exit code for skipped
                        job_state.status = "completed"
                        job_state.exit_code = ExitCode.SKIPPED
                        job_state.completed_at = datetime.now().isoformat(timespec="seconds")
                        session.add(job_state)
                        session.commit()
                        logger.info(f"Job {job_state.name} skipped due to failed dependency")
                        completed.append(job_state.name)
                        continue

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

    def get_status_summary(self, group: str | None = None) -> dict:
        """Get aggregated status summary for jobs.

        Provides high-level statistics and job list for building displays.

        Args:
            group: Optional group filter (only include jobs in this group)

        Returns:
            Dict with keys:
            - total: Total number of jobs
            - completed: Number of completed jobs
            - running: Number of running jobs
            - pending: Number of pending jobs
            - succeeded: Number of successful jobs (exit_code 0)
            - failed: Number of failed jobs (exit_code != 0)
            - jobs: List of job dicts with status, metrics, artifacts
        """
        # Query jobs
        if group:
            jobs = self.get_group_jobs(group)
        else:
            jobs = self.get_all_jobs()

        # Count statuses
        total = len(jobs)
        completed = sum(1 for js in jobs.values() if js.status == "completed")
        running = sum(1 for js in jobs.values() if js.status == "running")
        pending = sum(1 for js in jobs.values() if js.status == "pending")
        succeeded = sum(1 for js in jobs.values() if js.status == "completed" and js.exit_code == 0)
        failed = sum(1 for js in jobs.values() if js.status == "completed" and js.exit_code != 0)

        # Build job list with relevant info
        job_list = []
        for name, job_state in jobs.items():
            job_dict = {
                "name": name,
                "status": job_state.status,
                "exit_code": job_state.exit_code if job_state.status == "completed" else None,
                "job_id": job_state.job_id,
                "request_id": job_state.request_id,
                "skypilot_status": job_state.skypilot_status,
                "logs_path": job_state.logs_path,
                "metrics": job_state.metrics or {},
                "wandb_url": job_state.wandb_url,
                "checkpoint_uri": job_state.checkpoint_uri,
                "started_at": job_state.started_at,
                "completed_at": job_state.completed_at,
            }

            # Add display-friendly fields for completed jobs
            if job_state.status == "completed":
                job_dict["success"] = job_state.exit_code == 0

                # Calculate duration
                if job_state.started_at and job_state.completed_at:
                    try:
                        started = datetime.fromisoformat(job_state.started_at)
                        completed_at = datetime.fromisoformat(job_state.completed_at)
                        job_dict["duration_s"] = (completed_at - started).total_seconds()
                    except Exception:
                        pass

            job_list.append(job_dict)

        return {
            "total": total,
            "completed": completed,
            "running": running,
            "pending": pending,
            "succeeded": succeeded,
            "failed": failed,
            "jobs": job_list,
        }

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
                    job_state.exit_code = ExitCode.CANCELLED
                    job_state.completed_at = datetime.now().isoformat(timespec="seconds")

                    session.add(job_state)
                    count += 1

            session.commit()
        return count

    def delete_job(self, name: str) -> bool:
        """Delete a job from the database and clean up its log files.

        Useful for retrying failed jobs - deletes the old state so a new job can be submitted.
        Can delete pending jobs (not started yet) or completed jobs.

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
            if job_state.status == "running":
                raise ValueError(f"Cannot delete job '{name}' with status 'running'. Cancel it first.")

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
