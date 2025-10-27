"""Job manager with worker pool, queue, and persistence."""

import time
from datetime import datetime
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine, select

from metta.jobs.job_config import JobConfig
from metta.jobs.job_metrics import (
    extract_checkpoint_path,
    extract_final_metrics,
    extract_skypilot_job_id,
    extract_wandb_info,
)
from metta.jobs.job_runner import LocalJob, RemoteJob
from metta.jobs.job_state import JobState, JobStatus


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
    ):
        self.base_dir = Path(base_dir)
        self.db_path = self.base_dir / "jobs.sqlite"
        self.log_dir = self.base_dir / "logs"
        self.max_local_jobs = max_local_jobs
        self.max_remote_jobs = max_remote_jobs
        self._active_jobs: dict[str, LocalJob | RemoteJob] = {}
        self._init_db()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(f"sqlite:///{self.db_path}")
        SQLModel.metadata.create_all(self._engine)

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
            job_state.started_at = datetime.utcnow().isoformat(timespec="seconds")
            session.add(job_state)
            session.commit()
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
            return LocalJob(config, log_dir)
        else:
            job_id = int(job_state.job_id) if job_state.job_id else None
            return RemoteJob(config, log_dir, job_id=job_id)

    def poll(self) -> list[str]:
        """Check running jobs for completion, start pending jobs, return completed names."""
        completed = []

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
                        job_state.completed_at = datetime.utcnow().isoformat(timespec="seconds")
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

    def cancel_group(self, group: str) -> int:
        count = 0
        with Session(self._engine) as session:
            all_jobs = session.exec(select(JobState)).all()
            group_jobs = [job for job in all_jobs if job.config.group == group]

            for job_state in group_jobs:
                if job_state.status in ("pending", "running"):
                    if job_state.name in self._active_jobs:
                        job = self._active_jobs[job_state.name]
                        job.cancel()
                        del self._active_jobs[job_state.name]
                    job_state.status = "cancelled"
                    session.add(job_state)
                    count += 1

            session.commit()
        return count
