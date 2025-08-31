import functools
import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum, auto
from typing import Any, Protocol, runtime_checkable

from cogweb.cogweb_client import CogwebClient
from metta.common.wandb.wandb_context import WandbConfig
from metta.sweep.protein_config import ProteinConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Utility Functions
# ============================================================================


def make_monitor_table(
    runs: list["RunInfo"],
    title: str = "Run Status Table",
    logger_prefix: str = "",
    include_score: bool = True,
    truncate_run_id: bool = True,
) -> list[str]:
    """Create a formatted table showing run status.

    Args:
        runs: List of RunInfo objects to display
        title: Title for the table
        logger_prefix: Prefix to add to each log line (e.g., "[OptimizingScheduler]")
        include_score: Whether to include the score column
        truncate_run_id: Whether to truncate run IDs to just show trial numbers

    Returns:
        List of formatted lines that can be logged
    """
    lines = []
    prefix = f"{logger_prefix} " if logger_prefix else ""

    # Title
    lines.append(f"{prefix}{title}:")
    lines.append(f"{prefix}{'=' * 70}")

    # Header
    if include_score:
        lines.append(f"{prefix}{'Run ID':<30} {'Status':<25} {'Score':<15}")
    else:
        lines.append(f"{prefix}{'Run ID':<30} {'Status':<40}")
    lines.append(f"{prefix}{'-' * 70}")

    # Rows
    for run in runs:
        # Format run ID
        display_id = run.run_id
        if truncate_run_id and "_trial_" in run.run_id:
            display_id = run.run_id.split("_trial_")[-1]
            display_id = f"trial_{display_id}" if not display_id.startswith("trial_") else display_id

        # Format score
        if include_score:
            score_str = f"{run.observation.score:.4f}" if run.observation else "N/A"
            lines.append(f"{prefix}{display_id:<30} {str(run.status):<25} {score_str:<15}")
        else:
            lines.append(f"{prefix}{display_id:<30} {str(run.status):<40}")

    lines.append(f"{prefix}{'=' * 70}")

    return lines


# ============================================================================
# Core Data Models
# ============================================================================


class JobTypes(StrEnum):
    LAUNCH_TRAINING = auto()
    LAUNCH_EVAL = auto()

    # For the future
    PAUSE_TRAINING = auto()
    RESUME_TRAINING = auto()
    CANCEL_JOB = auto()


@dataclass
class JobDefinition:
    run_id: str
    cmd: str  # e.g., "experiments.recipes.arena.train_shaped" or "experiments.recipes.arena.evaluate"
    gpus: int = 1
    nodes: int = 1
    args: list[str] = field(default_factory=list)  # positional arguments
    overrides: dict[str, Any] = field(default_factory=dict)  # key=value overrides for the tool
    config: dict[str, Any] = field(default_factory=dict)  # additional config from optimizer
    type: JobTypes = JobTypes.LAUNCH_TRAINING  # JobTypes enum value
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


class JobStatus(StrEnum):
    PENDING = auto()  # Initialized but not started
    IN_TRAINING = auto()
    TRAINING_DONE_NO_EVAL = auto()
    IN_EVAL = auto()
    EVAL_DONE_NOT_COMPLETED = auto()
    COMPLETED = auto()
    FAILED = auto()  # Job failed during training or evaluation


# DispatchType removed - dispatchers are passed directly to controller


@dataclass
class Observation:
    score: float
    cost: float
    suggestion: dict


@dataclass
class RunInfo:
    """Standardized run information returned by Store"""

    run_id: str
    group: str | None = None
    tags: list | None = None

    # Timestamps
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    last_heartbeat_at: datetime | None = None

    # Configuration and results
    summary: dict | None = None
    has_started_training: bool = False
    has_completed_training: bool = False
    has_started_eval: bool = False
    has_been_evaluated: bool = False
    has_failed: bool = False
    cost: float = 0
    runtime: float = 0

    # Sweep specific
    observation: Observation | None = None

    @property
    def status(self) -> JobStatus:
        if self.has_failed:
            return JobStatus.FAILED
        if not self.has_started_training:
            return JobStatus.PENDING
        if self.has_started_training and not self.has_completed_training:
            return JobStatus.IN_TRAINING
        if self.has_completed_training and not self.has_started_eval:
            return JobStatus.TRAINING_DONE_NO_EVAL
        if self.has_started_eval and not self.has_been_evaluated:
            return JobStatus.IN_EVAL
        if self.has_been_evaluated:
            if self.observation is not None:
                return JobStatus.COMPLETED
            return JobStatus.EVAL_DONE_NOT_COMPLETED
        return JobStatus.COMPLETED

    # Dispatch info
    # dispatch_id: str | None = None
    # dispatch_type: DispatchType | None = None


@dataclass
class JobResult:
    job: JobDefinition
    status: JobStatus
    metrics: dict[str, float]
    completed_at: datetime = field(default_factory=datetime.now)
    error: str | None = None


@dataclass
class SweepMetadata:
    """
    Metadata about a sweep stored in the Store.
    This is the persistent state that survives controller restarts.
    """

    sweep_id: str
    start_time: datetime = field(default_factory=datetime.now)
    last_scheduling: datetime = field(default_factory=datetime.now)
    runs_created: int = 0
    runs_pending: int = 0
    runs_in_progress: int = 0
    runs_completed: int = 0

    def to_metrics_dict(self) -> dict[str, Any]:
        """Convert to metrics dictionary for logging"""
        return {
            "runs_created": self.runs_created,
            "runs_completed": self.runs_completed,
            "runs_in_progress": self.runs_in_progress,
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
        }


# ============================================================================
# Protocol Definitions (all synchronous)
# ============================================================================


@runtime_checkable
class Scheduler(Protocol):
    """
    Implements sweep algorithms like ASHA, PBT, Bayesian optimization, etc.
    Decides which jobs to run, when to stop them, and how to adapt.
    Handles both training and evaluation job scheduling.
    """

    def schedule(self, sweep_metadata: SweepMetadata, all_runs: list[RunInfo]) -> list[JobDefinition]:
        """
        Decide which new jobs to create based on current state of all runs.
        This includes both new training jobs and evaluation jobs for completed training.
        """
        ...


@runtime_checkable
class Store(Protocol):
    """
    Single source of truth for all run and sweep state.
    All operations are synchronous with retry logic built in.
    """

    # Run operations
    def init_run(self, run_id: str, sweep_id: str | None = None) -> None:
        """Initialize a new run"""
        ...

    def fetch_runs(self, filters: dict) -> list[RunInfo]:
        """Fetch runs matching filter criteria, returns standardized RunInfo objects"""
        ...

    def update_run_summary(self, run_id, summary_update: dict) -> bool: ...


@runtime_checkable
class Dispatcher(Protocol):
    """
    Handles the mechanics of starting and monitoring jobs.
    All operations are synchronous with timeouts.
    Note: check_status and cancel_job implementations depend on dispatch type.
    """

    # Distinction: run_id is the job's identifier in WandB, dispatch_id is the Sky Job iD, the pid, etc...
    def dispatch(self, job: JobDefinition) -> str:
        """Start a job and return a dispatch ID"""
        ...


@runtime_checkable
class Optimizer(Protocol):
    """
    Suggests hyperparameters for new jobs.
    """

    def suggest(self, observations: list[Observation], n_suggestions: int = 1) -> list[dict[str, Any]]:
        """Suggest configurations for new jobs"""
        ...


# ============================================================================
# Retry Decorator
# ============================================================================


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator for retrying operations with exponential backoff"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 1
            current_delay = delay
            last_exception = None

            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_attempts:
                        logger.error(f"[Retry] Failed after {max_attempts} attempts: {e}")
                        raise
                    logger.warning(f"[Retry] Attempt {attempt} failed, retrying in {current_delay}s: {e}")
                    time.sleep(current_delay)
                    current_delay *= backoff
                    attempt += 1

            assert last_exception is not None
            raise last_exception

        return wrapper

    return decorator


# ============================================================================
# Local Dispatcher Implementation
# ============================================================================


class LocalDispatcher:
    """Runs jobs as local subprocesses."""

    def __init__(self):
        self._processes: dict[str, subprocess.Popen] = {}  # pid -> process
        self._run_to_pid: dict[str, str] = {}  # run_id -> pid for debugging

    def _reap_finished_processes(self):
        """Clean up finished subprocesses."""
        finished_pids = []
        for pid, process in self._processes.items():
            # poll() returns None if process is still running, returncode otherwise
            if process.poll() is not None:
                finished_pids.append(pid)
                logger.debug(f"[LocalDispatcher] Process {pid} finished with return code {process.returncode}")

        # Clean up finished processes
        for pid in finished_pids:
            del self._processes[pid]
            # Clean up run_id mapping
            run_id = next((rid for rid, p in self._run_to_pid.items() if p == pid), None)
            if run_id:
                del self._run_to_pid[run_id]

    def check_processes(self):
        """Check status of all processes."""
        self._reap_finished_processes()
        active_count = len(self._processes)
        if active_count > 0:
            logger.debug(f"[LocalDispatcher] Active subprocesses: {active_count}")
            for pid, process in self._processes.items():
                status = "running" if process.poll() is None else f"finished({process.returncode})"
                logger.debug(f"[LocalDispatcher]   PID {pid}: {status}")
        return active_count

    def dispatch(self, job: JobDefinition) -> str:
        """Dispatch job locally as subprocess."""

        # Reap any finished processes first to prevent zombie accumulation
        self._reap_finished_processes()

        # Build command
        cmd_parts = ["uv", "run", "./tools/run.py", job.cmd]

        # Add positional arguments first (if any)
        cmd_parts.extend(job.args)

        # Collect all args
        all_args = []

        # Add run_id for training jobs only (not for eval)
        if job.type == JobTypes.LAUNCH_TRAINING:
            all_args.append(f"run={job.run_id}")

        # Add metadata fields as args (used for evaluation jobs)
        for key, value in job.metadata.items():
            all_args.append(f"{key}={value}")

        # Add all args with --args flag
        if all_args:
            cmd_parts.append("--args")
            cmd_parts.extend(all_args)

        # Collect all overrides (from both overrides and config)
        all_overrides = []

        # Add explicit overrides
        for key, value in job.overrides.items():
            all_overrides.append(f"{key}={value}")

        # Add config from optimizer as additional overrides
        for key, value in job.config.items():
            all_overrides.append(f"{key}={value}")

        # Add all overrides with --overrides flag
        if all_overrides:
            cmd_parts.append("--overrides")
            cmd_parts.extend(all_overrides)

        # Extract trial portion for cleaner display
        display_id = job.run_id.split("_trial_")[-1] if "_trial_" in job.run_id else job.run_id
        display_id = f"trial_{display_id}" if not display_id.startswith("trial_") else display_id

        # Get job type name (e.g., "LAUNCH_TRAINING" -> "training")
        job_type_name = job.type.name

        logger.info(f"[LocalDispatcher] Dispatching local {job_type_name} for {display_id}: {' '.join(cmd_parts)}")

        try:
            # Start subprocess - optionally stream output for debugging
            # For production, use DEVNULL to avoid deadlock
            # For debugging, comment out the DEVNULL lines
            process = subprocess.Popen(
                cmd_parts,
                stdout=subprocess.DEVNULL,  # Comment out for debugging
                stderr=subprocess.DEVNULL,  # Comment out for debugging
                text=True,
            )

            # Use PID as the dispatch_id
            pid = str(process.pid)

            self._processes[pid] = process
            self._run_to_pid[job.run_id] = pid

            logger.info(f"[LocalDispatcher] Started {display_id} with PID {pid}")

            return pid

        except Exception as e:
            logger.error(f"[LocalDispatcher] Failed to start local run {job.run_id}: {e}")
            raise


# ============================================================================
# Stateless Sweep Controller
# ============================================================================


class SweepController:
    """Stateless orchestrator for sweep execution."""

    def __init__(
        self,
        sweep_id: str,
        scheduler: Scheduler,
        optimizer: Optimizer,
        dispatcher: Dispatcher,
        store: Store,
        protein_config: ProteinConfig,
        max_parallel_jobs: int = 10,
        monitoring_interval: int = 5,
    ):
        # Configuration only - no state
        self.sweep_id = sweep_id
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.dispatcher = dispatcher
        self.store = store
        self.protein_config = protein_config
        self.monitoring_interval = monitoring_interval
        self.max_parallel_jobs = max_parallel_jobs

    def _compute_metadata_from_runs(self, all_runs: list[RunInfo]) -> SweepMetadata:
        """Compute sweep metadata from runs."""
        metadata = SweepMetadata(sweep_id=self.sweep_id)
        metadata.runs_created = len(all_runs)  # Total number of runs

        for run in all_runs:
            if run.status == JobStatus.PENDING:
                metadata.runs_pending += 1
            elif run.status in [JobStatus.IN_TRAINING, JobStatus.TRAINING_DONE_NO_EVAL, JobStatus.IN_EVAL]:
                metadata.runs_in_progress += 1
            elif run.status in [JobStatus.COMPLETED, JobStatus.EVAL_DONE_NOT_COMPLETED]:
                metadata.runs_completed += 1

        return metadata

    def run(self) -> None:
        """Main control loop for sweep execution."""
        while True:
            try:
                # 1. Fetch ALL runs from store
                all_run_infos = self.store.fetch_runs(filters={"group": self.sweep_id})  # Returns list[RunInfo]

                # 2. Update sweep metadata based on ALL runs
                metadata = self._compute_metadata_from_runs(all_run_infos)

                # 3. Hand everything to scheduler - it decides what to do
                # Always get jobs from scheduler (including eval jobs)
                new_jobs = self.scheduler.schedule(sweep_metadata=metadata, all_runs=all_run_infos)

                # Check if the sweep is complete
                # TODO: We should modify scheduler to always have that attribute.
                if hasattr(self.scheduler, "is_complete") and self.scheduler.is_complete:  # type: ignore
                    logger.info("[SweepController]  Sweep completed successfully.")
                    break

                # Filter jobs based on capacity constraints
                # EVAL jobs always go through, TRAINING jobs respect the limit
                filtered_jobs = []
                training_jobs_count = 0

                for job in new_jobs:
                    if job.type == JobTypes.LAUNCH_EVAL:
                        # Always allow eval jobs
                        filtered_jobs.append(job)
                    elif job.type == JobTypes.LAUNCH_TRAINING:
                        # Check if we have capacity for training jobs
                        if metadata.runs_in_progress + training_jobs_count < self.max_parallel_jobs:
                            filtered_jobs.append(job)
                            training_jobs_count += 1
                        else:
                            logger.debug(
                                f"[SweepController] At max parallel jobs limit ({self.max_parallel_jobs}), "
                                f"skipping training job {job.run_id}"
                            )
                    else:
                        # Other job types (if any) go through
                        filtered_jobs.append(job)

                new_jobs = filtered_jobs

                # 4. Execute scheduler's decisions
                for job in new_jobs:
                    try:
                        if job.type == JobTypes.LAUNCH_TRAINING:
                            self.store.init_run(job.run_id, sweep_id=self.sweep_id)
                            # Store the suggestion (hyperparameters) in the run summary
                            if job.config:  # job.config contains the optimizer suggestion
                                self.store.update_run_summary(job.run_id, {"suggestion": job.config})
                            logger.info(f"[SweepController] Created run {job.run_id}")
                        elif job.type == JobTypes.LAUNCH_EVAL:
                            success = self.store.update_run_summary(
                                job.run_id,
                                {
                                    "has_started_eval": True,  # other status properties can be deduced from this.
                                },
                            )
                            if not success:
                                logger.error(
                                    f"[SweepController] Failed to update run summary for eval job {job.run_id}, "
                                    "skipping dispatch"
                                )
                                raise RuntimeError(f"Failed to update run summary for eval job {job.run_id}")
                            logger.info(f"[SweepController] Launching eval for job {job.run_id}")

                        # Only dispatch if store operations succeeded
                        dispatch_id = self.dispatcher.dispatch(job)
                        logger.info(f"[SweepController] Dispatched {job.run_id} with dispatch_id {dispatch_id}")

                    except Exception as e:
                        logger.error(f"[SweepController] Failed to initialize/dispatch job {job.run_id}: {e}")
                        logger.error(
                            f"[SweepController] Skipping dispatch for {job.run_id} to prevent resource overload"
                        )
                        # Continue with next job rather than crashing the whole sweep
                        continue

                # 5. Finally, update transient states and mark completions
                # TODO: Refactor: Sweep config and Optimizer config
                for run in all_run_infos:
                    if run.status == JobStatus.EVAL_DONE_NOT_COMPLETED:
                        assert run.summary is not None
                        cost = run.cost if run.cost != 0 else run.runtime
                        score = run.summary.get(self.protein_config.metric)
                        if score is None:
                            raise ValueError(f"No metric {self.protein_config.metric} found in run summary.")
                        self.store.update_run_summary(
                            run.run_id,
                            {
                                "observation": {
                                    "cost": cost,
                                    "score": score,
                                    "suggestion": run.summary.get("suggestion"),
                                }
                            },
                        )

                # 5. Sleep
                time.sleep(self.monitoring_interval)

            except KeyboardInterrupt:
                logger.info("[SweepController] Received interrupt signal, stopping sweep")
                break
            except Exception as e:
                logger.error(f"[SweepController] Error in control loop: {e}")
                time.sleep(self.monitoring_interval)


# ============================================================================
# Main Orchestration Entry Point
# ============================================================================


@dataclass
class SweepOrchestratorConfig:
    sweep_name: str
    sweep_server_uri: str
    wandb: WandbConfig
    protein_config: ProteinConfig
    max_parallel_jobs: int = 10
    monitoring_interval: int = 60


def orchestrate_sweep(
    config: SweepOrchestratorConfig,
    scheduler: Scheduler,
    optimizer: Optimizer,
    dispatcher: Dispatcher,
    store: Store,
) -> None:
    """Entry point for running a sweep."""
    cogweb_client = CogwebClient.get_client(base_url=config.sweep_server_uri)
    sweep_client = cogweb_client.sweep_client()

    sweep_info = sweep_client.get_sweep(config.sweep_name)
    if not sweep_info.exists:
        logger.info(f"[Orchestrator] Registering sweep {config.sweep_name}")
        sweep_client.create_sweep(config.sweep_name, config.wandb.project, config.wandb.entity, config.sweep_name)

    # Create the sweep controller (stateless)
    controller = SweepController(
        sweep_id=config.sweep_name,
        scheduler=scheduler,
        optimizer=optimizer,
        dispatcher=dispatcher,
        store=store,
        protein_config=config.protein_config,
        max_parallel_jobs=config.max_parallel_jobs,
        monitoring_interval=config.monitoring_interval,
    )

    try:
        controller.run()
    finally:
        logger.info("[Orchestrator] Bye")
