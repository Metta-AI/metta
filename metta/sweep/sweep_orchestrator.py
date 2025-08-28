from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, Any, runtime_checkable
from enum import StrEnum, auto
import time
import logging
import subprocess
import uuid

from cogweb.cogweb_client import CogwebClient

from metta.common.wandb.wandb_context import WandbConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Core Data Models
# ============================================================================


@dataclass
class JobDefinition:
    run_id: str
    cmd: str  # e.g., "experiments.recipes.arena.train_shaped" or "experiments.recipes.arena.evaluate"
    gpus: int = 1
    nodes: int = 1
    args: list[str] = field(default_factory=list)  # positional arguments
    overrides: dict[str, Any] = field(default_factory=dict)  # key=value overrides for the tool
    config: dict[str, Any] = field(default_factory=dict)  # additional config from optimizer
    type: str = "train"  # "train" or "eval"
    parent_job_id: str | None = None  # For eval jobs, the training job that produced the policy
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


class JobStatus(StrEnum):
    PENDING = auto()
    IN_TRAINING = auto()
    IN_EVAL = auto()
    COMPLETED = auto()


class DispatchType(StrEnum):
    LOCAL = auto()
    SKYPILOT = auto()
    CENTRAL_QUEUE = auto()


@dataclass
class Observation:
    score: float
    cost: float
    suggestion: dict


@dataclass
class RunInfo:
    """Standardized run information returned by Store"""

    run_id: str
    sweep_id: str
    status: JobStatus

    # Timestamps
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    last_heartbeat_at: datetime | None = None

    # Configuration and results
    summary: dict = {}
    has_started_training: bool = False
    has_completed_training: bool = False
    has_been_evaluated: bool = False

    # Sweep specific
    observation: Observation | None = None
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

    # Sweep metadata operations
    def fetch_sweep_metadata(self, sweep_id: str) -> SweepMetadata | None:
        """Fetch sweep metadata, returns None if sweep doesn't exist"""
        ...

    # Run operations
    def init_run(self, job: JobDefinition) -> None:
        """Initialize a new run"""
        ...

    def fetch_runs_by_status(self, status: JobStatus) -> list[RunInfo]:
        """Fetch all runs with a given status"""
        ...

    def fetch_runs(self, filter: dict) -> list[RunInfo]:
        """Fetch runs matching filter criteria, returns standardized RunInfo objects"""
        ...

    def update_run_summary(self, run_id, summary: dict) -> bool: ...
    def get_dispatch_id(self, run_id: str) -> str | None:
        """Get the dispatch ID for a run"""
        ...


@runtime_checkable
class Dispatcher(Protocol):
    """
    Handles the mechanics of starting and monitoring jobs.
    All operations are synchronous with timeouts.
    Note: check_status and cancel_job implementations depend on dispatch type.
    """

    # Distinction: run_id is the job's identifier in WandB, dispatch_id is the Sky Job iD, the pid, etc...
    def dispatch(self, job: JobDefinition, dispatch_type: DispatchType) -> str:
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
                        logger.error(f"Failed after {max_attempts} attempts: {e}")
                        raise
                    logger.warning(f"Attempt {attempt} failed, retrying in {current_delay}s: {e}")
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
    """
    Local dispatcher that runs jobs as subprocesses.
    All operations are synchronous.
    Uses process PID as the dispatch_id for natural process management.
    """

    def __init__(self):
        self._processes: dict[str, subprocess.Popen] = {}  # pid -> process
        self._run_to_pid: dict[str, str] = {}  # run_id -> pid for debugging

    def dispatch(self, job: JobDefinition, dispatch_type: DispatchType) -> str:
        """Dispatch a job locally as a subprocess and return its PID as dispatch_id"""
        if dispatch_type != DispatchType.LOCAL:
            raise ValueError(f"LocalDispatcher only supports LOCAL dispatch, got {dispatch_type}")

        # Build command
        cmd_parts = ["uv", "run", "./tools/run.py", job.cmd]

        # Add positional arguments
        cmd_parts.extend(job.args)

        # Add overrides
        for key, value in job.overrides.items():
            cmd_parts.append(f"{key}={value}")

        # Add config from optimizer as additional overrides
        for key, value in job.config.items():
            cmd_parts.append(f"{key}={value}")

        # Add run ID
        cmd_parts.append(f"run={job.run_id}")

        logger.info(f"Dispatching local run {job.run_id}: {' '.join(cmd_parts)}")

        try:
            # Start subprocess
            process = subprocess.Popen(
                cmd_parts,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Use PID as the dispatch_id
            pid = str(process.pid)

            self._processes[pid] = process
            self._run_to_pid[job.run_id] = pid

            logger.info(f"Started run {job.run_id} with PID {pid}")

            return pid

        except Exception as e:
            logger.error(f"Failed to start local run {job.run_id}: {e}")
            raise


# ============================================================================
# Stateless Sweep Controller
# ============================================================================


class SweepController:
    """
    Stateless orchestrator for sweeps. All state is stored in the Store.
    Controller can be restarted at any time without losing progress.
    """

    def __init__(
        self,
        sweep_id: str,
        scheduler: Scheduler,
        optimizer: Optimizer,
        dispatcher: Dispatcher,
        store: Store,
        max_parallel_jobs: int = 10,
        dispatch_type: DispatchType = DispatchType.SKYPILOT,
        scheduling_interval: int = 30,
        monitoring_interval: int = 5,
    ):
        # Configuration only - no state
        self.sweep_id = sweep_id
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.dispatcher = dispatcher
        self.store = store
        self.dispatch_type = dispatch_type
        self.scheduling_interval = scheduling_interval
        self.monitoring_interval = monitoring_interval

        # For local dispatch, enforce max_parallel_jobs = 1
        if dispatch_type == DispatchType.LOCAL:
            if max_parallel_jobs != 1:
                logger.warning(f"Local dispatch requires max_parallel_jobs=1, overriding {max_parallel_jobs}")
            self.max_parallel_jobs = 1
        else:
            self.max_parallel_jobs = max_parallel_jobs

    # ========== Lifecycle Methods ==========

    def _compute_metadata_from_runs(self, all_runs: list[RunInfo]) -> SweepMetadata:
        """Compute sweep metadata from all runs"""
        metadata = SweepMetadata(sweep_id=self.sweep_id)

        for run in all_runs:
            if run.status == JobStatus.COMPLETED:
                metadata.runs_completed += 1
            elif run.status in [JobStatus.PENDING, JobStatus.IN_TRAINING, JobStatus.IN_EVAL]:
                metadata.runs_in_progress += 1

        return metadata

    def run(self) -> None:
        """Main control loop - simplified"""
        while True:
            try:
                # 1. Fetch ALL runs from store
                all_run_infos = self.store.fetch_runs(filter={"sweep_id": self.sweep_id})  # Returns list[RunInfo]

                # 2. Update sweep metadata based on ALL runs
                metadata = self._compute_metadata_from_runs(all_run_infos)

                # 3. Hand everything to scheduler - it decides what to do
                new_jobs = self.scheduler.schedule(sweep_metadata=metadata, all_runs=all_run_infos)

                # 4. Execute scheduler's decisions
                for job in new_jobs:
                    self.store.init_run(job)
                    logger.info(f"Created run {job.run_id}")
                    dispatch_id = self.dispatcher.dispatch(job, self.dispatch_type)
                    logger.info(f"Dispatched {job.run_id} with PID {dispatch_id}")

                # 5. Sleep
                time.sleep(self.monitoring_interval)

            except KeyboardInterrupt:
                logger.info("Received interrupt signal, stopping sweep")
                break
            except Exception as e:
                logger.error(f"Error in control loop: {e}")
                time.sleep(self.monitoring_interval)


# ============================================================================
# Main Orchestration Entry Point
# ============================================================================


@dataclass
class SweepOrchestratorConfig:
    sweep_name: str
    sweep_server_uri: str
    wandb: WandbConfig
    max_parallel_jobs: int = 10
    dispatch_type: DispatchType = DispatchType.SKYPILOT
    scheduling_interval: int = 30
    monitoring_interval: int = 5


def orchestrate_sweep(
    config: SweepOrchestratorConfig,
    scheduler: Scheduler,
    optimizer: Optimizer,
    dispatcher: Dispatcher,
    store: Store,
) -> None:
    """
    Entry point for running a sweep. Creates one controller for one sweep.

    The controller is completely stateless - all state is in the Store.
    This means the controller can be killed and restarted at any point
    without losing progress.
    """
    cogweb_client = CogwebClient.get_client(base_url=config.sweep_server_uri)
    sweep_client = cogweb_client.sweep_client()

    sweep_info = sweep_client.get_sweep(config.sweep_name)
    if not sweep_info.exists:
        logger.info(f"Registering sweep {config.sweep_name}")
        sweep_client.create_sweep(config.sweep_name, config.wandb.project, config.wandb.entity, config.sweep_name)

    # Create the sweep controller (stateless)
    controller = SweepController(
        sweep_id=config.sweep_name,
        scheduler=scheduler,
        optimizer=optimizer,
        dispatcher=dispatcher,
        store=store,
        max_parallel_jobs=config.max_parallel_jobs,
        dispatch_type=config.dispatch_type,
        scheduling_interval=config.scheduling_interval,
        monitoring_interval=config.monitoring_interval,
    )

    try:
        controller.run()
    finally:
        logger.info("Bye")
