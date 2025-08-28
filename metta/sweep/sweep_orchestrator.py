from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, Any, runtime_checkable
from enum import StrEnum, auto
import time
import logging
import subprocess
import uuid

from cogweb.cogweb_client import CogwebClient
from pydantic import ConfigDict

from metta.common.wandb.wandb_context import WandbConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Core Data Models
# ============================================================================

@dataclass
class JobDefinition:
    run_id: str
    cmd: str  # e.g., "experiments.recipes.arena.train_shaped" or "experiments.recipes.arena.evaluate"
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
    SCHEDULED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class JobResult:
    job: JobDefinition
    status: JobStatus
    metrics: dict[str, float]
    completed_at: datetime = field(default_factory=datetime.now)
    error: str | None = None


class DispatchType(StrEnum):
    LOCAL = auto()
    SKYPILOT = auto()
    CENTRAL_QUEUE = auto()


@dataclass
class SweepMetadata:
    """
    Metadata about a sweep stored in the Store.
    This is the persistent state that survives controller restarts.
    """
    sweep_id: str
    phase: str = "initializing"
    running: bool = False
    start_time: datetime = field(default_factory=datetime.now)
    last_scheduling: datetime = field(default_factory=datetime.now)
    jobs_created: int = 0
    jobs_completed: int = 0
    jobs_failed: int = 0
    jobs_cancelled: int = 0
    eval_jobs_created: int = 0
    eval_jobs_completed: int = 0
    eval_jobs_failed: int = 0
    
    def to_metrics_dict(self) -> dict[str, Any]:
        """Convert to metrics dictionary for logging"""
        return {
            "phase": self.phase,
            "running": self.running,
            "jobs_created": self.jobs_created,
            "jobs_completed": self.jobs_completed,
            "jobs_failed": self.jobs_failed,
            "jobs_cancelled": self.jobs_cancelled,
            "eval_jobs_created": self.eval_jobs_created,
            "eval_jobs_completed": self.eval_jobs_completed,
            "eval_jobs_failed": self.eval_jobs_failed,
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
    """
    def initialize(self, sweep_id: str) -> list[JobDefinition]:
        """Generate initial jobs for warmup phase"""
        ...
    
    def schedule(self, sweep_metadata: SweepMetadata, observations: list[JobResult]) -> list[JobDefinition]:
        """Decide which new jobs to create based on observations"""
        ...
    
    def schedule_evaluations(self, jobs_needing_eval: list[JobDefinition]) -> list[JobDefinition]:
        """
        Create evaluation jobs for completed training jobs.
        Returns a list of eval JobDefinitions with appropriate cmd, args, and parent_job_id.
        """
        ...
    
    def should_stop_job(self, job: JobDefinition, current_metrics: dict[str, float]) -> bool:
        """Decide if a running job should be stopped early (e.g., ASHA pruning)"""
        ...


@runtime_checkable
class Store(Protocol):
    """
    Single source of truth for all job and sweep state.
    All operations are synchronous with retry logic built in.
    """
    # Sweep metadata operations
    def fetch_sweep_metadata(self, sweep_id: str) -> SweepMetadata | None:
        """Fetch sweep metadata, returns None if sweep doesn't exist"""
        ...
    
    def store_sweep_metadata(self, metadata: SweepMetadata) -> None:
        """Store or update sweep metadata"""
        ...
    
    def update_sweep_phase(self, sweep_id: str, phase: str) -> None:
        """Update just the phase of a sweep"""
        ...
    
    def update_sweep_running(self, sweep_id: str, running: bool) -> None:
        """Update the running flag of a sweep"""
        ...
    
    def increment_sweep_counter(self, sweep_id: str, counter: str, amount: int = 1) -> None:
        """Increment a counter (jobs_created, jobs_completed, etc.)"""
        ...
    
    def update_last_scheduling(self, sweep_id: str, timestamp: datetime) -> None:
        """Update the last scheduling timestamp"""
        ...
    
    # Job operations
    def fetch_jobs_by_status(self, status: JobStatus) -> list[JobDefinition]:
        """Fetch all jobs with a given status"""
        ...
    
    def fetch_job_by_id(self, job_id: str) -> JobDefinition | None:
        """Fetch a specific job by ID"""
        ...
    
    def fetch_jobs_needing_evaluation(self) -> list[JobDefinition]:
        """Fetch completed training jobs that haven't been evaluated yet"""
        ...
    
    def has_evaluation_results(self, job_id: str) -> bool:
        """Check if a job has evaluation results (metrics with 'evaluator/' prefix)"""
        ...
    
    def store_job(self, job: JobDefinition) -> None:
        """Store a new job"""
        ...

    def update_job_status(self, job_id: str, status: JobStatus, dispatch_id: str | None = None) -> None:
        """Update job status and optionally store dispatch ID"""
        ...
    
    def get_dispatch_id(self, job_id: str) -> str | None:
        """Get the dispatch ID for a job"""
        ...

    def fetch_job_results(self, limit: int = 100) -> list[JobResult]:
        """Fetch completed job results"""
        ...
    
    def fetch_job_metrics(self, job_id: str) -> dict[str, float] | None:
        """Fetch latest metrics for a job (used for early stopping decisions)"""
        ...
    
    def store_job_metrics(self, job_id: str, metrics: dict[str, float]) -> None:
        """Store intermediate metrics for a running job"""
        ...
    
    def store_job_result(self, result: JobResult) -> None:
        """Store final job result"""
        ...


@runtime_checkable
class Dispatcher(Protocol):
    """
    Handles the mechanics of starting and monitoring jobs.
    All operations are synchronous with timeouts.
    Note: check_status and cancel_job implementations depend on dispatch type.
    """
    def dispatch(self, job: JobDefinition, dispatch_type: DispatchType) -> str:
        """Start a job and return a dispatch ID"""
        ...

    def check_status(self, dispatch_id: str) -> JobStatus:
        """
        Check if a job is still running.
        Implementation varies by dispatch type - may raise NotImplementedError.
        """
        ...
    
    def cancel_job(self, dispatch_id: str, timeout: float = 10.0) -> None:
        """
        Stop a running job with timeout.
        Implementation varies by dispatch type - may raise NotImplementedError.
        """
        ...


@runtime_checkable
class Optimizer(Protocol):
    """
    Suggests hyperparameters for new jobs.
    """
    def suggest(self, observations: list[JobResult], n_suggestions: int = 1) -> list[dict[str, Any]]:
        """Suggest configurations for new jobs"""
        ...

    def update(self, results: list[JobResult]) -> None:
        """Update optimizer state with new results"""
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
    """
    
    def __init__(self):
        self._processes: dict[str, subprocess.Popen] = {}
        self._job_info: dict[str, JobDefinition] = {}
    
    def dispatch(self, job: JobDefinition, dispatch_type: DispatchType) -> str:
        """Dispatch a job locally as a subprocess"""
        if dispatch_type != DispatchType.LOCAL:
            raise ValueError(f"LocalDispatcher only supports LOCAL dispatch, got {dispatch_type}")
        
        # Generate unique dispatch ID
        dispatch_id = f"local_{job.run_id}_{uuid.uuid4().hex[:8]}"
        
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
        
        logger.info(f"Dispatching local job {job.run_id}: {' '.join(cmd_parts)}")
        
        try:
            # Start subprocess
            process = subprocess.Popen(
                cmd_parts,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            
            self._processes[dispatch_id] = process
            self._job_info[dispatch_id] = job
            
            return dispatch_id
            
        except Exception as e:
            logger.error(f"Failed to start local job {job.run_id}: {e}")
            raise
    
    def check_status(self, dispatch_id: str) -> JobStatus:
        """Check the status of a dispatched job"""
        if dispatch_id not in self._processes:
            return JobStatus.FAILED
        
        process = self._processes[dispatch_id]
        poll_result = process.poll()
        
        if poll_result is None:
            # Process is still running
            return JobStatus.RUNNING
        elif poll_result == 0:
            # Process completed successfully
            return JobStatus.COMPLETED
        else:
            # Process failed
            logger.error(f"Job {dispatch_id} failed with exit code {poll_result}")
            return JobStatus.FAILED
    
    def cancel_job(self, dispatch_id: str, timeout: float = 10.0) -> None:
        """Cancel a running job with timeout"""
        if dispatch_id not in self._processes:
            return
        
        process = self._processes[dispatch_id]
        if process.poll() is None:
            logger.info(f"Terminating job {dispatch_id}")
            process.terminate()
            
            # Wait for graceful termination
            start_time = time.time()
            while process.poll() is None and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            # Force kill if still running
            if process.poll() is None:
                logger.warning(f"Force killing job {dispatch_id}")
                process.kill()
                process.wait()  # Wait for kill to complete


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

    def setup(self) -> None:
        """Initialize sweep metadata in store"""
        metadata = self.store.fetch_sweep_metadata(self.sweep_id)
        if metadata is None:
            # Create new sweep metadata
            metadata = SweepMetadata(
                sweep_id=self.sweep_id,
                phase="setup",
                running=False,
            )
            self.store.store_sweep_metadata(metadata)
        else:
            # Update existing sweep
            self.store.update_sweep_phase(self.sweep_id, "setup")
        
        logger.info(f"Setting up sweep {self.sweep_id}")
        
    def warmup(self) -> None:
        """Create initial jobs for warmup phase"""
        self.store.update_sweep_phase(self.sweep_id, "warmup")
        
        # Get initial jobs from scheduler
        initial_jobs = self.scheduler.initialize(self.sweep_id)
        
        # Get optimizer suggestions for the jobs
        if initial_jobs:
            configs = self.optimizer.suggest([], n_suggestions=len(initial_jobs))
            for job, config in zip(initial_jobs, configs):
                job.config.update(config)
                self.store.store_job(job)
                self.store.update_job_status(job.run_id, JobStatus.PENDING)
                self.store.increment_sweep_counter(self.sweep_id, "jobs_created")
        
        logger.info(f"Created {len(initial_jobs)} warmup jobs")

    @retry(max_attempts=3, delay=1.0)
    def _fetch_state(self) -> tuple[SweepMetadata, list[JobDefinition], list[JobDefinition]]:
        """Fetch current state from store with retries"""
        metadata = self.store.fetch_sweep_metadata(self.sweep_id)
        if metadata is None:
            raise RuntimeError(f"Sweep {self.sweep_id} metadata not found")
        
        pending_jobs = self.store.fetch_jobs_by_status(JobStatus.PENDING)
        running_jobs = self.store.fetch_jobs_by_status(JobStatus.RUNNING)
        return metadata, pending_jobs, running_jobs

    @retry(max_attempts=3, delay=2.0)
    def _dispatch_job(self, job: JobDefinition) -> str:
        """Dispatch a job with retries"""
        return self.dispatcher.dispatch(job, self.dispatch_type)

    def run(self) -> None:
        """Main control loop - completely stateless"""
        self.store.update_sweep_phase(self.sweep_id, "running")
        self.store.update_sweep_running(self.sweep_id, True)
        
        while True:
            try:
                # Fetch all state from store
                metadata, pending_jobs, running_jobs = self._fetch_state()
                
                # Check if we should stop
                if not metadata.running:
                    logger.info(f"Sweep {self.sweep_id} stopped by external signal")
                    break
                
                # Dispatch pending jobs if we have capacity
                while len(running_jobs) < self.max_parallel_jobs and pending_jobs:
                    job = pending_jobs.pop(0)
                    try:
                        dispatch_id = self._dispatch_job(job)
                        self.store.update_job_status(job.run_id, JobStatus.RUNNING, dispatch_id)
                        running_jobs.append(job)
                        logger.info(f"Dispatched job {job.run_id}")
                    except Exception as e:
                        logger.error(f"Failed to dispatch job {job.run_id}: {e}")
                        self.store.update_job_status(job.run_id, JobStatus.FAILED)
                        self.store.increment_sweep_counter(self.sweep_id, "jobs_failed")
                
                # Check status of running jobs (only for dispatch types that support it)
                if self.dispatch_type == DispatchType.LOCAL:
                    for job in running_jobs:
                        dispatch_id = self.store.get_dispatch_id(job.run_id)
                        if dispatch_id:
                            status = self.dispatcher.check_status(dispatch_id)
                            
                            if status == JobStatus.COMPLETED:
                                metrics = self.store.fetch_job_metrics(job.run_id) or {}
                                result = JobResult(job=job, status=status, metrics=metrics)
                                self.store.store_job_result(result)
                                self.store.update_job_status(job.run_id, JobStatus.COMPLETED)
                                
                                # Update appropriate counter based on job type
                                if job.type == "eval":
                                    self.store.increment_sweep_counter(self.sweep_id, "eval_jobs_completed")
                                    logger.info(f"Eval job {job.run_id} completed")
                                else:
                                    self.store.increment_sweep_counter(self.sweep_id, "jobs_completed")
                                    logger.info(f"Job {job.run_id} completed")
                                
                            elif status == JobStatus.FAILED:
                                result = JobResult(job=job, status=status, metrics={})
                                self.store.store_job_result(result)
                                self.store.update_job_status(job.run_id, JobStatus.FAILED)
                                
                                # Update appropriate counter based on job type
                                if job.type == "eval":
                                    self.store.increment_sweep_counter(self.sweep_id, "eval_jobs_failed")
                                    logger.error(f"Eval job {job.run_id} failed")
                                else:
                                    self.store.increment_sweep_counter(self.sweep_id, "jobs_failed")
                                    logger.error(f"Job {job.run_id} failed")
                            
                            # Check if scheduler wants to stop this job (ASHA early stopping)
                            # Only supported for LOCAL dispatch currently
                            elif status == JobStatus.RUNNING:
                                metrics = self.store.fetch_job_metrics(job.run_id)
                                if metrics and self.scheduler.should_stop_job(job, metrics):
                                    self.dispatcher.cancel_job(dispatch_id)
                                    self.store.update_job_status(job.run_id, JobStatus.CANCELLED)
                                    self.store.increment_sweep_counter(self.sweep_id, "jobs_cancelled")
                                    logger.info(f"Stopped job {job.run_id} (early stopping)")
                else:
                    # For non-local dispatch types, we need different status checking mechanisms
                    # (e.g., polling SkyPilot API, checking queue status, etc.)
                    # For now, just log that we're not checking status
                    if len(running_jobs) > 0:
                        logger.debug(f"Status checking not implemented for {self.dispatch_type} dispatch")
                
                # Scheduling: create new jobs periodically
                time_since_scheduling = (datetime.now() - metadata.last_scheduling).total_seconds()
                if time_since_scheduling >= self.scheduling_interval:
                    # Get completed results and make scheduling decision
                    results = self.store.fetch_job_results(limit=100)
                    new_jobs = self.scheduler.schedule(metadata, results)
                    
                    if new_jobs:
                        # Update optimizer with results and get suggestions
                        self.optimizer.update(results)
                        configs = self.optimizer.suggest(results, n_suggestions=len(new_jobs))
                        
                        for job, config in zip(new_jobs, configs):
                            job.config.update(config)
                            self.store.store_job(job)
                            self.store.update_job_status(job.run_id, JobStatus.PENDING)
                            self.store.increment_sweep_counter(self.sweep_id, "jobs_created")
                        
                        logger.info(f"Scheduled {len(new_jobs)} new jobs")
                    
                    # Schedule evaluations for completed training jobs without eval results
                    jobs_needing_eval = self.store.fetch_jobs_needing_evaluation()
                    if jobs_needing_eval:
                        eval_jobs = self.scheduler.schedule_evaluations(jobs_needing_eval)
                        
                        for eval_job in eval_jobs:
                            # Eval jobs don't need optimizer configs
                            self.store.store_job(eval_job)
                            self.store.update_job_status(eval_job.run_id, JobStatus.PENDING)
                            self.store.increment_sweep_counter(self.sweep_id, "eval_jobs_created")
                        
                        if eval_jobs:
                            logger.info(f"Scheduled {len(eval_jobs)} evaluation jobs")
                    
                    self.store.update_last_scheduling(self.sweep_id, datetime.now())
                    
                    # Log sweep metrics periodically
                    metrics_data = metadata.to_metrics_dict()
                    logger.info(f"Sweep {self.sweep_id} metrics: {metrics_data}")
                
                # Sleep before next iteration
                time.sleep(self.monitoring_interval)
                
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, stopping sweep")
                self.store.update_sweep_running(self.sweep_id, False)
                break
            except Exception as e:
                logger.error(f"Error in control loop: {e}")
                time.sleep(self.monitoring_interval)

    def cooldown(self) -> None:
        """Wait for running jobs to complete or timeout"""
        self.store.update_sweep_phase(self.sweep_id, "cooldown")
        self.store.update_sweep_running(self.sweep_id, False)
        
        # Wait for remaining jobs to complete or timeout
        timeout = 60
        start = datetime.now()
        
        while (datetime.now() - start).total_seconds() < timeout:
            running_jobs = self.store.fetch_jobs_by_status(JobStatus.RUNNING)
            if not running_jobs:
                break
                
            logger.info(f"Waiting for {len(running_jobs)} jobs to complete...")
            time.sleep(5)
        
        # Cancel any remaining jobs (only supported for LOCAL dispatch)
        if self.dispatch_type == DispatchType.LOCAL:
            running_jobs = self.store.fetch_jobs_by_status(JobStatus.RUNNING)
            for job in running_jobs:
                dispatch_id = self.store.get_dispatch_id(job.run_id)
                if dispatch_id:
                    self.dispatcher.cancel_job(dispatch_id)
                    self.store.update_job_status(job.run_id, JobStatus.CANCELLED)
                    self.store.increment_sweep_counter(self.sweep_id, "jobs_cancelled")
                    logger.warning(f"Cancelled job {job.run_id} during cooldown")
        else:
            logger.warning(f"Job cancellation not implemented for {self.dispatch_type} dispatch")

    def teardown(self) -> None:
        """Final cleanup and reporting"""
        self.store.update_sweep_phase(self.sweep_id, "completed")
        
        metadata = self.store.fetch_sweep_metadata(self.sweep_id)
        if metadata:
            final_metrics = metadata.to_metrics_dict()
            logger.info(f"Sweep {self.sweep_id} completed: {final_metrics}")


# ============================================================================
# Main Orchestration Entry Point
# ============================================================================

class SweepOrchestratorConfig(ConfigDict):
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
        sweep_client.create_sweep(
            config.sweep_name,
            config.wandb.project,
            config.wandb.entity,
            config.sweep_name
        )
    
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
        controller.setup()
        controller.warmup()
        controller.run()
    finally:
        controller.cooldown()
        controller.teardown()