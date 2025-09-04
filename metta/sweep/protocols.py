"""Protocol definitions for sweep orchestration components."""

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from metta.sweep.models import JobDefinition, Observation, RunInfo, SweepMetadata


@runtime_checkable
class Scheduler(Protocol):
    """
    Implements sweep algorithms like ASHA, PBT, Bayesian optimization, etc.
    Decides which jobs to run, when to stop them, and how to adapt.
    Handles both training and evaluation job scheduling.
    """

    def schedule(
        self,
        sweep_metadata: "SweepMetadata",
        all_runs: list["RunInfo"],
        dispatched_trainings: set[str],
        dispatched_evals: set[str],
    ) -> list["JobDefinition"]:
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

    def fetch_runs(self, filters: dict) -> list["RunInfo"]:
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
    def dispatch(self, job: "JobDefinition") -> str:
        """Start a job and return a dispatch ID"""
        ...


@runtime_checkable
class Optimizer(Protocol):
    """
    Suggests hyperparameters for new jobs.
    """

    def suggest(self, observations: list["Observation"], n_suggestions: int = 1) -> list[dict[str, Any]]:
        """Suggest configurations for new jobs"""
        ...
