"""Simplified protocols for adaptive experiments."""

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .models import JobDefinition, RunInfo


@runtime_checkable
class ExperimentScheduler(Protocol):
    """
    Simple scheduler protocol - gets runs, returns jobs to dispatch.

    The scheduler contains all experiment logic. Examples:
    - HyperparameterScheduler: Bayesian optimization
    - ValidationScheduler: Multi-seed validation
    - AblationScheduler: Component ablation study
    """

    def schedule(self, runs: list["RunInfo"], available_training_slots: int) -> list["JobDefinition"]:
        """
        Decide which jobs to dispatch next based on current run state and available resources.

        Args:
            runs: All runs in the experiment (completed, running, failed)
            available_training_slots: How many LAUNCH_TRAINING jobs can be dispatched right now

        Returns:
            Jobs to dispatch. LAUNCH_TRAINING jobs should not exceed available_training_slots.
            LAUNCH_EVAL jobs don't count against the limit and can always be dispatched.
        """
        ...

    def is_experiment_complete(self, runs: list["RunInfo"]) -> bool:
        """
        Check if the experiment is finished and no more work will be scheduled.

        Args:
            runs: All runs in the experiment (completed, running, failed)

        Returns:
            True if experiment is complete and controller should terminate
        """
        ...


@runtime_checkable
class Store(Protocol):
    """
    Single source of truth for all run and experiment state.
    All operations are synchronous with retry logic built in.
    """

    # Run operations
    def init_run(
        self,
        run_id: str,
        group: str | None = None,
        tags: list[str] | None = None,
        initial_summary: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a new run with optional initial summary data"""
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
    """

    # Distinction: run_id is the job's identifier in WandB, dispatch_id is the Sky Job iD, the pid, etc...
    def dispatch(self, job: "JobDefinition") -> str:
        """Start a job and return a dispatch ID"""
        ...


@runtime_checkable
class Optimizer(Protocol):
    """Suggests hyperparameters for new jobs."""

    def suggest(self, observations: list[dict[str, Any]], n_suggestions: int = 1) -> list[dict[str, Any]]:
        """Suggest configurations for new jobs"""
        ...


@runtime_checkable
class SchedulerConfig(Protocol):
    """Protocol for scheduler configuration objects expected by AdaptiveTool.

    Must be serializable; at minimum provide a model_dump() -> dict interface
    (e.g., Pydantic models). Dataclasses are also acceptable if converted prior
    to passing into the tool.
    """

    def model_dump(self) -> dict[str, Any]:  # pragma: no cover - protocol only
        ...


@runtime_checkable
class ExperimentState(Protocol):
    """Optional typed state object for experiments (serializable, Pydantic-like).

    Experiments that benefit from persistent or shared state (e.g., learning
    progress, advanced optimizers) can define a dedicated state model.
    """

    def model_dump(self) -> dict[str, Any]:  # pragma: no cover - protocol only
        ...

    @classmethod
    def model_validate(cls, data: dict[str, Any]) -> "ExperimentState":  # pragma: no cover - protocol only
        ...
