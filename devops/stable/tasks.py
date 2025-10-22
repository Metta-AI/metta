"""Defines the actual tasks that are run when validating a release."""

from __future__ import annotations

from datetime import datetime
from operator import ge, gt
from typing import Callable, Literal, Optional

from pydantic import BaseModel

from metta.common.util.text_styles import blue, cyan, green, magenta, red, yellow
from metta.jobs.models import JobConfig
from metta.jobs.state import JobState

# Type definitions
Outcome = Literal["passed", "failed", "skipped", "inconclusive"]
# Acceptance Rules are tuples of (metric_name, operator, expected_value), e.g. ("overview/sps", ge, 40000)
AcceptanceRule = tuple[str, Callable[[float, float], bool], float]


def _parse_args_list(args: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Parse args list like ["run=test", "trainer.total_timesteps=100"] into args and overrides.

    Args that contain '.' are treated as overrides, others as args.

    Returns:
        (args_dict, overrides_dict)
    """
    args_dict = {}
    overrides_dict = {}

    for arg in args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            if "." in key:
                overrides_dict[key] = value
            else:
                args_dict[key] = value
        else:
            # Positional arg without =, skip for now
            pass

    return args_dict, overrides_dict


def _evaluate_thresholds(metrics: dict[str, float], checks: list[AcceptanceRule]) -> tuple[Outcome, list[str]]:
    """Evaluate metrics against acceptance rules."""
    failures: list[str] = []
    for key, op, expected in checks:
        if key not in metrics:
            failures.append(f"{key}: metric missing (expected vs {expected})")
            continue
        if not op(metrics[key], expected):
            failures.append(f"{key}: expected {op.__name__} {expected}, saw {metrics[key]}")
    return ("passed" if not failures else "failed", failures)


class TaskResult(BaseModel):
    """Result of running a validation task.

    Artifacts provide explicit data flow between tasks:
    - checkpoint_uri: Location of trained model checkpoint
    - wandb_run_id: WandB run ID for metrics tracking
    """

    name: str
    started_at: str
    ended_at: str
    outcome: Outcome  # "passed", "failed", "skipped"
    exit_code: int = 0
    metrics: dict[str, float] = {}
    artifacts: dict[str, str] = {}  # checkpoint_uri, wandb_run_id, etc.
    logs_path: Optional[str] = None
    job_id: Optional[str] = None
    error: Optional[str] = None

    def display_detailed(self) -> None:
        """Print detailed verification information with highlighted artifacts."""

        print(f"\n{'=' * 80}")
        print(blue(f"📋 TASK VERIFICATION: {self.name}"))
        print(f"{'=' * 80}\n")

        # Outcome
        outcome_icon = {"passed": "✅", "failed": "❌", "skipped": "⏭️", "inconclusive": "❓"}
        outcome_color = {"passed": green, "failed": red, "skipped": yellow, "inconclusive": yellow}
        icon = outcome_icon.get(self.outcome, "❓")
        color = outcome_color.get(self.outcome, yellow)
        print(color(f"{icon} Outcome: {self.outcome.upper()}"))

        # Exit code
        if self.exit_code != 0:
            print(red(f"⚠️  Exit Code: {self.exit_code}"))
        else:
            print(green(f"✓ Exit Code: {self.exit_code}"))

        # Error
        if self.error:
            print(red(f"\n❗ Error: {self.error}"))

        # Metrics
        if self.metrics:
            print("\n📊 Metrics:")
            for key, value in self.metrics.items():
                print(f"   • {key}: {value:.4f}")

        # Artifacts with highlighting
        if self.artifacts:
            print("\n📦 Artifacts:")
            for key, value in self.artifacts.items():
                highlighted = self._highlight_artifact(value, cyan, magenta)
                print(f"   • {key}: {highlighted}")

        # Job ID and logs path
        if self.job_id:
            print(f"\n🆔 Job ID: {self.job_id}")

        if self.logs_path:
            print(f"📝 Logs: {self.logs_path}")

    def _highlight_artifact(self, value: str, cyan_fn, magenta_fn) -> str:
        """Highlight artifact URLs."""
        if value.startswith("wandb://"):
            return magenta_fn(f"📦 {value}")
        elif value.startswith("s3://"):
            return magenta_fn(f"📦 {value}")
        elif value.startswith("file://"):
            return magenta_fn(f"📦 {value}")
        elif value.startswith("http"):
            return cyan_fn(f"🔗 {value}")
        return value


# ============================================================================
# Task Base Classes
# ============================================================================


class Task:
    """Base task - combines execution config with validation logic.

    Tasks are composed of:
    - job_config: What to run, where, with what resources (execution via JobManager)
    - Validation criteria: acceptance rules, metrics extraction (business logic)

    TaskRunner will submit job_config to JobManager and use convert_result()
    to transform JobState into TaskResult with business logic applied.
    """

    def __init__(self, job_config: JobConfig):
        self.job_config = job_config
        self.name = job_config.name
        self.result: Optional[TaskResult] = None  # Cached after first run
        self.dependencies: list[Task] = []  # Set by subclasses or helpers

    def evaluate_result(self, job_state: JobState) -> TaskResult:
        """Evaluate job execution state and return business outcome.

        Override in subclasses for custom validation logic.
        """
        # Determine outcome based on exit code
        if job_state.exit_code == 124:
            outcome = "failed"
            error = "Timeout exceeded"
        elif job_state.exit_code != 0:
            outcome = "failed"
            error = f"Job failed with exit code {job_state.exit_code}"
        else:
            outcome = "passed"
            error = None

        return TaskResult(
            name=self.name,
            started_at=job_state.started_at or datetime.utcnow().isoformat(timespec="seconds"),
            ended_at=job_state.completed_at or datetime.utcnow().isoformat(timespec="seconds"),
            exit_code=job_state.exit_code or 0,
            logs_path=job_state.logs_path,
            job_id=job_state.job_id,
            outcome=outcome,
            error=error,
        )


class TrainingTask(Task):
    """Base for training tasks - extracts metrics and checkpoints.

    Combines JobConfig (execution) with validation criteria (acceptance, metrics).
    """

    def __init__(
        self,
        job_config: JobConfig,
        acceptance: list[AcceptanceRule] | None = None,
        wandb_metrics: list[str] | None = None,
    ):
        super().__init__(job_config)
        self.acceptance = acceptance or []
        self.wandb_metrics = wandb_metrics or []
        self.log_dir: Optional[str] = None  # Set by runner

    def evaluate_result(self, job_state: JobState) -> TaskResult:
        """Override to add metrics extraction and acceptance checking."""
        # Check exit code first
        if job_state.exit_code == 124:
            return TaskResult(
                name=self.name,
                started_at=job_state.started_at or datetime.utcnow().isoformat(timespec="seconds"),
                ended_at=job_state.completed_at or datetime.utcnow().isoformat(timespec="seconds"),
                exit_code=124,
                logs_path=job_state.logs_path,
                outcome="failed",
                error="Timeout exceeded",
            )
        elif job_state.exit_code != 0:
            return TaskResult(
                name=self.name,
                started_at=job_state.started_at or datetime.utcnow().isoformat(timespec="seconds"),
                ended_at=job_state.completed_at or datetime.utcnow().isoformat(timespec="seconds"),
                exit_code=job_state.exit_code,
                logs_path=job_state.logs_path,
                outcome="failed",
                error=f"Job failed with exit code {job_state.exit_code}",
            )

        # Job succeeded - use JobState metrics and artifacts
        # JobManager already extracted wandb info and stored in job_state
        metrics = job_state.metrics
        artifacts = (
            {
                "wandb_run_id": job_state.wandb_run_id,
                "wandb_url": job_state.wandb_url,
                "checkpoint_uri": job_state.checkpoint_uri,
            }
            if job_state.wandb_run_id
            else {}
        )

        # Fetch additional metrics from wandb if specified and not already in job_state
        if self.wandb_metrics and job_state.wandb_url:
            # TODO: JobManager should handle this, but for now we can fetch here
            pass  # JobManager will populate metrics

        # Evaluate acceptance criteria
        if self.acceptance:
            outcome, failed_msgs = _evaluate_thresholds(metrics, self.acceptance)
            error = "; ".join(failed_msgs) if failed_msgs else None
        else:
            outcome = "passed"
            error = None

        return TaskResult(
            name=self.name,
            started_at=job_state.started_at or datetime.utcnow().isoformat(timespec="seconds"),
            ended_at=job_state.completed_at or datetime.utcnow().isoformat(timespec="seconds"),
            exit_code=job_state.exit_code or 0,
            logs_path=job_state.logs_path,
            metrics=metrics,
            artifacts=artifacts,
            job_id=job_state.job_id,
            outcome=outcome,
            error=error,
        )


class LocalTrainingTask(TrainingTask):
    """Local training task.

    JobManager will execute based on job_config.execution="local".
    """

    pass  # No custom logic needed


class RemoteTrainingTask(TrainingTask):
    """Remote training task via SkyPilot.

    JobManager will execute based on job_config.execution="remote".
    Job ID reattachment and state saving are handled by JobManager.
    """

    pass  # No custom logic needed


# ============================================================================
# Helper Functions - Declarative Task Creation
# ============================================================================


def ci(name: str = "metta_ci", timeout_s: int = 1800) -> Task:
    """Create a CI task that runs 'metta ci'."""
    job_config = JobConfig(
        name=name,
        module="__unused__",  # Not used for command-based tasks
        execution="local",
        timeout_s=timeout_s,
        metadata={"cmd": ["metta", "ci"]},
    )
    return Task(job_config)


def local_train(
    name: str,
    module: str,
    args: list[str],
    timeout_s: int = 600,
    acceptance: list[AcceptanceRule] | None = None,
    wandb_metrics: list[str] | None = None,
) -> TrainingTask:
    """Create a local training task.

    Example:
        smoke = local_train(
            name="arena_smoke",
            module="experiments.recipes.arena.train",
            args=["run=smoke", "trainer.total_timesteps=1000"],
        )
    """
    # Parse args list into dict format for JobConfig
    args_dict, overrides_dict = _parse_args_list(args)

    job_config = JobConfig(
        name=name,
        module=module,
        args=args_dict,
        overrides=overrides_dict,
        execution="local",
        timeout_s=timeout_s,
        job_type="train",
    )

    return LocalTrainingTask(
        job_config=job_config,
        acceptance=acceptance,
        wandb_metrics=wandb_metrics,
    )


def remote_train(
    name: str,
    module: str,
    args: list[str],
    timeout_s: int = 7200,
    gpus: int = 1,
    nodes: int = 1,
    use_spot: bool = False,
    acceptance: list[AcceptanceRule] | None = None,
    wandb_metrics: list[str] | None = None,
) -> TrainingTask:
    """Create a remote training task via SkyPilot.

    Example:
        train = remote_train(
            name="arena_100m",
            module="experiments.recipes.arena.train",
            args=["trainer.total_timesteps=100000000"],
            acceptance=[("overview/sps", ge, 40000)],
            wandb_metrics=["overview/sps", "env_agent/heart.get"],
        )
    """
    # Parse args list into dict format for JobConfig
    args_dict, overrides_dict = _parse_args_list(args)

    job_config = JobConfig(
        name=name,
        module=module,
        args=args_dict,
        overrides=overrides_dict,
        execution="remote",
        timeout_s=timeout_s,
        gpus=gpus,
        nodes=nodes,
        spot=use_spot,
        job_type="train",
    )

    return RemoteTrainingTask(
        job_config=job_config,
        acceptance=acceptance,
        wandb_metrics=wandb_metrics,
    )


def evaluate(
    name: str,
    module: str,
    args: list[str] | None = None,
    training_task: TrainingTask | None = None,
    timeout_s: int = 1800,
) -> Task:
    """Create an evaluation task.

    If training_task is provided, TaskRunner will automatically inject
    checkpoint_uri as policy_uri.

    Example:
        eval_task = evaluate(
            name="arena_eval",
            module="experiments.recipes.arena.evaluate",
            training_task=train_task,  # Direct reference!
        )
    """
    # Parse args list into dict format for JobConfig
    args_list = args or []
    args_dict, overrides_dict = _parse_args_list(args_list)

    job_config = JobConfig(
        name=name,
        module=module,
        args=args_dict,
        overrides=overrides_dict,
        execution="local",
        timeout_s=timeout_s,
        job_type="eval",
    )

    task = Task(job_config)
    # Set dependencies - TaskRunner will inject checkpoint_uri from dependency results
    if training_task:
        task.dependencies = [training_task]

    return task


# ============================================================================
# Validation Plan
# ============================================================================


def get_all_tasks() -> list[Task]:
    """Define all tasks for the release pipeline.

    This is the only place you need to edit to add/modify tasks!
    Dependencies are explicit via direct task references.

    Pipeline order:
    1. CI (tests + linting)
    2. Local smoke test
    3. Remote single GPU training (100M timesteps)
    4. Remote multi-GPU training (2B timesteps)
    5. Evaluation on trained policy
    """
    # CI checks
    ci_task = ci()

    # Local smoke test
    smoke_run = f"stable.smoke.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    smoke = local_train(
        name="arena_local_smoke",
        module="experiments.recipes.arena_basic_easy_shaped.train",
        args=[f"run={smoke_run}", "trainer.total_timesteps=1000"],
        timeout_s=600,
    )

    # Single GPU training - 100M timesteps
    train_100m = remote_train(
        name="arena_single_gpu_100m",
        module="experiments.recipes.arena_basic_easy_shaped.train",
        args=["trainer.total_timesteps=100000000"],
        timeout_s=7200,
        gpus=1,
        nodes=1,
        acceptance=[
            ("overview/sps", ge, 40000),
            ("env_agent/heart.get", gt, 0.5),
        ],
        wandb_metrics=["overview/sps", "env_agent/heart.get"],
    )

    # Multi-GPU training - 2B timesteps
    train_2b = remote_train(
        name="arena_multi_gpu_2b",
        module="experiments.recipes.arena_basic_easy_shaped.train",
        args=["trainer.total_timesteps=2000000000"],
        timeout_s=172800,  # 48 hours
        gpus=4,
        nodes=4,
        acceptance=[
            ("overview/sps", ge, 40000),
            ("env_agent/heart.get", gt, 10.0),
        ],
        wandb_metrics=["overview/sps", "env_agent/heart.get"],
    )

    # Evaluation - depends on single-GPU 100M training run
    eval_task = evaluate(
        name="arena_evaluate",
        module="experiments.recipes.arena_basic_easy_shaped.evaluate",
        training_task=train_100m,  # Explicit dependency!
        timeout_s=1800,
    )

    return [ci_task, smoke, train_100m, train_2b, eval_task]
