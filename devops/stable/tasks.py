"""Defines the actual tasks that are run when validating a release."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from operator import ge, gt
from typing import Callable, Literal, Optional

from pydantic import BaseModel

from devops.job_runner import JobResult, LocalJob, RemoteJob
from devops.stable.metrics import extract_metrics, extract_wandb_run_info
from metta.common.util.text_styles import blue, cyan, green, magenta, red, yellow

# Type definitions
Outcome = Literal["passed", "failed", "skipped", "inconclusive"]
# Acceptance Rules are tuples of (metric_name, operator, expected_value), e.g. ("overview/sps", ge, 40000)
AcceptanceRule = tuple[str, Callable[[float, float], bool], float]


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


class Task(ABC):
    """Base task - knows how to run itself and manage dependencies."""

    def __init__(self, name: str):
        self.name = name
        self.result: Optional[TaskResult] = None  # Cached after first run
        self.dependencies: list[Task] = []  # Set by subclasses or helpers

    @abstractmethod
    def execute(self) -> JobResult:
        """Execute the job - override this in subclasses."""
        pass

    def run(self) -> TaskResult:
        """Run task with caching."""
        if self.result:
            return self.result

        job_result = self.execute()
        self.result = self._convert_result(job_result)
        return self.result

    def _convert_result(self, job_result: JobResult) -> TaskResult:
        """Convert JobResult to TaskResult. Override for custom behavior."""
        started = datetime.utcnow().isoformat(timespec="seconds")
        ended = datetime.utcnow().isoformat(timespec="seconds")

        # Check exit code
        if job_result.exit_code == 124:
            return TaskResult(
                name=self.name,
                started_at=started,
                ended_at=ended,
                exit_code=124,
                logs_path=job_result.logs_path,
                outcome="failed",
                error="Timeout exceeded",
            )
        elif job_result.exit_code != 0:
            # Differentiate between launch errors and job execution errors
            if job_result.error:
                # Launch error - show the launch.py output
                error_msg = f"Launch failed:\n{job_result.error}"
            else:
                # Job execution error - point to logs
                error_msg = f"Job failed with exit code {job_result.exit_code}"

            return TaskResult(
                name=self.name,
                started_at=started,
                ended_at=ended,
                exit_code=job_result.exit_code,
                logs_path=job_result.logs_path,
                outcome="failed",
                error=error_msg,
            )

        # Success
        return TaskResult(
            name=self.name,
            started_at=started,
            ended_at=ended,
            exit_code=0,
            logs_path=job_result.logs_path,
            job_id=job_result.job_id,
            outcome="passed",
        )


class LocalCommandTask(Task):
    """Task that runs a local command."""

    def __init__(self, name: str, cmd: list[str], timeout_s: int = 900, log_dir: Optional[str] = None):
        super().__init__(name)
        self.cmd = cmd
        self.timeout_s = timeout_s
        self.log_dir = log_dir

    def execute(self) -> JobResult:
        # log_dir must be set by runner before execution
        if not self.log_dir:
            raise ValueError(f"log_dir not set for task {self.name}")

        job = LocalJob(
            name=self.name,
            cmd=self.cmd,
            timeout_s=self.timeout_s,
            log_dir=self.log_dir,
        )
        return job.wait(stream_output=True)


class TrainingTask(Task):
    """Base for training tasks - extracts metrics and checkpoints."""

    def __init__(
        self,
        name: str,
        module: str,
        args: list[str],
        timeout_s: int = 3600,
        acceptance: list[AcceptanceRule] | None = None,
        wandb_metrics: list[str] | None = None,
        log_dir: Optional[str] = None,
    ):
        super().__init__(name)
        self.module = module
        self.args = args
        self.timeout_s = timeout_s
        self.acceptance = acceptance or []
        self.wandb_metrics = wandb_metrics or []
        self.log_dir = log_dir

    def _convert_result(self, job_result: JobResult) -> TaskResult:
        """Override to add metrics extraction and acceptance checking."""
        started = datetime.utcnow().isoformat(timespec="seconds")
        ended = datetime.utcnow().isoformat(timespec="seconds")

        # Check exit code first
        if job_result.exit_code == 124:
            return TaskResult(
                name=self.name,
                started_at=started,
                ended_at=ended,
                exit_code=124,
                logs_path=job_result.logs_path,
                outcome="failed",
                error="Timeout exceeded",
            )
        elif job_result.exit_code != 0:
            # Differentiate between launch errors and job execution errors
            if job_result.error:
                # Launch error - show the launch.py output
                error_msg = f"Launch failed:\n{job_result.error}"
            else:
                # Job execution error - point to logs
                error_msg = f"Job failed with exit code {job_result.exit_code}"

            return TaskResult(
                name=self.name,
                started_at=started,
                ended_at=ended,
                exit_code=job_result.exit_code,
                logs_path=job_result.logs_path,
                outcome="failed",
                error=error_msg,
            )

        # Extract WandB metrics
        log_text = job_result.get_logs()
        metrics = extract_metrics(log_text, wandb_metrics=self.wandb_metrics)

        # Evaluate acceptance criteria
        if self.acceptance:
            outcome, failed_msgs = _evaluate_thresholds(metrics, self.acceptance)
            error = "; ".join(failed_msgs) if failed_msgs else None
        else:
            outcome = "passed"
            error = None

        # Extract WandB info and construct checkpoint URI
        artifacts = {}
        wandb_info = extract_wandb_run_info(log_text)
        if wandb_info:
            entity, project, run_id = wandb_info
            artifacts["wandb_run_id"] = run_id
            artifacts["checkpoint_uri"] = f"wandb://run/{run_id}"

        return TaskResult(
            name=self.name,
            started_at=started,
            ended_at=ended,
            exit_code=job_result.exit_code,
            logs_path=job_result.logs_path,
            metrics=metrics,
            artifacts=artifacts,
            job_id=job_result.job_id,
            outcome=outcome,
            error=error,
        )


class LocalTrainingTask(TrainingTask):
    """Local training task."""

    def execute(self) -> JobResult:
        # log_dir must be set by runner before execution
        if not self.log_dir:
            raise ValueError(f"log_dir not set for task {self.name}")

        cmd = ["uv", "run", "./tools/run.py", self.module, *self.args]
        job = LocalJob(
            name=self.name,
            cmd=cmd,
            timeout_s=self.timeout_s,
            log_dir=self.log_dir,
        )
        return job.wait(stream_output=True)


class RemoteTrainingTask(TrainingTask):
    """Remote training task via SkyPilot."""

    def __init__(self, *args, base_args: list[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_args = base_args or []

    def execute(self) -> JobResult:
        # log_dir must be set by runner before execution
        if not self.log_dir:
            raise ValueError(f"log_dir not set for task {self.name}")

        # Build the command to run
        cmd = ["uv", "run", "./tools/run.py", self.module, *self.args]

        # Parse base_args to configure resources
        # base_args example: ["--no-spot", "--gpus=4", "--nodes=2"]
        gpus = None
        nodes = None
        use_spot = True  # Default to spot instances

        for arg in self.base_args:
            if arg == "--no-spot":
                use_spot = False
            elif arg.startswith("--gpus="):
                gpus = arg.split("=")[1]
            elif arg.startswith("--nodes="):
                nodes = int(arg.split("=")[1])

        # Build resources dict for RemoteJob
        resources = {}
        if gpus:
            resources["accelerators"] = gpus
            resources["use_spot"] = use_spot

        # Create RemoteJob with the same API as LocalJob
        job = RemoteJob(
            name=self.name,
            cmd=cmd,
            timeout_s=self.timeout_s,
            log_dir=self.log_dir,
            resources=resources,
            num_nodes=nodes if nodes and nodes > 1 else 1,
        )
        return job.wait(stream_output=True)


class EvaluationTask(LocalCommandTask):
    """Evaluation task - runs evaluation with policy from training dependency."""

    def __init__(
        self,
        name: str,
        module: str,
        args: list[str] | None = None,
        training_task: TrainingTask | None = None,
        timeout_s: int = 1800,
    ):
        args = list(args) if args else []

        # Store for later use in execute()
        self._training_task = training_task
        self._module = module
        self._base_args = args

        # Build initial command (will be updated in execute if needed)
        cmd = ["uv", "run", "./tools/run.py", module, *args]
        super().__init__(name, cmd, timeout_s)

        # Set dependencies AFTER super().__init__() to avoid being overwritten
        if training_task:
            self.dependencies = [training_task]

    def execute(self) -> JobResult:
        """Execute evaluation, injecting policy_uri from training dependency."""
        # Get checkpoint URI from training dependency
        args = list(self._base_args)
        if self._training_task and self._training_task.result:
            checkpoint_uri = self._training_task.result.artifacts.get("checkpoint_uri")
            if checkpoint_uri:
                has_policy = any(arg.startswith("policy_uri=") for arg in args)
                if not has_policy:
                    args.append(f"policy_uri={checkpoint_uri}")

        # Update command with policy URI
        self.cmd = ["uv", "run", "./tools/run.py", self._module, *args]

        return super().execute()


# ============================================================================
# Helper Functions - Declarative Task Creation
# ============================================================================


def ci(name: str = "metta_ci", timeout_s: int = 1800) -> Task:
    cmd = ["metta", "ci"]
    return LocalCommandTask(name=name, cmd=cmd, timeout_s=timeout_s)


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
    return LocalTrainingTask(
        name=name,
        module=module,
        args=args,
        timeout_s=timeout_s,
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
    base_args = [f"--gpus={gpus}", f"--nodes={nodes}"]
    if not use_spot:
        base_args.insert(0, "--no-spot")

    return RemoteTrainingTask(
        name=name,
        module=module,
        args=args,
        timeout_s=timeout_s,
        acceptance=acceptance,
        wandb_metrics=wandb_metrics,
        base_args=base_args,
    )


def evaluate(
    name: str,
    module: str,
    args: list[str] | None = None,
    training_task: TrainingTask | None = None,
    timeout_s: int = 1800,
) -> Task:
    """Create an evaluation task.

    Example:
        eval_task = evaluate(
            name="arena_eval",
            module="experiments.recipes.arena.evaluate",
            training_task=train_task,  # Direct reference!
        )
    """
    return EvaluationTask(
        name=name,
        module=module,
        args=args,
        training_task=training_task,
        timeout_s=timeout_s,
    )


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
        timeout_s=86400,
        gpus=4,
        nodes=4,
        acceptance=[
            ("overview/sps", ge, 40000),
            ("env_agent/heart.get", gt, 5.0),
        ],
        wandb_metrics=["overview/sps", "env_agent/heart.get"],
    )

    # Evaluation - depends on multi-GPU training
    eval_task = evaluate(
        name="arena_evaluate",
        module="experiments.recipes.arena_basic_easy_shaped.evaluate",
        training_task=train_100m,  # Explicit dependency!
        timeout_s=1800,
    )

    return [ci_task, smoke, train_100m, train_2b, eval_task]
