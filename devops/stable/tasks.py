"""Defines the actual tasks that are run when validating a release."""

from __future__ import annotations

from datetime import datetime
from operator import ge, gt
from typing import Callable

from metta.jobs.job_config import JobConfig, MetricsSource, RemoteConfig

AcceptanceRule = tuple[str, Callable[[float, float], bool], float]


def _parse_args_list(args: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """Parse args like ["run=test", "trainer.total_timesteps=100"] into args and overrides."""
    args_dict = {}
    overrides_dict = {}
    for arg in args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            if "." in key:
                overrides_dict[key] = value
            else:
                args_dict[key] = value
    return args_dict, overrides_dict


class Task:
    """Validation task: JobConfig + acceptance rules + dependencies.

    Replaces the old Task/TaskResult duplication - TaskRunner queries JobState directly
    for outcomes instead of caching in a separate TaskResult object.
    """

    def __init__(
        self,
        job_config: JobConfig,
        acceptance: list[AcceptanceRule] | None = None,
        dependency_names: list[str] | None = None,
    ):
        self.job_config = job_config
        self.name = job_config.name
        self.acceptance = acceptance or []
        self.dependency_names = dependency_names or []


def cmd_task(name: str, cmd: str, timeout_s: int = 1800, remote: RemoteConfig | None = None) -> Task:
    """Create a task that runs a shell command (local or remote).

    Args:
        name: Task name
        cmd: Shell command to execute
        timeout_s: Timeout in seconds
        remote: Remote execution config (None = local)
    """
    return Task(JobConfig(name=name, cmd=cmd, timeout_s=timeout_s, remote=remote))


def cogames_task(
    name: str,
    mission: str,
    variants: list[str],
    steps: int,
    timeout_s: int = 1800,
    remote: RemoteConfig | None = None,
    stats_to_track: list[str] | None = None,
    acceptance: list[AcceptanceRule] | None = None,
    eval_episodes: int = 10,
) -> Task:
    """Create a cogames training+evaluation task with stats tracking.

    Args:
        name: Task name
        mission: Mission name (e.g., "training_facility.harvest")
        variants: List of variant names (e.g., ["lonely_heart", "heart_chorus"])
        steps: Number of training steps
        timeout_s: Timeout in seconds
        remote: Remote execution config (None = local)
        stats_to_track: Stats to parse from eval results (e.g., ["avg_game_stats.food_spawned"])
        acceptance: Acceptance criteria for validation
        eval_episodes: Number of evaluation episodes to run (default: 10)
    """
    # Build cogames train+eval wrapper command
    variants_args = " ".join(f"--variant {v}" for v in variants)
    checkpoints_dir = f"./train_dir/{name}/checkpoints"
    cmd = (
        f"uv run ./devops/stable/cogames_train_eval.py "
        f"--mission {mission} {variants_args} "
        f"--steps {steps} "
        f"--checkpoints-dir {checkpoints_dir} "
        f"--eval-episodes {eval_episodes}"
    )

    # Extract metric keys from acceptance criteria
    metrics_to_track = []
    acceptance_criteria_dict = None
    if acceptance:
        metrics_to_track = [key for key, _op, _expected in acceptance]
        # Convert AcceptanceRule list to dict format for JobConfig
        op_to_symbol = {"ge": ">=", "gt": ">", "le": "<=", "lt": "<", "eq": "=="}
        acceptance_criteria_dict = {
            key: (op_to_symbol.get(op.__name__, op.__name__), expected) for key, op, expected in acceptance
        }
    elif stats_to_track:
        metrics_to_track = stats_to_track

    job_config = JobConfig(
        name=name,
        cmd=cmd,
        timeout_s=timeout_s,
        remote=remote,
        metrics_source=MetricsSource.COGAMES_LOG,
        metrics_to_track=metrics_to_track,
        acceptance_criteria=acceptance_criteria_dict,
    )
    return Task(job_config=job_config, acceptance=acceptance, dependency_names=[])


def tool_task(
    name: str,
    tool: str,
    args: list[str] | None = None,
    timeout_s: int = 1800,
    remote: RemoteConfig | None = None,
    acceptance: list[AcceptanceRule] | None = None,
    dependency_names: list[str] | None = None,
) -> Task:
    """Create a tool-based task (train/eval/etc).

    Args:
        name: Task name
        tool: Tool to run via tools/run.py (e.g., "arena.train")
        args: Arguments as list of "key=value" strings
        timeout_s: Timeout in seconds
        remote: Remote execution config (None = local)
        acceptance: Acceptance criteria for validation
        dependency_names: Names of tasks this depends on
    """
    args_dict, overrides_dict = _parse_args_list(args or [])

    # Extract metric keys from acceptance criteria
    metrics_to_track = []
    acceptance_criteria_dict = None
    if acceptance:
        metrics_to_track = [key for key, _op, _expected in acceptance]
        # Convert AcceptanceRule list to dict format for JobConfig
        op_to_symbol = {"ge": ">=", "gt": ">", "le": "<=", "lt": "<", "eq": "=="}
        acceptance_criteria_dict = {
            key: (op_to_symbol.get(op.__name__, op.__name__), expected) for key, op, expected in acceptance
        }

    # Determine metrics source: training tools use WandB, others have no metrics
    metrics_source = MetricsSource.WANDB if tool.endswith(".train") else MetricsSource.NONE

    job_config = JobConfig(
        name=name,
        tool=tool,
        args=args_dict,
        overrides=overrides_dict,
        timeout_s=timeout_s,
        remote=remote,
        metrics_source=metrics_source,
        metrics_to_track=metrics_to_track,
        acceptance_criteria=acceptance_criteria_dict,
    )
    return Task(job_config=job_config, acceptance=acceptance, dependency_names=dependency_names)


def get_all_tasks() -> list[Task]:
    """Define all release validation tasks with explicit dependencies."""
    # CI checks
    python_ci_task = cmd_task("python_ci", "metta pytest --ci")
    cpp_ci_task = cmd_task("cpp_ci", "metta cpptest --test")
    cpp_benchmark_task = cmd_task("cpp_benchmark", "metta cpptest --benchmark")

    # Local smoke test
    smoke_run = f"stable.smoke.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    smoke = tool_task(
        name="arena_local_smoke",
        tool="arena_basic_easy_shaped.train",
        args=[f"run={smoke_run}", "trainer.total_timesteps=1000"],
        timeout_s=600,
    )

    # Cogames smoke tests
    # Each epoch ≈ 17k steps, so 100k steps ≈ 6 epochs
    cogames_local_smoke = cogames_task(
        name="cogames_local_smoke",
        mission="training_facility.harvest",
        variants=["lonely_heart"],
        steps=100000,
        timeout_s=30,
        eval_episodes=3,
        stats_to_track=["SPS", "avg_agent_metrics.heart.gained"],
        acceptance=[
            ("SPS", ge, 1000),  # Very permissive - just ensure it runs (CPU)
        ],
    )

    # Remote: 600 epochs × 17,024 steps/epoch = 10,214,400 steps
    cogames_remote_smoke = cogames_task(
        name="cogames_remote_smoke",
        mission="training_facility.harvest",
        variants=["lonely_heart"],
        steps=10_214_400,  # Reach epoch 600
        timeout_s=1800,  # 30 minutes
        remote=RemoteConfig(gpus=1, nodes=1),
        eval_episodes=10,
        stats_to_track=["SPS", "avg_agent_metrics.heart.gained"],
        acceptance=[
            ("SPS", ge, 10000),  # At least 10k samples/sec on GPU
            ("avg_agent_metrics.heart.gained", ge, 0.1),  # Agent should gain some hearts
        ],
    )

    # Single GPU training - 100M timesteps
    train_100m = tool_task(
        name="arena_single_gpu_100m",
        tool="arena_basic_easy_shaped.train",
        args=["trainer.total_timesteps=100000000"],
        timeout_s=7200,
        remote=RemoteConfig(gpus=1, nodes=1),
        acceptance=[
            ("overview/sps", ge, 40000),
            ("env_agent/heart.gained", gt, 0.1),
        ],
    )

    # Multi-GPU training - 2B timesteps
    train_2b = tool_task(
        name="arena_multi_gpu_2b",
        tool="arena_basic_easy_shaped.train",
        args=["trainer.total_timesteps=2000000000"],
        timeout_s=172800,  # 48 hours
        remote=RemoteConfig(gpus=4, nodes=4),
        acceptance=[
            ("overview/sps", ge, 40000),
            ("env_agent/heart.gained", gt, 1.0),
        ],
    )

    # Evaluation - depends on single-GPU 100M training run
    eval_task = tool_task(
        name="arena_evaluate",
        tool="arena_basic_easy_shaped.evaluate",
        dependency_names=["arena_single_gpu_100m"],  # Dependency by name
        timeout_s=1800,
    )

    return [
        python_ci_task,
        cpp_ci_task,
        cpp_benchmark_task,
        smoke,
        cogames_local_smoke,
        cogames_remote_smoke,
        train_100m,
        train_2b,
        eval_task,
    ]
