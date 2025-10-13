"""Defines the actual tasks that are run when validating a release."""

from __future__ import annotations

from datetime import datetime
from operator import ge, gt
from typing import Callable

from metta.jobs.job_config import AcceptanceCriterion, JobConfig, RemoteConfig

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


def ci_task(name: str, cmd: list[str], timeout_s: int = 1800) -> Task:
    """Create a CI task that runs a shell command."""
    return Task(JobConfig(name=name, module="__unused__", timeout_s=timeout_s, metadata={"cmd": cmd}))


def tool_task(
    name: str,
    module: str,
    args: list[str] | None = None,
    timeout_s: int = 1800,
    remote: RemoteConfig | None = None,
    acceptance: list[AcceptanceRule] | None = None,
    dependency_names: list[str] | None = None,
) -> Task:
    """Create a tool-based task (train/eval/etc).

    Args:
        name: Task name
        module: Python module to run (e.g., "arena.train")
        args: Arguments as list of "key=value" strings
        timeout_s: Timeout in seconds
        remote: Remote execution config (None = local)
        acceptance: Acceptance criteria for validation
        dependency_names: Names of tasks this depends on
    """
    args_dict, overrides_dict = _parse_args_list(args or [])
    # Automatically detect training jobs by module path (e.g., "arena.train", "navigation.train")
    is_training = module.endswith(".train")

    # Convert acceptance criteria from AcceptanceRule to AcceptanceCriterion for JobConfig
    metrics_to_track = []
    acceptance_criteria_list = []
    if acceptance:
        # operator.__name__ gives "ge", "gt", etc. - convert to symbols
        op_to_symbol = {"ge": ">=", "gt": ">", "le": "<=", "lt": "<", "eq": "=="}
        for metric, op, threshold in acceptance:
            metrics_to_track.append(metric)
            operator_symbol = op_to_symbol.get(op.__name__, op.__name__)
            acceptance_criteria_list.append(
                AcceptanceCriterion(metric=metric, operator=operator_symbol, threshold=threshold)
            )

    # Assert that only training jobs can have metrics to track
    if metrics_to_track and not is_training:
        raise ValueError(f"Task {name} has metrics_to_track but is not a training job")

    job_config = JobConfig(
        name=name,
        module=module,
        args=args_dict,
        overrides=overrides_dict,
        timeout_s=timeout_s,
        remote=remote,
        is_training_job=is_training,
        metrics_to_track=metrics_to_track,
        acceptance_criteria=acceptance_criteria_list,
    )
    return Task(job_config=job_config, acceptance=acceptance, dependency_names=dependency_names)


def get_all_tasks() -> list[Task]:
    """Define all release validation tasks with explicit dependencies."""
    # CI checks
    python_ci_task = ci_task("python_ci", ["metta", "pytest", "--ci"])
    cpp_ci_task = ci_task("cpp_ci", ["metta", "cpptest", "--test"])
    cpp_benchmark_task = ci_task("cpp_benchmark", ["metta", "cpptest", "--benchmark"])

    # Local smoke test
    smoke_run = f"stable.smoke.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    smoke = tool_task(
        name="arena_local_smoke",
        module="arena_basic_easy_shaped.train",
        args=[f"run={smoke_run}", "trainer.total_timesteps=1000"],
        timeout_s=600,
    )

    # Single GPU training - 100M timesteps
    train_100m = tool_task(
        name="arena_single_gpu_100m",
        module="arena_basic_easy_shaped.train",
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
        module="arena_basic_easy_shaped.train",
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
        module="arena_basic_easy_shaped.evaluate",
        dependency_names=["arena_single_gpu_100m"],  # Dependency by name
        timeout_s=1800,
    )

    return [python_ci_task, cpp_ci_task, cpp_benchmark_task, smoke, train_100m, train_2b, eval_task]
