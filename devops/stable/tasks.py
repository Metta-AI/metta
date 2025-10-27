"""Defines the actual tasks that are run when validating a release."""

from __future__ import annotations

from datetime import datetime
from operator import ge, gt
from typing import Callable

from metta.jobs.job_config import JobConfig, RemoteConfig

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


def python_ci(name: str = "python_ci", timeout_s: int = 1800) -> Task:
    return Task(
        JobConfig(name=name, module="__unused__", timeout_s=timeout_s, metadata={"cmd": ["metta", "pytest", "--ci"]})
    )


def cpp_ci(name: str = "cpp_ci", timeout_s: int = 1800) -> Task:
    return Task(
        JobConfig(name=name, module="__unused__", timeout_s=timeout_s, metadata={"cmd": ["metta", "cpptest", "--test"]})
    )


def cpp_benchmark(name: str = "cpp_benchmark", timeout_s: int = 1800) -> Task:
    return Task(
        JobConfig(
            name=name, module="__unused__", timeout_s=timeout_s, metadata={"cmd": ["metta", "cpptest", "--benchmark"]}
        )
    )


def local_train(
    name: str,
    module: str,
    args: list[str],
    timeout_s: int = 600,
    acceptance: list[AcceptanceRule] | None = None,
) -> Task:
    """Create a local training task (remote=None in JobConfig)."""
    args_dict, overrides_dict = _parse_args_list(args)
    job_config = JobConfig(
        name=name, module=module, args=args_dict, overrides=overrides_dict, timeout_s=timeout_s, job_type="train"
    )
    return Task(job_config=job_config, acceptance=acceptance)


def remote_train(
    name: str,
    module: str,
    args: list[str],
    timeout_s: int = 7200,
    gpus: int = 1,
    nodes: int = 1,
    use_spot: bool = False,
    acceptance: list[AcceptanceRule] | None = None,
) -> Task:
    """Create a remote training task (remote=RemoteConfig in JobConfig)."""
    args_dict, overrides_dict = _parse_args_list(args)
    job_config = JobConfig(
        name=name,
        module=module,
        args=args_dict,
        overrides=overrides_dict,
        timeout_s=timeout_s,
        job_type="train",
        remote=RemoteConfig(gpus=gpus, nodes=nodes, spot=use_spot),
    )
    return Task(job_config=job_config, acceptance=acceptance)


def evaluate(
    name: str,
    module: str,
    args: list[str] | None = None,
    dependency_names: list[str] | None = None,
    timeout_s: int = 1800,
) -> Task:
    """Create evaluation task. If dependency_names provided, can look up checkpoint_uri."""
    args_list = args or []
    args_dict, overrides_dict = _parse_args_list(args_list)
    job_config = JobConfig(
        name=name, module=module, args=args_dict, overrides=overrides_dict, timeout_s=timeout_s, job_type="eval"
    )
    return Task(job_config, dependency_names=dependency_names)


def get_all_tasks() -> list[Task]:
    """Define all release validation tasks with explicit dependencies."""
    # CI checks
    python_ci_task = python_ci()
    cpp_ci_task = cpp_ci()
    cpp_benchmark_task = cpp_benchmark()

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
            ("env_agent/heart.gained", gt, 0.5),
        ],
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
            ("env_agent/heart.gained", gt, 10.0),
        ],
    )

    # Evaluation - depends on single-GPU 100M training run
    eval_task = evaluate(
        name="arena_evaluate",
        module="experiments.recipes.arena_basic_easy_shaped.evaluate",
        dependency_names=["arena_single_gpu_100m"],  # Dependency by name
        timeout_s=1800,
    )

    return [python_ci_task, cpp_ci_task, cpp_benchmark_task, smoke, train_100m, train_2b, eval_task]
