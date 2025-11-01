"""Defines the actual tasks that are run when validating a release."""

from __future__ import annotations

from datetime import datetime

from metta.jobs.job_config import AcceptanceCriterion, JobConfig, RemoteConfig


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


def ci_task(name: str, cmd: list[str], timeout_s: int = 1800) -> JobConfig:
    """Create a CI task that runs a shell command."""
    return JobConfig(name=name, module="__unused__", timeout_s=timeout_s, metadata={"cmd": cmd})


def tool_task(
    name: str,
    module: str,
    args: list[str] | None = None,
    timeout_s: int = 1800,
    remote: RemoteConfig | None = None,
    acceptance_criteria: list[AcceptanceCriterion] | None = None,
    dependency_names: list[str] | None = None,
) -> JobConfig:
    """Create a tool-based task (train/eval/etc).

    Args:
        name: Task name
        module: Python module to run (e.g., "arena.train")
        args: Arguments as list of "key=value" strings
        timeout_s: Timeout in seconds
        remote: Remote execution config (None = local)
        acceptance_criteria: Acceptance criteria for validation
        dependency_names: Names of tasks this depends on
    """
    args_dict, overrides_dict = _parse_args_list(args or [])
    # Automatically detect training jobs by module path (e.g., "arena.train", "navigation.train")
    is_training = module.endswith(".train")

    # Extract metrics to track from acceptance criteria
    metrics_to_track = [criterion.metric for criterion in (acceptance_criteria or [])]

    # Assert that only training jobs can have metrics to track
    if metrics_to_track and not is_training:
        raise ValueError(f"Task {name} has metrics_to_track but is not a training job")

    return JobConfig(
        name=name,
        module=module,
        args=args_dict,
        overrides=overrides_dict,
        timeout_s=timeout_s,
        remote=remote,
        is_training_job=is_training,
        metrics_to_track=metrics_to_track,
        acceptance_criteria=acceptance_criteria or [],
        dependency_names=dependency_names or [],
    )


def get_all_tasks() -> list[JobConfig]:
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
        acceptance_criteria=[
            AcceptanceCriterion(metric="overview/sps", operator=">=", threshold=40000),
            AcceptanceCriterion(metric="env_agent/heart.gained", operator=">", threshold=0.1),
        ],
    )

    # Multi-GPU training - 2B timesteps
    train_2b = tool_task(
        name="arena_multi_gpu_2b",
        module="arena_basic_easy_shaped.train",
        args=["trainer.total_timesteps=2000000000"],
        timeout_s=172800,  # 48 hours
        remote=RemoteConfig(gpus=4, nodes=4),
        acceptance_criteria=[
            AcceptanceCriterion(metric="overview/sps", operator=">=", threshold=40000),
            AcceptanceCriterion(metric="env_agent/heart.gained", operator=">", threshold=1.0),
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
