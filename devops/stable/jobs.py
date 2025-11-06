"""Defines the actual jobs that are run when validating a release."""

import datetime

import metta.jobs.job_config


def ci_job(name: str, cmd: list[str], timeout_s: int = 1800) -> metta.jobs.job_config.JobConfig:
    """Create a CI job that runs a shell command."""
    return metta.jobs.job_config.JobConfig(name=name, module="__unused__", timeout_s=timeout_s, metadata={"cmd": cmd})


def tool_job(
    name: str,
    tool_path: str,
    args: list[str] | None = None,
    timeout_s: int = 1800,
    remote: metta.jobs.job_config.RemoteConfig | None = None,
    acceptance_criteria: list[metta.jobs.job_config.AcceptanceCriterion] | None = None,
    dependency_names: list[str] | None = None,
) -> metta.jobs.job_config.JobConfig:
    """Create a tool-based job (train/eval/etc).

    Args:
        name: Job name
        tool_path: Tool path to run (e.g., "arena.train", "navigation.evaluate")
        args: Arguments as list of "key=value" strings (e.g., ["run=test", "trainer.total_timesteps=100"])
        timeout_s: Timeout in seconds
        remote: Remote execution config (None = local)
        acceptance_criteria: Acceptance criteria for validation
        dependency_names: Names of jobs this depends on
    """
    # Automatically detect training jobs by tool path (e.g., "arena.train", "navigation.train")
    is_training = tool_path.endswith(".train")

    # Extract metrics to track from acceptance criteria
    metrics_to_track = [criterion.metric for criterion in (acceptance_criteria or [])]

    # Assert that only training jobs can have metrics to track
    if metrics_to_track and not is_training:
        raise ValueError(f"Job {name} has metrics_to_track but is not a training job")

    return metta.jobs.job_config.JobConfig(
        name=name,
        module=tool_path,
        args=args or [],
        timeout_s=timeout_s,
        remote=remote,
        is_training_job=is_training,
        metrics_to_track=metrics_to_track,
        acceptance_criteria=acceptance_criteria or [],
        dependency_names=dependency_names or [],
    )


def get_all_jobs() -> list[metta.jobs.job_config.JobConfig]:
    """Define all release validation jobs with explicit dependencies."""
    # CI checks
    python_ci = ci_job("python_ci", ["metta", "pytest", "--ci"])
    cpp_ci = ci_job("cpp_ci", ["metta", "cpptest", "--test"])
    cpp_benchmark = ci_job("cpp_benchmark", ["metta", "cpptest", "--benchmark"])

    # Local smoke test
    smoke_run = f"stable.smoke.{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    smoke = tool_job(
        name="arena_local_smoke",
        tool_path="arena_basic_easy_shaped.train",
        args=[f"run={smoke_run}", "trainer.total_timesteps=1000"],
        timeout_s=600,
    )

    # Single GPU training - 100M timesteps
    train_100m = tool_job(
        name="arena_single_gpu_100m",
        tool_path="arena_basic_easy_shaped.train",
        args=["trainer.total_timesteps=100000000"],
        timeout_s=7200,
        remote=metta.jobs.job_config.RemoteConfig(gpus=1, nodes=1),
        acceptance_criteria=[
            metta.jobs.job_config.AcceptanceCriterion(metric="overview/sps", operator=">=", threshold=40000),
            metta.jobs.job_config.AcceptanceCriterion(metric="env_agent/heart.gained", operator=">", threshold=0.1),
        ],
    )

    # Multi-GPU training - 2B timesteps
    train_2b = tool_job(
        name="arena_multi_gpu_2b",
        tool_path="arena_basic_easy_shaped.train",
        args=["trainer.total_timesteps=2000000000"],
        timeout_s=172800,  # 48 hours
        remote=metta.jobs.job_config.RemoteConfig(gpus=4, nodes=4),
        acceptance_criteria=[
            metta.jobs.job_config.AcceptanceCriterion(metric="overview/sps", operator=">=", threshold=80000),
            metta.jobs.job_config.AcceptanceCriterion(metric="env_agent/heart.gained", operator=">", threshold=1.0),
        ],
    )

    # Evaluation - depends on single-GPU 100M training run
    eval_job = tool_job(
        name="arena_evaluate",
        tool_path="arena_basic_easy_shaped.evaluate",
        dependency_names=["arena_single_gpu_100m"],  # Dependency by name
        timeout_s=1800,
    )

    return [python_ci, cpp_ci, cpp_benchmark, smoke, train_100m, train_2b, eval_job]
