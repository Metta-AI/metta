"""Defines the actual jobs that are run when validating a release."""

from __future__ import annotations

from datetime import datetime

from metta.jobs.job_config import AcceptanceCriterion, JobConfig, MetricsSource, RemoteConfig


def ci_job(name: str, cmd_parts: list[str], timeout_s: int = 1800) -> JobConfig:
    """Create a CI job that runs a shell command.

    Args:
        name: Job name
        cmd_parts: Command parts as list (e.g., ["metta", "pytest", "--ci"])
        timeout_s: Timeout in seconds
    """
    import shlex

    cmd_string = shlex.join(cmd_parts)
    return JobConfig(name=name, cmd=cmd_string, timeout_s=timeout_s)


def tool_job(
    name: str,
    tool_path: str,
    args: list[str] | None = None,
    timeout_s: int = 1800,
    remote: RemoteConfig | None = None,
    acceptance_criteria: list[AcceptanceCriterion] | None = None,
    dependency_names: list[str] | None = None,
) -> JobConfig:
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
    # Parse args list into args dict and overrides dict
    args_dict = {}
    overrides_dict = {}
    for arg in args or []:
        if "=" in arg:
            key, value = arg.split("=", 1)
            if "." in key:
                overrides_dict[key] = value
            else:
                args_dict[key] = value

    # Automatically detect training jobs by tool path (e.g., "arena.train", "navigation.train")
    is_training = tool_path.endswith(".train")

    # Extract metrics to track from acceptance criteria
    metrics_to_track = [criterion.metric for criterion in (acceptance_criteria or [])]

    return JobConfig(
        name=name,
        tool=tool_path,
        args=args_dict,
        overrides=overrides_dict,
        timeout_s=timeout_s,
        remote=remote,
        metrics_source=MetricsSource.WANDB if is_training else MetricsSource.NONE,
        metrics_to_track=metrics_to_track,
        acceptance_criteria=acceptance_criteria or [],
        dependency_names=dependency_names or [],
    )


def cogames_job(
    name: str,
    mission: str,
    variants: list[str],
    steps: int,
    timeout_s: int = 1800,
    remote: RemoteConfig | None = None,
    acceptance_criteria: list[AcceptanceCriterion] | None = None,
    eval_episodes: int = 10,
) -> JobConfig:
    """Create a cogames training+evaluation job with log-based metrics.

    Args:
        name: Job name
        mission: Mission name (e.g., "training_facility.harvest")
        variants: List of variant names (e.g., ["lonely_heart", "heart_chorus"])
        steps: Number of training steps
        timeout_s: Timeout in seconds
        remote: Remote execution config (None = local)
        acceptance_criteria: Acceptance criteria for validation
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

    # Extract metrics to track from acceptance criteria
    metrics_to_track = [criterion.metric for criterion in (acceptance_criteria or [])]

    return JobConfig(
        name=name,
        cmd=cmd,
        timeout_s=timeout_s,
        remote=remote,
        metrics_source=MetricsSource.COGAMES_LOG,
        metrics_to_track=metrics_to_track,
        acceptance_criteria=acceptance_criteria or [],
    )


def get_all_jobs() -> list[JobConfig]:
    """Define all release validation jobs with explicit dependencies."""
    # CI checks
    python_ci = ci_job("python_ci", ["metta", "pytest", "--ci"])
    cpp_ci = ci_job("cpp_ci", ["metta", "cpptest", "--test"])
    cpp_benchmark = ci_job("cpp_benchmark", ["metta", "cpptest", "--benchmark"])

    # Local smoke test
    smoke_run = f"stable.smoke.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        remote=RemoteConfig(gpus=1, nodes=1),
        acceptance_criteria=[
            AcceptanceCriterion(metric="overview/sps", operator=">=", threshold=40000),
            AcceptanceCriterion(metric="env_agent/heart.gained", operator=">", threshold=0.1),
        ],
    )

    # Multi-GPU training - 2B timesteps
    train_2b = tool_job(
        name="arena_multi_gpu_2b",
        tool_path="arena_basic_easy_shaped.train",
        args=["trainer.total_timesteps=2000000000"],
        timeout_s=172800,  # 48 hours
        remote=RemoteConfig(gpus=4, nodes=4),
        acceptance_criteria=[
            AcceptanceCriterion(metric="overview/sps", operator=">=", threshold=80000),
            AcceptanceCriterion(metric="env_agent/heart.gained", operator=">", threshold=1.0),
        ],
    )

    # Evaluation - depends on single-GPU 100M training run
    eval_job = tool_job(
        name="arena_evaluate",
        tool_path="arena_basic_easy_shaped.evaluate",
        dependency_names=["arena_single_gpu_100m"],  # Dependency by name
        timeout_s=1800,
    )

    # Cogames validation jobs
    cogames_local_smoke = cogames_job(
        name="cogames_local_smoke",
        mission="training_facility.harvest",
        variants=["lonely_heart"],
        steps=100000,
        timeout_s=30,
        eval_episodes=3,
        acceptance_criteria=[
            AcceptanceCriterion(metric="SPS", operator=">=", threshold=1000),
        ],
    )

    # Remote: 600 epochs × 17,024 steps/epoch = 10,214,400 steps
    cogames_remote_smoke = cogames_job(
        name="cogames_remote_smoke",
        mission="training_facility.harvest",
        variants=["lonely_heart"],
        steps=10_214_400,  # Reach epoch 600
        timeout_s=1800,  # 30 minutes
        remote=RemoteConfig(gpus=1, nodes=1),
        eval_episodes=10,
        acceptance_criteria=[
            AcceptanceCriterion(metric="SPS", operator=">=", threshold=10000),
            AcceptanceCriterion(metric="avg_agent_metrics.heart.gained", operator=">=", threshold=0.1),
        ],
    )

    return [
        python_ci,
        cpp_ci,
        cpp_benchmark,
        smoke,
        cogames_local_smoke,
        cogames_remote_smoke,
        train_100m,
        train_2b,
        eval_job,
    ]
