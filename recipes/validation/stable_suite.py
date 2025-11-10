"""Stable release suite - comprehensive tests for releases.

Includes both CI smoke tests and long-running stable validation tests.
Job names are fully qualified with user/timestamp and version.
"""

from __future__ import annotations

from metta.jobs.job_config import AcceptanceCriterion, JobConfig, RemoteConfig


def get_stable_jobs(version: str) -> list[JobConfig]:
    """Define stable-specific test jobs for recipes.

    These are comprehensive long-running tests that validate releases on remote infrastructure.
    Does NOT include CI smoke tests - those are combined in devops/stable/jobs.py.

    Args:
        version: Version prefix for job names (e.g., "v0.1.0")

    Returns:
        List of fully configured job configs with version-prefixed names
    """
    # ========================================
    # Arena Basic Easy Shaped - Stable Tests
    # ========================================

    # Single GPU training - 100M timesteps
    arena_train_name = f"{version}_arena_single_gpu_100m"
    arena_train_100m = JobConfig(
        name=arena_train_name,
        module="recipes.prod.arena_basic_easy_shaped.train",
        args=[
            f"run={arena_train_name}",
            "trainer.total_timesteps=100000000",
        ],
        timeout_s=7200,
        remote=RemoteConfig(gpus=1, nodes=1),
        is_training_job=True,
        metrics_to_track=["overview/sps", "env_agent/heart.gained"],
        acceptance_criteria=[
            AcceptanceCriterion(metric="overview/sps", operator=">=", threshold=40000),
            AcceptanceCriterion(metric="env_agent/heart.gained", operator=">", threshold=0.1),
        ],
    )

    # Multi-GPU training - 2B timesteps
    arena_train_2b_name = f"{version}_arena_multi_gpu_2b"
    arena_train_2b = JobConfig(
        name=arena_train_2b_name,
        module="recipes.prod.arena_basic_easy_shaped.train",
        args=[
            f"run={arena_train_2b_name}",
            "trainer.total_timesteps=2000000000",
        ],
        timeout_s=172800,
        remote=RemoteConfig(gpus=4, nodes=4),
        is_training_job=True,
        metrics_to_track=["overview/sps", "env_agent/heart.gained"],
        acceptance_criteria=[
            AcceptanceCriterion(metric="overview/sps", operator=">=", threshold=80000),
            AcceptanceCriterion(metric="env_agent/heart.gained", operator=">", threshold=1.0),
        ],
    )

    # Evaluation - depends on single-GPU 100M training run
    arena_eval_name = f"{version}_arena_evaluate"
    arena_eval = JobConfig(
        name=arena_eval_name,
        module="recipes.prod.arena_basic_easy_shaped.evaluate",
        args=[f'policy_uris=["s3://softmax-public/policies/{arena_train_name}:latest"]'],
        dependency_names=[arena_train_name],
        timeout_s=1800,
    )

    # ========================================
    # CvC Small Maps - Stable Tests
    # ========================================

    # Multi-GPU training - 2B timesteps
    cvc_small_train_name = f"{version}_cvc_small_multi_gpu_2b"
    cvc_small_train_2b = JobConfig(
        name=cvc_small_train_name,
        module="recipes.prod.cvc.small_maps.train",
        args=[
            f"run={cvc_small_train_name}",
            "trainer.total_timesteps=2000000000",
            "num_cogs=4",
            'variants=["lonely_heart","heart_chorus","pack_rat","neutral_faced"]',
        ],
        timeout_s=172800,
        remote=RemoteConfig(gpus=4, nodes=4),
        is_training_job=True,
        metrics_to_track=["overview/sps", "env_agent/heart.gained"],
        acceptance_criteria=[
            AcceptanceCriterion(metric="overview/sps", operator=">=", threshold=80000),
            AcceptanceCriterion(metric="env_agent/heart.gained", operator=">", threshold=1.0),
        ],
    )

    return [
        arena_train_100m,
        arena_train_2b,
        arena_eval,
        cvc_small_train_2b,
    ]
