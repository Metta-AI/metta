"""Stable release suite - comprehensive tests for releases.

All tests use the "stable." prefix to assert they're running stable recipes.
Includes both CI smoke tests and long-running stable validation tests.
"""

from __future__ import annotations

from metta.jobs.job_config import AcceptanceCriterion, JobConfig, RemoteConfig
from recipes.validation.ci_suite import get_ci_jobs


def get_stable_jobs() -> list[JobConfig]:
    """Define stable release test jobs for recipes.

    These are comprehensive tests that validate releases on remote infrastructure.
    Includes CI smoke tests plus longer-running validation tests.
    """
    # Include all CI smoke tests
    ci_jobs, _group = get_ci_jobs()

    # ========================================
    # Arena Basic Easy Shaped - Stable Tests
    # ========================================

    # Single GPU training - 100M timesteps
    arena_train_100m = JobConfig(
        name="arena_single_gpu_100m",
        module="recipes.prod.arena_basic_easy_shaped.train",
        args=["trainer.total_timesteps=100000000"],
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
    arena_train_2b = JobConfig(
        name="arena_multi_gpu_2b",
        module="recipes.prod.arena_basic_easy_shaped.train",
        args=["trainer.total_timesteps=2000000000"],
        timeout_s=172800,  # 48 hours
        remote=RemoteConfig(gpus=4, nodes=4),
        is_training_job=True,
        metrics_to_track=["overview/sps", "env_agent/heart.gained"],
        acceptance_criteria=[
            AcceptanceCriterion(metric="overview/sps", operator=">=", threshold=80000),
            AcceptanceCriterion(metric="env_agent/heart.gained", operator=">", threshold=1.0),
        ],
    )

    # Evaluation - depends on single-GPU 100M training run
    arena_eval = JobConfig(
        name="arena_evaluate",
        module="recipes.prod.arena_basic_easy_shaped.evaluate",
        dependency_names=["arena_single_gpu_100m"],
        timeout_s=1800,
    )

    # ========================================
    # CvC Small Maps - Stable Tests
    # ========================================

    # Multi-GPU training - 2B timesteps
    cvc_small_train_2b = JobConfig(
        name="cvc_small_multi_gpu_2b",
        module="recipes.prod.cvc.small_maps.train",
        args=[
            "trainer.total_timesteps=2000000000",
            "num_cogs=4",
            'variants=["lonely_heart","heart_chorus","pack_rat","neutral_faced"]',
        ],
        timeout_s=172800,  # 48 hours
        remote=RemoteConfig(gpus=4, nodes=4),
        is_training_job=True,
        metrics_to_track=["overview/sps", "env_agent/heart.gained"],
        acceptance_criteria=[
            AcceptanceCriterion(metric="overview/sps", operator=">=", threshold=80000),
            AcceptanceCriterion(metric="env_agent/heart.gained", operator=">", threshold=1.0),
        ],
    )

    return ci_jobs + [
        arena_train_100m,
        arena_train_2b,
        arena_eval,
        cvc_small_train_2b,
    ]
