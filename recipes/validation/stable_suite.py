"""Stable release suite - comprehensive tests for releases.

Includes both CI smoke tests and long-running stable validation tests.
Job names are fully qualified with user/timestamp and version.
"""

from __future__ import annotations

from metta.jobs.job_config import AcceptanceCriterion, JobConfig, RemoteConfig


def get_stable_jobs(prefix: str) -> list[JobConfig]:
    """Define stable-specific test jobs for recipes.

    These are comprehensive long-running tests that validate releases on remote infrastructure.
    Does NOT include CI smoke tests - those are combined in devops/stable/jobs.py.

    Args:
        prefix: Prefix for job names (e.g., "stable.v0.1.0")

    Returns:
        List of fully configured job configs with version-prefixed names
    """
    # ========================================
    # Arena Basic Easy Shaped - Stable Tests
    # ========================================

    # Single GPU training - 100M timesteps
    arena_train_name = f"{prefix}.arena_single_gpu_100m"
    arena_train_100m = JobConfig(
        name=arena_train_name,
        recipe="recipes.prod.arena_basic_easy_shaped.train",
        args={
            "run": arena_train_name,
            "trainer.total_timesteps": "100000000",
        },
        timeout_s=7200,
        remote=RemoteConfig(gpus=1, nodes=1),
        metrics_to_track=["overview/sps", "env_agent/heart.gained"],
        acceptance_criteria=[
            AcceptanceCriterion(metric="overview/sps", operator=">=", threshold=40000),
            AcceptanceCriterion(metric="env_agent/heart.gained", operator=">", threshold=0.1),
        ],
    )

    # Evaluation for single-GPU 100M training run
    arena_eval_100m_name = f"{prefix}.arena_evaluate_100m"
    arena_eval_100m = JobConfig(
        name=arena_eval_100m_name,
        recipe="recipes.prod.arena_basic_easy_shaped.evaluate",
        args={"policy_uris": f'["s3://softmax-public/policies/{arena_train_name}:latest"]'},
        dependency_names=[arena_train_name],
        timeout_s=1800,
    )

    # Multi-GPU training - 2B timesteps
    arena_train_2b_name = f"{prefix}.arena_multi_gpu_2b"
    arena_train_2b = JobConfig(
        name=arena_train_2b_name,
        recipe="recipes.prod.arena_basic_easy_shaped.train",
        args={
            "run": arena_train_2b_name,
            "trainer.total_timesteps": "2000000000",
        },
        timeout_s=172800,
        remote=RemoteConfig(gpus=4, nodes=4),
        metrics_to_track=["overview/sps", "env_agent/heart.gained"],
        acceptance_criteria=[
            AcceptanceCriterion(metric="overview/sps", operator=">=", threshold=80000),
            AcceptanceCriterion(metric="env_agent/heart.gained", operator=">", threshold=1.0),
        ],
    )

    # Evaluation for multi-GPU 2B training run
    arena_eval_2b_name = f"{prefix}.arena_evaluate_2b"
    arena_eval_2b = JobConfig(
        name=arena_eval_2b_name,
        recipe="recipes.prod.arena_basic_easy_shaped.evaluate",
        args={"policy_uris": f'["s3://softmax-public/policies/{arena_train_2b_name}:latest"]'},
        dependency_names=[arena_train_2b_name],
        timeout_s=1800,
    )

    # ========================================
    # CvC Small Maps - Stable Tests
    # ========================================

    # Multi-GPU training - 2B timesteps
    cvc_small_train_name = f"{prefix}.cvc_small_multi_gpu_2b"
    cvc_small_train_2b = JobConfig(
        name=cvc_small_train_name,
        recipe="recipes.prod.cvc.small_maps.train",
        args={
            "run": cvc_small_train_name,
            "trainer.total_timesteps": "2000000000",
            "num_cogs": "4",
            "variants": '["lonely_heart","heart_chorus","pack_rat","neutral_faced"]',
        },
        timeout_s=172800,
        remote=RemoteConfig(gpus=4, nodes=4),
        metrics_to_track=["overview/sps", "env_agent/heart.gained"],
        acceptance_criteria=[AcceptanceCriterion(metric="overview/sps", operator=">=", threshold=80000)],
    )

    # Evaluation for CvC multi-GPU 2B training run
    cvc_eval_2b_name = f"{prefix}.cvc_evaluate_2b"
    cvc_eval_2b = JobConfig(
        name=cvc_eval_2b_name,
        recipe="recipes.prod.cvc.small_maps.evaluate",
        args={"policy_uris": f'["s3://softmax-public/policies/{cvc_small_train_name}:latest"]'},
        dependency_names=[cvc_small_train_name],
        timeout_s=1800,
    )

    # ========================================
    # CoGames - Stable Tests
    # ========================================

    # CoGames training - 100k steps with S3 upload
    # Config derived from get_stable_train_config() in recipes/prod/cogames.py
    cogames_train_name = f"{prefix}.cogames_train_100k"
    cogames_s3_uri = f"s3://softmax-public/cogames/{cogames_train_name}/checkpoint.pt"
    cogames_train_100k = JobConfig(
        name=cogames_train_name,
        recipe="recipes.prod.cogames.train",
        args={
            "run": cogames_train_name,
            "mission": "training_facility.harvest",
            "variant": ["standard"],
            "steps": "100000",
            "checkpoints": f"/tmp/{cogames_train_name}",
            "s3_uri": cogames_s3_uri,
        },
        timeout_s=3600,
        remote=RemoteConfig(gpus=1, nodes=1),
    )

    # CoGames evaluation - downloads from S3 and evaluates
    # Config derived from get_stable_eval_config() in recipes/prod/cogames.py
    cogames_eval_name = f"{prefix}.cogames_eval_100k"
    cogames_eval_100k = JobConfig(
        name=cogames_eval_name,
        recipe="recipes.prod.cogames.evaluate",
        args={
            "mission": "training_facility.harvest",
            "variant": ["standard"],
            "policy_uri": cogames_s3_uri,
            "episodes": "20",
        },
        dependency_names=[cogames_train_name],
        timeout_s=1800,
    )

    return [
        arena_train_100m,
        arena_eval_100m,
        arena_train_2b,
        arena_eval_2b,
        cvc_small_train_2b,
        cvc_eval_2b,
        cogames_train_100k,
        cogames_eval_100k,
    ]
