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
        module="recipes.prod.arena_basic_easy_shaped.train",
        args=[
            f"run={arena_train_name}",
            "trainer.total_timesteps=100000000",
            "evaluator.evaluate_local=True",
            "evaluator.evaluate_remote=False",
        ],
        timeout_s=7200,
        remote=RemoteConfig(gpus=1, nodes=1),
        is_training_job=True,
        metrics_to_track=["overview/sps", "env_game/assembler.heart.created"],
        acceptance_criteria=[
            AcceptanceCriterion(metric="overview/sps", operator=">=", threshold=40000),
            AcceptanceCriterion(metric="env_game/assembler.heart.created", operator=">", threshold=0.1),
        ],
    )

    # Multi-GPU training - 2B timesteps
    arena_train_2b_name = f"{prefix}.arena_multi_gpu_2b"
    arena_train_2b = JobConfig(
        name=arena_train_2b_name,
        module="recipes.prod.arena_basic_easy_shaped.train",
        args=[
            f"run={arena_train_2b_name}",
            "trainer.total_timesteps=2000000000",
            "evaluator.evaluate_local=True",
            "evaluator.evaluate_remote=False",
        ],
        timeout_s=172800,
        remote=RemoteConfig(gpus=4, nodes=4),
        is_training_job=True,
        metrics_to_track=["overview/sps", "env_game/assembler.heart.created"],
        acceptance_criteria=[
            AcceptanceCriterion(metric="overview/sps", operator=">=", threshold=80000),
            AcceptanceCriterion(metric="env_game/assembler.heart.created", operator=">", threshold=1.0),
        ],
    )

    # Evaluation - depends on single-GPU 100M training run
    arena_eval_name = f"{prefix}.arena_evaluate"
    arena_eval = JobConfig(
        name=arena_eval_name,
        module="recipes.prod.arena_basic_easy_shaped.evaluate",
        args=[f'policy_uris=["s3://softmax-public/policies/{arena_train_name}:latest"]'],
        dependency_names=[arena_train_name],
        timeout_s=1800,
        metrics_to_track=["heart_delta_pct"],
        acceptance_criteria=[
            AcceptanceCriterion(metric="heart_delta_pct", operator=">=", threshold=0),
        ],
    )

    # ========================================
    # CvC Fixed Maps - Stable Tests
    # ========================================

    # 200-epoch mettabox sanity check (~105M timesteps)
    cvc_fixed_maps_200ep_name = f"{prefix}.cvc_fixed_maps_mettabox_200ep"
    cvc_fixed_maps_200ep_timesteps = 200 * 524_288  # 200 epochs * default batch size
    cvc_fixed_maps_train_200ep = JobConfig(
        name=cvc_fixed_maps_200ep_name,
        module="recipes.experiment.cogs_v_clips.train",
        args=[
            f"run={cvc_fixed_maps_200ep_name}",
            f"trainer.total_timesteps={cvc_fixed_maps_200ep_timesteps}",
            "num_cogs=4",
            'variants=["lonely_heart","heart_chorus","pack_rat"]',
            "evaluator.evaluate_local=True",
            "evaluator.evaluate_remote=False",
        ],
        timeout_s=43200,
        remote=RemoteConfig(gpus=1, nodes=1),
        is_training_job=True,
        metrics_to_track=["overview/sps"],
        acceptance_criteria=[AcceptanceCriterion(metric="overview/sps", operator=">=", threshold=30000)],
    )

    # Multi-GPU training - 2B timesteps
    cvc_fixed_maps_train_name = f"{prefix}.cvc_fixed_maps_multi_gpu_2b"
    cvc_fixed_maps_train_2b = JobConfig(
        name=cvc_fixed_maps_train_name,
        module="recipes.experiment.cogs_v_clips.train",
        args=[
            f"run={cvc_fixed_maps_train_name}",
            "trainer.total_timesteps=2000000000",
            "num_cogs=4",
            'variants=["lonely_heart","heart_chorus","pack_rat"]',
            "evaluator.evaluate_local=True",
            "evaluator.evaluate_remote=False",
        ],
        timeout_s=172800,
        remote=RemoteConfig(gpus=4, nodes=4),
        is_training_job=True,
        metrics_to_track=["overview/sps"],
        acceptance_criteria=[AcceptanceCriterion(metric="overview/sps", operator=">=", threshold=80000)],
    )

    return [
        arena_train_100m,
        arena_train_2b,
        arena_eval,
        cvc_fixed_maps_train_200ep,
        cvc_fixed_maps_train_2b,
    ]
