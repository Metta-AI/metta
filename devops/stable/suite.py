"""Job definitions for CI and stable validation."""

from __future__ import annotations

import os
from datetime import datetime

from devops.stable.runner import AcceptanceCriterion, Job, Operator, create_job


def get_user_timestamp() -> str:
    user = os.environ.get("USER", "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{user}.{timestamp}"


def get_ci_jobs(prefix: str | None = None) -> tuple[list[Job], str]:
    """Get CI jobs for recipe smoke tests."""
    group = prefix if prefix else get_user_timestamp()

    arena_train_name = f"{group}.arena_train"
    cvc_train_name = f"{group}.cvc_train"

    jobs = [
        create_job(
            name=arena_train_name,
            cmd=[
                "uv",
                "run",
                "./tools/run.py",
                "train",
                "arena",
                f"run={arena_train_name}",
                "trainer.total_timesteps=100",
                "checkpointer.epoch_interval=1",
                "training_env.forward_pass_minibatch_target_size=96",
                "training_env.vectorization=serial",
                "evaluator.evaluate_local=False",
                "evaluator.evaluate_remote=False",
            ],
            timeout_s=300,
            wandb_run_name=arena_train_name,
        ),
        create_job(
            name=f"{group}.arena_eval",
            cmd=[
                "uv",
                "run",
                "./tools/run.py",
                "evaluate",
                "arena",
                f"dir_path=./train_dir/{arena_train_name}/checkpoints/",
            ],
            timeout_s=300,
            dependencies=[arena_train_name],
        ),
        create_job(
            name=f"{group}.arena_play",
            cmd=[
                "uv",
                "run",
                "./tools/run.py",
                "play",
                "arena",
                "max_steps=10",
                "render=log",
                "open_browser_on_start=False",
            ],
            timeout_s=120,
        ),
        create_job(
            name=cvc_train_name,
            cmd=[
                "uv",
                "run",
                "./tools/run.py",
                "cvc.single_mission.train",
                f"run={cvc_train_name}",
                "mission=easy_hearts",
                "trainer.total_timesteps=64",
                "trainer.minibatch_size=8",
                "trainer.batch_size=64",
                "trainer.bptt_horizon=8",
                "trainer.update_epochs=1",
                "training_env.forward_pass_minibatch_target_size=8",
                "training_env.vectorization=serial",
                "training_env.auto_workers=False",
                "training_env.num_workers=1",
                "training_env.async_factor=1",
                "training_env.maps_cache_size=4",
                "evaluator.epoch_interval=0",
                "evaluator.evaluate_local=False",
                "evaluator.evaluate_remote=False",
                "checkpointer.epoch_interval=1",
                "wandb.enabled=False",
                "num_cogs=2",
                'variants=["lonely_heart","heart_chorus","pack_rat"]',
            ],
            timeout_s=240,
        ),
        create_job(
            name=f"{group}.cvc_play",
            cmd=[
                "uv",
                "run",
                "./tools/run.py",
                "cvc.single_mission.play",
                "max_steps=10",
                "render=log",
                "open_browser_on_start=False",
            ],
            timeout_s=120,
        ),
    ]

    return jobs, group


def get_stable_jobs(prefix: str) -> list[Job]:
    """Get stable validation jobs (long-running GPU training)."""
    arena_100m_name = f"{prefix}.arena_single_gpu_100m"
    arena_2b_name = f"{prefix}.arena_multi_gpu_2b"
    cvc_200ep_name = f"{prefix}.cvc_fixed_maps_mettabox_200ep"
    cvc_2b_name = f"{prefix}.cvc_fixed_maps_multi_gpu_2b"

    return [
        # Arena single GPU - 100M timesteps
        create_job(
            name=arena_100m_name,
            cmd=[
                "uv",
                "run",
                "./tools/run.py",
                "train",
                "arena",
                f"run={arena_100m_name}",
                "trainer.total_timesteps=100000000",
            ],
            timeout_s=7200,
            gpus=1,
            wandb_run_name=arena_100m_name,
            acceptance=[
                AcceptanceCriterion(metric="overview/sps", threshold=40000),
                AcceptanceCriterion(metric="env_agent/heart.gained", operator=Operator.GT, threshold=0.1),
            ],
        ),
        # Arena multi GPU - 2B timesteps
        create_job(
            name=arena_2b_name,
            cmd=[
                "uv",
                "run",
                "./tools/run.py",
                "train",
                "arena",
                f"run={arena_2b_name}",
                "trainer.total_timesteps=2000000000",
            ],
            timeout_s=172800,
            gpus=4,
            nodes=4,
            wandb_run_name=arena_2b_name,
            acceptance=[
                AcceptanceCriterion(metric="overview/sps", threshold=80000),
                AcceptanceCriterion(metric="env_agent/heart.gained", operator=Operator.GT, threshold=1.0),
            ],
        ),
        # Arena evaluation - depends on 100M training
        create_job(
            name=f"{prefix}.arena_evaluate",
            cmd=[
                "uv",
                "run",
                "./tools/run.py",
                "evaluate",
                "arena",
                f'policy_uris=["s3://softmax-public/policies/{arena_100m_name}:latest"]',
            ],
            timeout_s=1800,
            dependencies=[arena_100m_name],
        ),
        # CvC 200 epochs (~105M timesteps)
        create_job(
            name=cvc_200ep_name,
            cmd=[
                "uv",
                "run",
                "./tools/run.py",
                "cvc.single_mission.train",
                f"run={cvc_200ep_name}",
                f"trainer.total_timesteps={200 * 524288}",
                "num_cogs=4",
                'variants=["lonely_heart","heart_chorus","pack_rat"]',
            ],
            timeout_s=43200,
            gpus=1,
            wandb_run_name=cvc_200ep_name,
            acceptance=[
                AcceptanceCriterion(metric="overview/sps", threshold=40000),
            ],
        ),
        # CvC multi GPU - 2B timesteps
        create_job(
            name=cvc_2b_name,
            cmd=[
                "uv",
                "run",
                "./tools/run.py",
                "cvc.single_mission.train",
                f"run={cvc_2b_name}",
                "trainer.total_timesteps=2000000000",
                "num_cogs=4",
                'variants=["lonely_heart","heart_chorus","pack_rat"]',
            ],
            timeout_s=172800,
            gpus=4,
            nodes=4,
            wandb_run_name=cvc_2b_name,
            acceptance=[
                AcceptanceCriterion(metric="overview/sps", threshold=80000),
            ],
        ),
    ]


def get_all_jobs(prefix: str) -> list[Job]:
    """Get all validation jobs (CI + stable)."""
    ci_jobs, _ = get_ci_jobs(prefix)
    return ci_jobs + get_stable_jobs(prefix)
