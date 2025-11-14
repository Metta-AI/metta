"""CI test suite - lightweight tests that run on every commit."""

from __future__ import annotations

import os
from datetime import datetime

from metta.jobs.job_config import JobConfig


def get_user_timestamp() -> str:
    """Get a timestamped group name for this CI run with username to avoid collisions."""
    user = os.environ.get("USER", "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{user}.{timestamp}"


def get_ci_jobs(prefix: str | None = None) -> tuple[list[JobConfig], str]:
    """Define CI test jobs for recipes.

    These are lightweight smoke tests that run quickly on every commit.

    Args:
        version: Version prefix for job names (e.g., "v0.1.0" for stable, or None for timestamped)

    Returns:
        Tuple of (job configs, group name for this CI run)
    """
    group = prefix if prefix else get_user_timestamp()

    arena_train_name = f"{group}.arena_train"
    arena_eval_name = f"{group}.arena_eval"
    arena_play_name = f"{group}.arena_play"
    cvc_small_train_name = f"{group}.cvc_small_train"
    cvc_small_play_name = f"{group}.cvc_small_play"
    cogames_train_name = f"{group}.cogames_train"
    cogames_eval_name = f"{group}.cogames_eval"

    arena_train = JobConfig(
        name=arena_train_name,
        tool="recipes.prod.arena_basic_easy_shaped.train",
        args={
            "run": arena_train_name,
            "trainer.total_timesteps": "10000",
            "checkpointer.epoch_interval": "1",
        },
        timeout_s=300,
        group=group,
    )

    # Evaluate the trained policy from the training run
    # TODO: make this use s3 and not local file when github ci perms are set to be able to fetch from s3
    # policy_uri = "s3://softmax-public/policies/{arena_train_name}:latest"
    arena_eval = JobConfig(
        name=arena_eval_name,
        tool="recipes.prod.arena_basic_easy_shaped.evaluate_latest_in_dir",
        args={"dir_path": f"./train_dir/{arena_train_name}/checkpoints/"},
        dependency_names=[arena_train_name],
        timeout_s=300,
        group=group,
    )

    # Play test with random policy (run with minimal steps)
    arena_play = JobConfig(
        name=arena_play_name,
        tool="recipes.prod.arena_basic_easy_shaped.play",
        args={"max_steps": "100", "render": "log", "open_browser_on_start": "False"},  # Headless mode for CI
        timeout_s=60,
        group=group,  # Tag with group for monitoring
    )

    # CvC Small Maps - Train just enough to get a single checkpoint
    cvc_small_train = JobConfig(
        name=cvc_small_train_name,
        tool="recipes.prod.cvc.small_maps.train",
        args={
            "run": cvc_small_train_name,
            "trainer.total_timesteps": "10000",
            "checkpointer.epoch_interval": "1",
            "num_cogs": "4",
            "variants": '["lonely_heart","heart_chorus","pack_rat","neutral_faced"]',
        },
        timeout_s=300,
        group=group,
    )

    # CvC Small Maps - Play test with random policy
    cvc_small_play = JobConfig(
        name=cvc_small_play_name,
        tool="recipes.prod.cvc.small_maps.play",
        args={"max_steps": "100", "render": "log", "open_browser_on_start": "False"},  # Headless mode for CI
        timeout_s=60,
        group=group,  # Tag with group for monitoring
    )

    # CoGames - Train and evaluate
    cogames_train = JobConfig(
        name=cogames_train_name,
        cmd=f"devops/stable/cogames_train_eval.py --mission cogs_vs_clips --variant mini --steps 1000 --checkpoints-dir ./train_dir/{cogames_train_name}/checkpoints --eval-episodes 5 --artifacts '{{}}'",
        timeout_s=300,
        group=group,
    )

    return [
        arena_train,
        arena_eval,
        arena_play,
        cvc_small_train,
        cvc_small_play,
        cogames_train,
    ], group
