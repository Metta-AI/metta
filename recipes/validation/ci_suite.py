"""CI test suite - lightweight tests that run on every commit."""

from __future__ import annotations

import os
from datetime import datetime

from metta.jobs.job_config import JobConfig


def get_ci_jobs() -> tuple[list[JobConfig], str]:
    """Define CI test jobs for recipes.

    These are lightweight smoke tests that run quickly on every commit.

    Returns:
        Tuple of (job configs, group name for this CI run)
    """
    # Create timestamped group name for this CI run with username to avoid collisions
    user = os.environ.get("USER", "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    group = f"{user}.ci.{timestamp}"
    run_prefix = f"{user}.ci.{timestamp}"

    # Version job names with timestamp to avoid collisions on reruns
    arena_train_name = f"{run_prefix}.arena_train"
    arena_eval_name = f"{run_prefix}.arena_eval"
    arena_play_name = f"{run_prefix}.arena_play"
    cvc_small_train_name = f"{run_prefix}.cvc_small_train"
    cvc_small_play_name = f"{run_prefix}.cvc_small_play"

    # Train just enough to get a single checkpoint
    # With checkpoint every epoch, this ensures at least 1 checkpoint
    arena_train = JobConfig(
        name=arena_train_name,
        module="recipes.prod.arena_basic_easy_shaped.train",
        args=[
            f"run={run_prefix}.arena_train",
            "trainer.total_timesteps=10000",  # Train for 10k timesteps - enough for checkpoint
            "checkpointer.epoch_interval=1",  # Checkpoint every epoch
        ],
        timeout_s=300,  # 5 minutes should be plenty
        is_training_job=True,
        group=group,  # Tag with group for monitoring
    )

    # Evaluate the trained policy
    arena_eval = JobConfig(
        name=arena_eval_name,
        module="recipes.prod.arena_basic_easy_shaped.evaluate",
        dependency_names=[arena_train_name],
        timeout_s=300,
        group=group,  # Tag with group for monitoring
    )

    # Play test with random policy (run with minimal steps)
    arena_play = JobConfig(
        name=arena_play_name,
        module="recipes.prod.arena_basic_easy_shaped.play",
        args=["max_steps=100", "render=log", "open_browser_on_start=False"],  # Headless mode for CI
        timeout_s=60,
        group=group,  # Tag with group for monitoring
    )

    # CvC Small Maps - Train just enough to get a single checkpoint
    cvc_small_train = JobConfig(
        name=cvc_small_train_name,
        module="recipes.prod.cvc.small_maps.train",
        args=[
            f"run={run_prefix}.cvc_small_train",
            "trainer.total_timesteps=10000",  # Train for 10k timesteps - enough for checkpoint
            "checkpointer.epoch_interval=1",  # Checkpoint every epoch
            "num_cogs=4",
            'variants=["lonely_heart","heart_chorus","pack_rat","neutral_faced"]',
        ],
        timeout_s=300,  # 5 minutes should be plenty
        is_training_job=True,
        group=group,  # Tag with group for monitoring
    )

    # CvC Small Maps - Play test with random policy
    cvc_small_play = JobConfig(
        name=cvc_small_play_name,
        module="recipes.prod.cvc.small_maps.play",
        args=["max_steps=100", "render=log", "open_browser_on_start=False"],  # Headless mode for CI
        timeout_s=60,
        group=group,  # Tag with group for monitoring
    )

    return [
        arena_train,
        arena_eval,
        arena_play,
        cvc_small_train,
        cvc_small_play,
    ], group
