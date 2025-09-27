"""Statistics processing helpers and policy-evaluator logging."""

import logging
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from metta.common.wandb.context import WandbRun
from metta.shared.eval_config import EvalResults
from mettagrid.profiling.stopwatch import Stopwatch
from mettagrid.util.dict_utils import unroll_nested_dict
from softmax.training.rl.evaluate import upload_replay_html
from softmax.training.rl.trainer_config import TrainerConfig
from softmax.training.rl.training import Experience
from softmax.training.rl.wandb import (
    POLICY_EVALUATOR_EPOCH_METRIC,
    POLICY_EVALUATOR_METRIC_PREFIX,
    POLICY_EVALUATOR_STEP_METRIC,
    setup_policy_evaluator_metrics,
)

logger = logging.getLogger(__name__)


def accumulate_rollout_stats(
    raw_infos: list,
    stats: dict[str, Any],
) -> None:
    """Accumulate rollout statistics from info dictionaries."""
    infos = defaultdict(list)

    # Batch process info dictionaries
    for i in raw_infos:
        for k, v in unroll_nested_dict(i):
            # Detach any tensors before accumulating to prevent memory leaks
            if torch.is_tensor(v):
                v = v.detach().cpu().item() if v.numel() == 1 else v.detach().cpu().numpy()
            elif isinstance(v, np.ndarray) and v.size == 1:
                v = v.item()
            infos[k].append(v)

    # Batch process stats
    for k, v in infos.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()

        if isinstance(v, list):
            stats.setdefault(k, []).extend(v)
        else:
            if k not in stats:
                stats[k] = v
            else:
                try:
                    stats[k] += v
                except TypeError:
                    stats[k] = [stats[k], v]  # fallback: bundle as list


def filter_movement_metrics(stats: dict[str, Any]) -> dict[str, Any]:
    """Filter movement metrics to only keep core values, removing derived stats."""
    filtered = {}

    # Core movement metrics we want to keep (without any suffix)
    # These will have the env_ prefix when passed to this function
    core_metrics = {
        "env_agent/movement.direction.up",
        "env_agent/movement.direction.down",
        "env_agent/movement.direction.left",
        "env_agent/movement.direction.right",
        "env_agent/movement.sequential_rotations",
        "env_agent/movement.rotation.to_up",
        "env_agent/movement.rotation.to_down",
        "env_agent/movement.rotation.to_left",
        "env_agent/movement.rotation.to_right",
    }

    for key, value in stats.items():
        # Check if this is a core metric (exact match)
        if key in core_metrics:
            filtered[key] = value
        # Skip any movement metric with derived stats suffixes
        elif key.startswith("env_agent/movement"):
            continue
        # Keep all non-movement metrics
        else:
            filtered[key] = value

    return filtered


def process_training_stats(
    raw_stats: dict[str, Any],
    losses_stats: dict[str, Any],
    experience: Experience,
    trainer_config: TrainerConfig,
) -> dict[str, Any]:
    """Process training statistics into a clean format.

    Args:
        raw_stats: Raw statistics dictionary (possibly with lists of values)
        losses_stats: Loss statistics dictionary
        experience: Experience object with stats() method
        trainer_config: Training configuration

    Returns:
        Dictionary with processed statistics including:
        - mean_stats: Raw stats converted to means
        - losses_stats: Loss statistics
        - experience_stats: Experience buffer statistics
        - environment_stats: Environment-specific stats
        - overview: High-level metrics like average reward
    """
    # Convert lists to means
    mean_stats = {}
    for k, v in raw_stats.items():
        try:
            mean_stats[k] = np.mean(v)
        except (TypeError, ValueError):
            mean_stats[k] = v

    # Get loss and experience statistics
    experience_stats = experience.stats() if hasattr(experience, "stats") else {}

    # Calculate environment statistics
    environment_stats = {
        f"env_{k.split('/')[0]}/{'/'.join(k.split('/')[1:])}": v for k, v in mean_stats.items() if "/" in k
    }

    # Filter movement metrics to only keep core values
    environment_stats = filter_movement_metrics(environment_stats)

    # Calculate overview statistics
    overview = {}

    # Calculate average reward from environment stats
    if "rewards" in experience_stats:
        overview["reward"] = experience_stats["rewards"]

    return {
        "mean_stats": mean_stats,
        "losses_stats": losses_stats,
        "experience_stats": experience_stats,
        "environment_stats": environment_stats,
        "overview": overview,
    }


def compute_timing_stats(
    timer: Stopwatch,
    agent_step: int,
) -> dict[str, Any]:
    """Compute timing statistics from a Stopwatch timer."""
    elapsed_times = timer.get_all_elapsed()
    wall_time = timer.get_elapsed()
    train_time = elapsed_times.get("_rollout", 0) + elapsed_times.get("_train", 0)

    lap_times = timer.lap_all(agent_step, exclude_global=False)
    wall_time_for_lap = lap_times.pop("global", 0)

    epoch_steps = timer.get_lap_steps()
    if epoch_steps is None:
        epoch_steps = 0

    epoch_steps_per_second = epoch_steps / wall_time_for_lap if wall_time_for_lap > 0 else 0
    steps_per_second = timer.get_rate(agent_step) if wall_time > 0 else 0

    timing_stats = {
        **{
            f"timing_per_epoch/frac/{op}": lap_elapsed / wall_time_for_lap if wall_time_for_lap > 0 else 0
            for op, lap_elapsed in lap_times.items()
        },
        **{
            f"timing_per_epoch/msec/{op}": lap_elapsed * 1000 if wall_time_for_lap > 0 else 0
            for op, lap_elapsed in lap_times.items()
        },
        "timing_per_epoch/sps": epoch_steps_per_second,
        **{
            f"timing_cumulative/frac/{op}": elapsed / wall_time if wall_time > 0 else 0
            for op, elapsed in elapsed_times.items()
        },
        "timing_cumulative/sps": steps_per_second,
    }

    return {
        "lap_times": lap_times,
        "elapsed_times": elapsed_times,
        "wall_time": wall_time,
        "train_time": train_time,
        "wall_time_for_lap": wall_time_for_lap,
        "epoch_steps": epoch_steps,
        "epoch_steps_per_second": epoch_steps_per_second,
        "steps_per_second": steps_per_second,
        "timing_stats": timing_stats,
    }


def process_policy_evaluator_stats(
    policy_uri: str, eval_results: EvalResults, run: WandbRun, epoch: int, agent_step: int, should_finish_run: bool
) -> None:
    metrics_to_log: dict[str, float] = {
        f"{POLICY_EVALUATOR_METRIC_PREFIX}/eval_{k}": v
        for k, v in eval_results.scores.to_wandb_metrics_format().items()
    }
    metrics_to_log.update(
        {
            f"overview/{POLICY_EVALUATOR_METRIC_PREFIX}/{category}_score": score
            for category, score in eval_results.scores.category_scores.items()
        }
    )
    if not metrics_to_log:
        logger.warning("No metrics to log for policy evaluator")
        return

    try:
        try:
            setup_policy_evaluator_metrics(run)
        except Exception:
            logger.warning("Failed to set default axes for policy evaluator metrics. Continuing")
            pass

        run.log({**metrics_to_log, POLICY_EVALUATOR_STEP_METRIC: agent_step, POLICY_EVALUATOR_EPOCH_METRIC: epoch})
        logger.info(f"Logged {len(metrics_to_log)} metrics to wandb for policy {policy_uri}")
        if eval_results.replay_urls:
            try:
                upload_replay_html(
                    replay_urls=eval_results.replay_urls,
                    agent_step=agent_step,  # type: ignore
                    epoch=epoch,  # type: ignore
                    wandb_run=run,
                    step_metric_key=POLICY_EVALUATOR_STEP_METRIC,
                    epoch_metric_key=POLICY_EVALUATOR_EPOCH_METRIC,
                )
            except Exception as e:
                logger.error(f"Failed to upload replays for {policy_uri}: {e}", exc_info=True)
    finally:
        if should_finish_run:
            run.finish()
