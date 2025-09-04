"""Statistics processing functions for Metta training."""

import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import wandb

from metta.agent.agent_config import AgentConfig
from metta.agent.metta_agent import PolicyAgent
from metta.common.util.constants import METTA_WANDB_ENTITY, METTA_WANDB_PROJECT
from metta.common.wandb.wandb_context import WandbRun
from metta.eval.eval_request_config import EvalResults, EvalRewardSummary
from metta.mettagrid.profiling.memory_monitor import MemoryMonitor
from metta.mettagrid.profiling.stopwatch import Stopwatch
from metta.mettagrid.profiling.system_monitor import SystemMonitor
from metta.mettagrid.util.dict_utils import unroll_nested_dict
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.evaluate import upload_replay_html
from metta.rl.experience import Experience
from metta.rl.trainer_config import TrainerConfig
from metta.rl.utils import should_run
from metta.rl.wandb import (
    POLICY_EVALUATOR_EPOCH_METRIC,
    POLICY_EVALUATOR_METRIC_PREFIX,
    POLICY_EVALUATOR_STEP_METRIC,
    setup_policy_evaluator_metrics,
)

logger = logging.getLogger(__name__)


@dataclass
class StatsTracker:
    """Manages training statistics and database tracking."""

    # Rollout stats collected during episodes
    rollout_stats: dict[str, Any] = field(default_factory=dict)

    # Gradient statistics (computed periodically)
    grad_stats: dict[str, float] = field(default_factory=dict)

    # Database tracking for stats service
    stats_epoch_start: int = 0
    stats_epoch_id: uuid.UUID | None = None
    stats_run_id: uuid.UUID | None = None

    def clear_rollout_stats(self) -> None:
        """Clear rollout stats after processing."""
        self.rollout_stats.clear()

    def clear_grad_stats(self) -> None:
        """Clear gradient stats after processing."""
        self.grad_stats.clear()

    def update_epoch_tracking(self, new_epoch_start: int) -> None:
        """Update epoch tracking after creating a new stats epoch."""
        self.stats_epoch_start = new_epoch_start


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
    task_reward_values = [v for k, v in environment_stats.items() if k.startswith("env_task_reward")]
    if task_reward_values:
        mean_reward = sum(task_reward_values) / len(task_reward_values)
        overview["reward"] = mean_reward

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


def build_wandb_stats(
    processed_stats: dict[str, Any],
    timing_info: dict[str, Any],
    weight_stats: dict[str, Any],
    grad_stats: dict[str, Any],
    system_stats: dict[str, Any],
    memory_stats: dict[str, Any],
    parameters: dict[str, Any],
    hyperparameters: dict[str, Any],
    evals: EvalRewardSummary,
    agent_step: int,
    epoch: int,
) -> dict[str, Any]:
    """Build complete statistics dictionary for wandb logging."""
    # Build overview with sps and rewards
    overview = {
        "sps": timing_info["epoch_steps_per_second"],
        **processed_stats["overview"],
    }

    # Add evaluation scores to overview
    for category, score in evals.category_scores.items():
        overview[f"{category}_score"] = score

    # Also add reward_vs_total_time if we have reward
    if "reward" in overview:
        overview["reward_vs_total_time"] = overview["reward"]

    # X-axis values for wandb
    metric_stats = {
        "metric/agent_step": agent_step,
        "metric/epoch": epoch,
        "metric/total_time": timing_info["wall_time"],
        "metric/train_time": timing_info["train_time"],
    }

    # Combine all stats
    return {
        **{f"overview/{k}": v for k, v in overview.items()},
        **{f"losses/{k}": v for k, v in processed_stats["losses_stats"].items()},
        **{f"experience/{k}": v for k, v in processed_stats["experience_stats"].items()},
        **{f"parameters/{k}": v for k, v in parameters.items()},
        **{f"hyperparameters/{k}": v for k, v in hyperparameters.items()},
        **{f"eval_{k}": v for k, v in evals.to_wandb_metrics_format().items()},
        **system_stats,  # Already has monitor/ prefix from SystemMonitor.stats()
        **{f"trainer_memory/{k}": v for k, v in memory_stats.items()},
        **processed_stats["environment_stats"],
        **weight_stats,
        **timing_info["timing_stats"],
        **metric_stats,
        **grad_stats,
    }


def process_stats(
    stats: dict[str, Any],
    losses_stats: dict[str, Any],
    evals: EvalRewardSummary,
    grad_stats: dict[str, float],
    experience: Experience,
    policy: PolicyAgent,
    timer: Stopwatch,
    trainer_cfg: TrainerConfig,
    agent_cfg: AgentConfig,
    agent_step: int,
    epoch: int,
    wandb_run: WandbRun | None,
    memory_monitor: MemoryMonitor,
    system_monitor: SystemMonitor,
    latest_saved_epoch: int,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Process and log training statistics."""
    if not wandb_run:
        return

    # Process training stats
    processed_stats = process_training_stats(
        raw_stats=stats,
        losses_stats=losses_stats,
        experience=experience,
        trainer_config=trainer_cfg,
    )

    # Compute timing stats
    timing_info = compute_timing_stats(
        timer=timer,
        agent_step=agent_step,
    )

    # Compute weight stats if configured
    weight_stats = {}
    if hasattr(agent_cfg, "analyze_weights_interval"):
        if should_run(epoch, agent_cfg.analyze_weights_interval):
            for metrics in policy.compute_weight_metrics():
                name = metrics.get("name", "unknown")
                for key, value in metrics.items():
                    if key != "name":
                        weight_stats[f"weights/{key}/{name}"] = value

    # Build parameters
    parameters = {
        "learning_rate": optimizer.param_groups[0]["lr"] if optimizer else trainer_cfg.optimizer.learning_rate,
        "epoch_steps": timing_info["epoch_steps"],
        "num_minibatches": experience.num_minibatches,
        "latest_saved_policy_epoch": latest_saved_epoch,
    }

    # Get system stats - note: can impact performance
    system_stats = system_monitor.stats()
    memory_stats = memory_monitor.stats()

    # Current hyperparameter values (after potential scheduler updates)
    # TODO: please don't hardcode PPO-specific hyperparameters.
    hyperparameters = {
        "learning_rate": parameters["learning_rate"],
        "ppo_clip_coef": trainer_cfg.losses.loss_configs["ppo"].clip_coef,
        "ppo_vf_clip_coef": trainer_cfg.losses.loss_configs["ppo"].vf_clip_coef,
        "ppo_ent_coef": trainer_cfg.losses.loss_configs["ppo"].ent_coef,
        "ppo_l2_reg_loss_coef": trainer_cfg.losses.loss_configs["ppo"].l2_reg_loss_coef,
        "ppo_l2_init_loss_coef": trainer_cfg.losses.loss_configs["ppo"].l2_init_loss_coef,
    }

    # Build complete stats
    all_stats = build_wandb_stats(
        processed_stats=processed_stats,
        timing_info=timing_info,
        weight_stats=weight_stats,
        grad_stats=grad_stats,
        system_stats=system_stats,
        memory_stats=memory_stats,
        hyperparameters=hyperparameters,
        parameters=parameters,
        evals=evals,
        agent_step=agent_step,
        epoch=epoch,
    )

    # Log to wandb
    wandb_run.log(all_stats, step=agent_step)


def process_policy_evaluator_stats(
    policy_uri: str,
    eval_results: EvalResults,
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

    # Policy records might not have epoch/agent_step metadata, but we still want to log
    metadata = CheckpointManager.get_policy_metadata(policy_uri)
    epoch = metadata.get("epoch")
    agent_step = metadata.get("agent_step")
    run_name = metadata.get("run_name")
    if epoch is None or agent_step is None or not run_name:
        logger.warning("No epoch or agent_step found in policy record - using defaults")

    # Sanitize run_name for wandb - remove version suffix and invalid characters
    # WandB run IDs cannot contain: :;,#?/'
    sanitized_run_name = run_name.split(":")[0] if run_name else None

    # TODO: improve this parsing to be more general
    run = wandb.init(
        id=sanitized_run_name,
        project=METTA_WANDB_PROJECT,
        entity=METTA_WANDB_ENTITY,
        resume="must",
    )
    try:
        try:
            setup_policy_evaluator_metrics(run)
        except Exception:
            logger.warning("Failed to set default axes for policy evaluator metrics. Continuing")
            pass

        run.log(
            {**metrics_to_log, POLICY_EVALUATOR_STEP_METRIC: agent_step or 0, POLICY_EVALUATOR_EPOCH_METRIC: epoch or 0}
        )
        logger.info(f"Logged {len(metrics_to_log)} metrics to wandb for policy {policy_uri}")
        if eval_results.replay_urls:
            try:
                upload_replay_html(
                    replay_urls=eval_results.replay_urls,
                    agent_step=agent_step,  # type: ignore
                    epoch=epoch,  # type: ignore
                    wandb_run=run,
                    metric_prefix=POLICY_EVALUATOR_METRIC_PREFIX,
                    step_metric_key=POLICY_EVALUATOR_STEP_METRIC,
                    epoch_metric_key=POLICY_EVALUATOR_EPOCH_METRIC,
                )
            except Exception as e:
                logger.error(f"Failed to upload replays for {policy_uri}: {e}", exc_info=True)
    finally:
        run.finish()
