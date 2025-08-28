"""Statistics processing functions for Metta training."""

import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from metta.agent.agent_config import AgentConfig
from metta.agent.metta_agent import PolicyAgent
from metta.common.profiling.memory_monitor import MemoryMonitor
from metta.common.profiling.stopwatch import Stopwatch
from metta.common.util.system_monitor import SystemMonitor
from metta.common.wandb.wandb_context import WandbRun
from metta.eval.eval_request_config import EvalRewardSummary
from metta.mettagrid.util.dict_utils import unroll_nested_dict
from metta.rl.experience import Experience
from metta.rl.kickstarter import Kickstarter
from metta.rl.losses import Losses
from metta.rl.trainer_config import TrainerConfig
from metta.rl.utils import should_run

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
    losses: Losses,
    experience: Experience,
    trainer_config: TrainerConfig,
    kickstarter: Kickstarter | None,
) -> dict[str, Any]:
    """Process training statistics into a clean format."""
    # Convert lists to means
    mean_stats = {}
    for k, v in raw_stats.items():
        try:
            mean_stats[k] = np.mean(v)
        except (TypeError, ValueError):
            mean_stats[k] = v

    # Get loss and experience statistics
    losses_stats = losses.stats() if hasattr(losses, "stats") else {}
    experience_stats = experience.stats() if hasattr(experience, "stats") else {}

    # Remove unused losses
    if trainer_config.ppo.l2_reg_loss_coef == 0:
        losses_stats.pop("l2_reg_loss", None)
    if trainer_config.ppo.l2_init_loss_coef == 0:
        losses_stats.pop("l2_init_loss", None)
    if kickstarter is None or not kickstarter.enabled:
        losses_stats.pop("ks_action_loss", None)
        losses_stats.pop("ks_value_loss", None)

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
    losses: Losses,
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
    optimizer: torch.optim.Optimizer,
    latest_saved_policy_record: None = None,
    kickstarter: Kickstarter | None = None,
) -> None:
    """Process and log training statistics."""
    if not wandb_run:
        return

    # Process training stats
    processed_stats = process_training_stats(
        raw_stats=stats,
        losses=losses,
        experience=experience,
        trainer_config=trainer_cfg,
        kickstarter=kickstarter,
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
        "latest_saved_policy_epoch": 0,  # Not used with SimpleCheckpointManager
    }

    # Get system stats - note: can impact performance
    system_stats = system_monitor.stats()
    memory_stats = memory_monitor.stats()

    # Current hyperparameter values (after potential scheduler updates)
    hyperparameters = {
        "learning_rate": parameters["learning_rate"],
        "ppo_clip_coef": trainer_cfg.ppo.clip_coef,
        "ppo_vf_clip_coef": trainer_cfg.ppo.vf_clip_coef,
        "ppo_ent_coef": trainer_cfg.ppo.ent_coef,
        "ppo_l2_reg_loss_coef": trainer_cfg.ppo.l2_reg_loss_coef,
        "ppo_l2_init_loss_coef": trainer_cfg.ppo.l2_init_loss_coef,
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
