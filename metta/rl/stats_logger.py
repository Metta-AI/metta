"""Statistics logging and tracking components."""

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import wandb

from metta.rl.experience import Experience
from metta.rl.losses import Losses
from metta.util.system_monitor import SystemMonitor
from metta.util.wandb.wandb_context import WandbRun
from mettagrid.util.stopwatch import Stopwatch

if TYPE_CHECKING:
    from metta.agent import BaseAgent

logger = logging.getLogger(__name__)


class StatsLogger:
    """Handles statistics collection and logging during training.

    This class manages collecting training statistics, processing them,
    and logging to wandb or other backends.
    """

    def __init__(
        self,
        wandb_run: Optional[WandbRun] = None,
        is_master: bool = True,
        world_size: int = 1,
    ):
        """Initialize stats logger.

        Args:
            wandb_run: Optional wandb run for logging
            is_master: Whether this is the master process
            world_size: Number of distributed processes
        """
        self.wandb_run = wandb_run
        self.is_master = is_master
        self.world_size = world_size

        # Stats storage
        self.stats = defaultdict(list)
        self.timing_stats = {}

        # Initialize wandb metrics
        if wandb_run and is_master:
            # Define metrics (wandb x-axis values)
            metrics = ["agent_step", "epoch", "total_time", "train_time"]
            for metric in metrics:
                wandb_run.define_metric(f"metric/{metric}")

            # Set default x-axis to agent_step
            wandb_run.define_metric("*", step_metric="metric/agent_step")

            # Set up plots with custom x-axes
            metric_overrides = [
                ("overview/reward_vs_total_time", "metric/total_time"),
            ]
            for metric_name, step_metric in metric_overrides:
                wandb_run.define_metric(metric_name, step_metric=step_metric)

    def add_stats(self, stats_dict: Dict[str, Any]) -> None:
        """Add statistics to the logger.

        Args:
            stats_dict: Dictionary of statistics to add
        """
        for k, v in stats_dict.items():
            if isinstance(v, list):
                self.stats.setdefault(k, []).extend(v)
            else:
                if k not in self.stats:
                    self.stats[k] = v
                else:
                    try:
                        self.stats[k] += v
                    except TypeError:
                        self.stats[k] = [self.stats[k], v]

    def process_and_log(
        self,
        agent_step: int,
        epoch: int,
        timer: Stopwatch,
        losses: Losses,
        experience: Experience,
        policy: "BaseAgent",
        system_monitor: SystemMonitor,
        evals: Dict[str, float],
        trainer_config: Any,
        analyze_weights_interval: int = 0,
    ) -> None:
        """Process collected stats and log to wandb.

        Args:
            agent_step: Current agent steps
            epoch: Current epoch
            timer: Timing information
            losses: Loss statistics
            experience: Experience buffer for stats
            policy: Policy for weight analysis
            system_monitor: System resource monitor
            evals: Evaluation metrics
            trainer_config: Trainer configuration
            analyze_weights_interval: Interval for weight analysis
        """
        if not self.is_master or not self.wandb_run:
            self.stats.clear()
            return

        # Convert lists to means
        mean_stats = {}
        for k, v in self.stats.items():
            try:
                mean_stats[k] = np.mean(v)
            except (TypeError, ValueError) as e:
                raise RuntimeError(
                    f"Cannot compute mean for stat '{k}' with value {v!r} (type: {type(v)}). "
                    f"All collected stats must be numeric values or lists of numeric values. "
                    f"Error: {e}"
                ) from e
        self.stats = mean_stats

        # Weight analysis
        weight_stats = {}
        if analyze_weights_interval != 0 and epoch % analyze_weights_interval == 0:
            for metrics in policy.compute_weight_metrics():
                name = metrics.get("name", "unknown")
                for key, value in metrics.items():
                    if key != "name":
                        weight_stats[f"weights/{key}/{name}"] = value

        # Timing statistics
        elapsed_times = timer.get_all_elapsed()
        wall_time = timer.get_elapsed()
        train_time = elapsed_times.get("_rollout", 0) + elapsed_times.get("_train", 0)

        lap_times = timer.lap_all(agent_step, exclude_global=False)
        wall_time_for_lap = lap_times.pop("global", 0)

        # Compute rates
        epoch_steps = timer.get_lap_steps()
        if epoch_steps is None:
            epoch_steps = agent_step
        epoch_steps_per_second = epoch_steps / wall_time_for_lap if wall_time_for_lap > 0 else 0
        steps_per_second = timer.get_rate(agent_step) if wall_time > 0 else 0

        # Scale by world size for distributed training
        epoch_steps_per_second *= self.world_size
        steps_per_second *= self.world_size

        timing_stats = {
            **{
                f"timing_per_epoch/frac/{op}": lap_elapsed / wall_time_for_lap if wall_time_for_lap > 0 else 0
                for op, lap_elapsed in lap_times.items()
            },
            "timing_per_epoch/sps": epoch_steps_per_second,
            **{
                f"timing_cumulative/frac/{op}": elapsed / wall_time if wall_time > 0 else 0
                for op, elapsed in elapsed_times.items()
            },
            "timing_cumulative/sps": steps_per_second,
        }

        # Environment stats processing
        environment_stats = {f"env_{k.split('/')[0]}/{'/'.join(k.split('/')[1:])}": v for k, v in self.stats.items()}

        # Overview metrics
        overview = {
            "sps": epoch_steps_per_second,
        }

        # Calculate average reward
        task_reward_values = [v for k, v in environment_stats.items() if k.startswith("env_task_reward")]
        if task_reward_values:
            mean_reward = sum(task_reward_values) / len(task_reward_values)
            overview["reward"] = mean_reward
            overview["reward_vs_total_time"] = mean_reward

        # Include custom stats from trainer config
        if hasattr(trainer_config, "stats") and hasattr(trainer_config.stats, "overview"):
            for k, v in trainer_config.stats.overview.items():
                if k in self.stats:
                    overview[v] = self.stats[k]

        # Add evaluation scores
        category_scores_map = {key.split("/")[0]: value for key, value in evals.items() if key.endswith("/score")}
        for category, score in category_scores_map.items():
            overview[f"{category}_score"] = score

        # Get loss stats
        loss_stats = losses.stats()

        # Filter out unused losses
        if trainer_config.l2_reg_loss_coef == 0:
            loss_stats.pop("l2_reg_loss", None)
        if trainer_config.l2_init_loss_coef == 0:
            loss_stats.pop("l2_init_loss", None)
        if "ks_action_loss" in loss_stats and loss_stats["ks_action_loss"] == 0:
            loss_stats.pop("ks_action_loss", None)
            loss_stats.pop("ks_value_loss", None)

        # X-axis metrics
        metric_stats = {
            "metric/agent_step": agent_step * self.world_size,
            "metric/epoch": epoch,
            "metric/total_time": wall_time,
            "metric/train_time": train_time,
        }

        # Other parameters
        parameters = {
            "learning_rate": policy.optimizer.param_groups[0]["lr"] if hasattr(policy, "optimizer") else 0,
            "epoch_steps": epoch_steps,
            "num_minibatches": experience.num_minibatches,
        }

        # Log everything to wandb
        self.wandb_run.log(
            {
                **{f"overview/{k}": v for k, v in overview.items()},
                **{f"losses/{k}": v for k, v in loss_stats.items()},
                **{f"experience/{k}": v for k, v in experience.stats().items()},
                **{f"parameters/{k}": v for k, v in parameters.items()},
                **{f"eval_{k}": v for k, v in evals.items()},
                **{f"monitor/{k}": v for k, v in system_monitor.stats().items()},
                **environment_stats,
                **weight_stats,
                **timing_stats,
                **metric_stats,
            }
        )

        # Clear stats for next epoch
        self.stats.clear()

    def log_replay_url(self, epoch: int, replay_url: str) -> None:
        """Log a replay URL to wandb.

        Args:
            epoch: Current epoch
            replay_url: URL of the replay
        """
        if not self.is_master or not self.wandb_run:
            return

        player_url = "https://metta-ai.github.io/metta/?replayUrl=" + replay_url
        link_summary = {"replays/link": wandb.Html(f'<a href="{player_url}">MetaScope Replay (Epoch {epoch})</a>')}
        self.wandb_run.log(link_summary)
