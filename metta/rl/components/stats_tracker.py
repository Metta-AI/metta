"""Tracks and processes statistics during training."""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

from metta.app_backend.stats_client import StatsClient
from metta.common.profiling.memory_monitor import MemoryMonitor
from metta.common.profiling.stopwatch import Stopwatch
from metta.common.util.system_monitor import SystemMonitor
from metta.eval.eval_request_config import EvalRewardSummary
from metta.rl.experience import Experience
from metta.rl.kickstarter import Kickstarter
from metta.rl.losses import Losses
from metta.rl.trainer_config import TrainerConfig
from metta.rl.util.optimization import compute_gradient_stats
from metta.rl.util.stats import (
    StatsTracker as UtilStatsTracker,
)
from metta.rl.util.stats import (
    accumulate_rollout_stats,
    build_wandb_stats,
    process_training_stats,
)
from metta.rl.util.utils import should_run

logger = logging.getLogger(__name__)


class StatsTracker:
    """Tracks and processes training statistics throughout the training process."""

    def __init__(
        self,
        trainer_config: TrainerConfig,
        timer: Stopwatch,
        is_master: bool = True,
        system_monitor: Optional[SystemMonitor] = None,
        memory_monitor: Optional[MemoryMonitor] = None,
        stats_client: Optional[StatsClient] = None,
    ):
        """Initialize stats tracker.

        Args:
            trainer_config: Training configuration
            timer: Stopwatch for timing operations
            is_master: Whether this is the master process
            system_monitor: Optional system monitor for resource tracking
            memory_monitor: Optional memory monitor for memory tracking
            stats_client: Optional stats client for tracking training runs
        """
        self.trainer_config = trainer_config
        self.timer = timer
        self.is_master = is_master
        self.system_monitor = system_monitor
        self.memory_monitor = memory_monitor
        self.stats_client = stats_client

        # Initialize stats tracker
        self.stats_tracker = UtilStatsTracker(rollout_stats=defaultdict(list))

        # Evaluation scores
        self.eval_scores = EvalRewardSummary()

    def initialize_stats_tracking(self, wandb_run: Optional[Any] = None) -> None:
        """Initialize stats tracking for training run.

        Args:
            wandb_run: Optional wandb run for extracting metadata
        """
        if self.stats_client is None:
            return

        if wandb_run is not None:
            name = wandb_run.name if wandb_run.name is not None else "unknown"
            url = wandb_run.url
            tags = list(wandb_run.tags) if wandb_run.tags is not None else None
            description = wandb_run.notes
        else:
            name = "unknown"
            url = None
            tags = None
            description = None

        try:
            self.stats_tracker.stats_run_id = self.stats_client.create_training_run(
                name=name, attributes={}, url=url, description=description, tags=tags
            ).id
        except Exception as e:
            logger.warning(f"Failed to create training run: {e}")

    def process_rollout_stats(self, raw_infos: List[Any]) -> None:
        """Process statistics from rollout phase.

        Args:
            raw_infos: Raw info dictionaries from environment
        """
        accumulate_rollout_stats(raw_infos, self.stats_tracker.rollout_stats)

    def compute_gradient_stats(self, agent: Any, epoch: int) -> None:
        """Compute gradient statistics if configured.

        Args:
            agent: The policy/agent
            epoch: Current training epoch
        """
        if should_run(epoch, self.trainer_config.grad_mean_variance_interval, self.is_master):
            self.stats_tracker.grad_stats = compute_gradient_stats(agent)

    def build_training_stats(
        self,
        losses: Losses,
        experience: Experience,
        kickstarter: Optional[Kickstarter],
        agent_step: int,
        epoch: int,
        current_lr: float,
        current_policy_generation: int,
        timing_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build complete statistics dictionary for logging.

        Args:
            losses: Loss values from training
            experience: Experience buffer
            kickstarter: Optional kickstarter
            agent_step: Current training step
            epoch: Current training epoch
            current_lr: Current learning rate
            current_policy_generation: Current policy generation
            timing_info: Timing statistics

        Returns:
            Dictionary of all statistics ready for logging
        """
        # Process collected stats (convert lists to means)
        processed_stats = process_training_stats(
            raw_stats=self.stats_tracker.rollout_stats,
            losses=losses,
            experience=experience,
            trainer_config=self.trainer_config,
            kickstarter=kickstarter,
        )

        # Update stats with mean values for consistency
        self.stats_tracker.rollout_stats = processed_stats["mean_stats"]

        # Build parameters dictionary
        parameters = {
            "learning_rate": current_lr,
            "epoch_steps": timing_info["epoch_steps"],
            "num_minibatches": experience.num_minibatches,
            "generation": current_policy_generation,
        }

        # Get system and memory stats
        system_stats = self.system_monitor.stats() if self.system_monitor else {}
        memory_stats = self.memory_monitor.stats() if self.memory_monitor else {}

        # Current hyperparameter values
        hyperparameters = {
            "learning_rate": current_lr,
            "ppo_clip_coef": self.trainer_config.ppo.clip_coef,
            "ppo_vf_clip_coef": self.trainer_config.ppo.vf_clip_coef,
            "ppo_ent_coef": self.trainer_config.ppo.ent_coef,
            "ppo_l2_reg_loss_coef": self.trainer_config.ppo.l2_reg_loss_coef,
            "ppo_l2_init_loss_coef": self.trainer_config.ppo.l2_init_loss_coef,
        }

        # Build complete stats dictionary
        all_stats = build_wandb_stats(
            processed_stats=processed_stats,
            timing_info=timing_info,
            weight_stats={},  # Weight stats computation moved to separate method
            grad_stats=self.stats_tracker.grad_stats,
            system_stats=system_stats,
            memory_stats=memory_stats,
            parameters=parameters,
            hyperparameters=hyperparameters,
            evals=self.eval_scores,
            agent_step=agent_step,
            epoch=epoch,
        )

        return all_stats

    def compute_weight_stats(self, agent: Any, epoch: int) -> Dict[str, float]:
        """Compute weight statistics if configured.

        Args:
            agent: The policy/agent
            epoch: Current training epoch

        Returns:
            Dictionary of weight statistics
        """
        weight_stats = {}
        if hasattr(self.trainer_config, "agent") and hasattr(self.trainer_config.agent, "analyze_weights_interval"):
            if (
                self.trainer_config.agent.analyze_weights_interval != 0
                and epoch % self.trainer_config.agent.analyze_weights_interval == 0
            ):
                for metrics in agent.compute_weight_metrics():
                    name = metrics.get("name", "unknown")
                    for key, value in metrics.items():
                        if key != "name":
                            weight_stats[f"weights/{key}/{name}"] = value
        return weight_stats

    def clear_stats(self) -> None:
        """Clear stats for next iteration."""
        self.stats_tracker.clear_rollout_stats()
        self.stats_tracker.clear_grad_stats()

    def update_eval_scores(self, eval_scores: EvalRewardSummary) -> None:
        """Update evaluation scores.

        Args:
            eval_scores: New evaluation scores
        """
        self.eval_scores = eval_scores
