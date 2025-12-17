"""Continual learning evaluator for multi-task evaluation and metrics.

This module provides components for evaluating policies on multiple tasks
to measure continual learning effects like forgetting and transfer.
"""

from collections import defaultdict
from typing import Literal

import torch
from pydantic import Field

from metta.common.config import Config
from metta.rl.training.component import TrainerComponent
from metta.sim.runner import run_simulations
from metta.sim.simulation_config import SimulationConfig
from mettagrid.config.mettagrid_config import MettaGridConfig


class ContinualLearningEvaluatorConfig(Config):
    """Configuration for continual learning evaluation."""

    epoch_interval: int = Field(default=50, ge=1, description="Evaluate every N epochs")

    num_episodes: int = Field(default=10, ge=1, description="Episodes to run per task evaluation")

    eval_mode: Literal["all", "past", "current", "future"] = Field(
        default="all", description="Which tasks to evaluate on"
    )

    parallel_evals: int = Field(default=4, ge=1, description="Number of parallel evaluations")


class ContinualLearningEvaluator(TrainerComponent):
    """Evaluates policy on multiple tasks to measure forgetting/transfer.

    This component periodically evaluates the current policy on all tasks
    in the sequential learning curriculum, computing metrics like:
    - Per-task performance over time
    - Forgetting (performance drop on past tasks)
    - Transfer (zero-shot performance on future tasks)
    """

    def __init__(
        self,
        config: ContinualLearningEvaluatorConfig,
        task_sequence: list[MettaGridConfig],
        device: torch.device,
    ):
        super().__init__(epoch_interval=config.epoch_interval)
        self.config = config
        self.task_sequence = task_sequence
        self.device = device

        # Track performance over time: {task_idx: [scores...]}
        self.performance_history: dict[int, list[float]] = defaultdict(list)

    def on_epoch_end(self, epoch: int) -> None:
        """Evaluate on selected tasks after training epoch."""
        if not self.should_handle_epoch(epoch):
            return

        # Get current policy
        policy_uri = self.context.latest_policy_uri()
        if not policy_uri:
            return

        # Determine current task from curriculum
        current_task_idx = self._get_current_task_index()

        # Select tasks to evaluate
        tasks_to_eval = self._select_tasks_to_evaluate(current_task_idx)

        # Evaluate on each task
        for task_idx in tasks_to_eval:
            score = self._evaluate_on_task(policy_uri, task_idx)
            self.performance_history[task_idx].append(score)
            self._log_task_performance(task_idx, score, epoch, current_task_idx)

        # Compute aggregate metrics
        self._log_aggregate_metrics(current_task_idx, epoch)

    def _get_current_task_index(self) -> int:
        """Infer current task index from curriculum state."""
        curriculum = self.context.curriculum

        # Try to get from curriculum's current task tracker
        if hasattr(curriculum, "_current_task_index"):
            return curriculum._current_task_index

        # Fallback: infer from active task IDs
        if hasattr(curriculum, "_active_tasks") and curriculum._active_tasks:
            task_id = list(curriculum._active_tasks.keys())[0]
            return task_id % len(self.task_sequence)

        return 0

    def _select_tasks_to_evaluate(self, current_idx: int) -> list[int]:
        """Select task indices based on eval_mode."""
        if self.config.eval_mode == "all":
            return list(range(len(self.task_sequence)))
        elif self.config.eval_mode == "past":
            return list(range(current_idx))
        elif self.config.eval_mode == "current":
            return [current_idx]
        elif self.config.eval_mode == "future":
            return list(range(current_idx + 1, len(self.task_sequence)))
        return []

    def _evaluate_on_task(self, policy_uri: str, task_idx: int) -> float:
        """Run evaluation episodes on a specific task."""
        task_config = self.task_sequence[task_idx]

        # Create simulation config
        sim_config = SimulationConfig(
            env=task_config,
            num_episodes=self.config.num_episodes,
            policy_uri=policy_uri,
        )

        # Run simulation
        results = run_simulations(
            simulations=[sim_config],
            device=self.device,
        )

        # Extract mean reward
        if results and len(results) > 0:
            return results[0].get("mean_reward", 0.0)
        return 0.0

    def _log_task_performance(
        self,
        task_idx: int,
        score: float,
        epoch: int,
        current_task_idx: int,
    ) -> None:
        """Log task-specific performance to wandb."""
        if not self.context.distributed.should_log():
            return

        # Categorize task
        if task_idx < current_task_idx:
            category = "past"
        elif task_idx == current_task_idx:
            category = "current"
        else:
            category = "future"

        # Log with hierarchical keys
        import wandb

        wandb.log(
            {
                f"continual/task_{task_idx}/score": score,
                f"continual/{category}_tasks/task_{task_idx}": score,
                "epoch": epoch,
            }
        )

    def _log_aggregate_metrics(self, current_task_idx: int, epoch: int) -> None:
        """Compute and log forgetting/transfer metrics."""
        if not self.context.distributed.should_log():
            return

        import wandb

        # Forgetting: avg drop from peak on past tasks
        forgetting_scores = []
        for task_idx in range(current_task_idx):
            history = self.performance_history[task_idx]
            if len(history) >= 2:
                peak_score = max(history)
                current_score = history[-1]
                forgetting = peak_score - current_score
                forgetting_scores.append(forgetting)

        if forgetting_scores:
            avg_forgetting = sum(forgetting_scores) / len(forgetting_scores)
            wandb.log(
                {
                    "continual/metrics/forgetting": avg_forgetting,
                    "epoch": epoch,
                }
            )

        # Transfer: zero-shot performance on future tasks
        transfer_scores = []
        for task_idx in range(current_task_idx + 1, len(self.task_sequence)):
            history = self.performance_history[task_idx]
            if history:
                transfer_scores.append(history[-1])

        if transfer_scores:
            avg_transfer = sum(transfer_scores) / len(transfer_scores)
            wandb.log(
                {
                    "continual/metrics/transfer": avg_transfer,
                    "epoch": epoch,
                }
            )
