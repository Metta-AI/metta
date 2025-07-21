"""Specialized state containers for functional training approach."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from metta.eval.eval_request_config import EvalRewardSummary


@dataclass
class TrainingProgress:
    """Tracks core training loop progress."""

    agent_step: int = 0
    epoch: int = 0

    def increment_step(self, num_steps: int) -> None:
        """Increment agent step counter."""
        self.agent_step += num_steps

    def increment_epoch(self, num_epochs: int = 1) -> None:
        """Increment epoch counter."""
        self.epoch += num_epochs


@dataclass
class StatsTracker:
    """Manages training statistics and database tracking."""

    # Rollout stats collected during episodes
    rollout_stats: Dict[str, Any] = field(default_factory=dict)

    # Gradient statistics (computed periodically)
    grad_stats: Dict[str, float] = field(default_factory=dict)

    # Database tracking for stats service
    stats_epoch_start: int = 0
    stats_epoch_id: Optional[Any] = None
    stats_run_id: Optional[Any] = None

    def clear_rollout_stats(self) -> None:
        """Clear rollout stats after processing."""
        self.rollout_stats.clear()

    def clear_grad_stats(self) -> None:
        """Clear gradient stats after processing."""
        self.grad_stats.clear()

    def update_epoch_tracking(self, new_epoch_start: int) -> None:
        """Update epoch tracking after creating a new stats epoch."""
        self.stats_epoch_start = new_epoch_start


@dataclass
class PolicyTracker:
    """Manages policy records throughout training."""

    initial_policy_record: Optional[Any] = None
    latest_saved_policy_record: Optional[Any] = None

    def update_latest(self, policy_record: Any) -> None:
        """Update the latest saved policy record."""
        self.latest_saved_policy_record = policy_record

    def has_saved_policy(self) -> bool:
        """Check if we have a saved policy."""
        return self.latest_saved_policy_record is not None


@dataclass
class EvaluationTracker:
    """Tracks evaluation results."""

    scores: EvalRewardSummary = field(default_factory=EvalRewardSummary)

    def update(self, new_scores: EvalRewardSummary) -> None:
        """Update evaluation scores."""
        self.scores = new_scores

    def has_scores(self) -> bool:
        """Check if we have any evaluation scores."""
        return bool(self.scores.category_scores) or bool(self.scores.simulation_scores)
