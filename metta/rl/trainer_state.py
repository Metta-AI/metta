from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from metta.agent.metta_agent import PolicyAgent
    from metta.common.profiling.stopwatch import Stopwatch
    from metta.eval.eval_request_config import EvalRewardSummary
    from metta.rl.experience import Experience
    from metta.rl.stats import StatsTracker


@dataclass(slots=True)
class TrainerState:
    """Lightweight, fast, mutable container for training loop state. Also provides a way for losses to influence
    trainer's control flow."""

    agent_step: int = 0
    epoch: int = 0
    update_epoch: int = 0
    mb_idx: int = 0
    optimizer: torch.optim.Optimizer | None = None
    training_env_id: slice | None = None

    # Control flags and scratch metrics
    stop_rollout: bool = False
    stop_update_epoch: bool = False

    # Hook-related fields for data sharing
    rollout_stats: dict[str, list[float]] | None = None
    loss_stats: dict[str, float] | None = None
    eval_scores: "EvalRewardSummary | None" = None
    experience: "Experience | None" = None
    policy: "PolicyAgent | None" = None
    latest_checkpoint_uri: str | None = None
    latest_wandb_uri: str | None = None
    stats_tracker: "StatsTracker | None" = None
    timer: "Stopwatch | None" = None
