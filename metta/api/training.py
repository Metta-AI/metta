"""Training state and helper functions for Metta."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TrainerState:
    """Mutable state for training that gets passed between functions."""

    agent_step: int = 0
    epoch: int = 0
    stats: Dict[str, Any] = field(default_factory=dict)
    grad_stats: Dict[str, float] = field(default_factory=dict)
    evals: Any = field(default_factory=dict)  # Will be EvalRewardSummary
    latest_saved_policy_record: Optional[Any] = None
    initial_policy_record: Optional[Any] = None
    # Stats tracking
    stats_epoch_start: int = 0
    stats_epoch_id: Optional[Any] = None
    stats_run_id: Optional[Any] = None
