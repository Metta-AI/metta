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
    evals: Dict[str, float] = field(default_factory=dict)
    latest_saved_policy_record: Optional[Any] = None
    initial_policy_record: Optional[Any] = None
    # Stats tracking
    stats_epoch_start: int = 0
    stats_epoch_id: Optional[Any] = None
    stats_run_id: Optional[Any] = None


def calculate_anneal_beta(
    epoch: int,
    total_timesteps: int,
    batch_size: int,
    prio_alpha: float,
    prio_beta0: float,
) -> float:
    """Calculate annealed beta for prioritized experience replay.

    Args:
        epoch: Current epoch
        total_timesteps: Total training timesteps
        batch_size: Batch size
        prio_alpha: Priority alpha
        prio_beta0: Initial beta value

    Returns:
        Annealed beta value
    """
    total_epochs = max(1, total_timesteps // batch_size)
    anneal_beta = prio_beta0 + (1 - prio_beta0) * prio_alpha * epoch / total_epochs
    return anneal_beta
