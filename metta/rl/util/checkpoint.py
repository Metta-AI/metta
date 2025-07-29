"""Checkpoint management utilities for training."""

import logging
from typing import Any, Optional

import torch

from metta.agent.policy_store import PolicyStore
from metta.common.profiling.stopwatch import Stopwatch
from metta.eval.eval_request_config import EvalRewardSummary
from metta.rl.kickstarter import Kickstarter
from metta.rl.trainer_checkpoint import TrainerCheckpoint
from metta.rl.util.policy_management import cleanup_old_policies, save_policy_with_metadata

logger = logging.getLogger(__name__)


def save_training_checkpoint(
    agent_step: int,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    policy_path: str,
    timer: Stopwatch,
    run_dir: str,
    kickstarter: Optional[Kickstarter] = None,
    force: bool = False,
) -> None:
    """Save training checkpoint.

    Args:
        agent_step: Current agent step
        epoch: Current epoch
        optimizer: Optimizer to save state from
        policy_path: Path to saved policy
        timer: Stopwatch timer
        run_dir: Run directory
        kickstarter: Optional kickstarter with teacher info
        force: Force save even if not scheduled
    """
    extra_args = {}
    if kickstarter and kickstarter.enabled and kickstarter.teacher_uri is not None:
        extra_args["teacher_pr_uri"] = kickstarter.teacher_uri

    checkpoint = TrainerCheckpoint(
        agent_step=agent_step,
        epoch=epoch,
        optimizer_state_dict=optimizer.state_dict(),
        stopwatch_state=timer.save_state(),
        policy_path=policy_path,
        extra_args=extra_args,
    )
    checkpoint.save(run_dir)
    logger.info(f"Saved training state at epoch {epoch}")


def save_policy_checkpoint(
    policy: Any,
    policy_store: PolicyStore,
    epoch: int,
    agent_step: int,
    evals: EvalRewardSummary,
    timer: Stopwatch,
    initial_policy_record: Optional[Any] = None,
    run_name: str = "",
    is_master: bool = True,
    checkpoint_dir: Optional[str] = None,
    force: bool = False,
) -> Optional[Any]:
    """Save policy checkpoint and optionally clean up old checkpoints.

    Args:
        policy: Policy to save
        policy_store: Policy store for saving
        epoch: Current epoch
        agent_step: Current agent step
        evals: Evaluation scores
        timer: Stopwatch timer
        initial_policy_record: Initial policy record for metadata
        run_name: Run name
        is_master: Whether this is the master rank
        checkpoint_dir: Directory containing checkpoints for cleanup
        force: Force save even if not scheduled

    Returns:
        Saved policy record or None
    """
    if not is_master:
        return None

    saved_record = save_policy_with_metadata(
        policy=policy,
        policy_store=policy_store,
        epoch=epoch,
        agent_step=agent_step,
        evals=evals,
        timer=timer,
        initial_policy_record=initial_policy_record,
        run_name=run_name,
        is_master=is_master,
    )

    if saved_record and checkpoint_dir:
        # Clean up old policies periodically
        if epoch % 10 == 0:
            cleanup_old_policies(checkpoint_dir, keep_last_n=5)

    return saved_record


def load_training_checkpoint(run_dir: str) -> Optional[TrainerCheckpoint]:
    """Load training checkpoint from run directory.

    Args:
        run_dir: Run directory to load from

    Returns:
        TrainerCheckpoint or None if not found
    """
    checkpoint = TrainerCheckpoint.load(run_dir)
    if checkpoint:
        logger.info(f"Loaded checkpoint from epoch {checkpoint.epoch}, step {checkpoint.agent_step}")
    return checkpoint


def restore_optimizer_state(
    optimizer: torch.optim.Optimizer,
    checkpoint: Optional[TrainerCheckpoint],
) -> bool:
    """Restore optimizer state from checkpoint.

    Args:
        optimizer: Optimizer to restore state to
        checkpoint: Checkpoint containing optimizer state

    Returns:
        True if successfully restored, False otherwise
    """
    if not checkpoint or not checkpoint.optimizer_state_dict:
        return False

    try:
        optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        logger.info("Successfully loaded optimizer state from checkpoint")
        return True
    except ValueError:
        logger.warning("Optimizer state dict doesn't match. Starting with fresh optimizer state.")
        return False


def should_checkpoint(epoch: int, checkpoint_interval: int, is_master: bool = True) -> bool:
    """Check if checkpoint should be saved at this epoch.

    Args:
        epoch: Current epoch
        checkpoint_interval: Interval between checkpoints
        is_master: Whether this is the master rank

    Returns:
        True if should checkpoint
    """
    if not is_master or checkpoint_interval <= 0:
        return False
    return epoch % checkpoint_interval == 0
