from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch

from metta.agent.policy_store import PolicyStore
from metta.rl.trainer_checkpoint import TrainerCheckpoint
from metta.rl.util.policy_management import cleanup_old_policies, save_policy_with_metadata

from .optimizer import Optimizer

__all__ = [
    "cleanup_distributed",
    "load_checkpoint",
    "save_checkpoint",
    "ensure_initial_policy",
]


def cleanup_distributed() -> None:
    """Destroy the torch distributed process group if initialized."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def load_checkpoint(
    checkpoint_dir: str,
    agent: torch.nn.Module,
    optimizer: Optional[Optimizer],
    policy_store: PolicyStore,
    device: torch.device,
) -> Tuple[int, int, Optional[str]]:
    """Load checkpoint returning (agent_step, epoch, policy_path)."""
    checkpoint = TrainerCheckpoint.load(checkpoint_dir)
    if checkpoint is None:
        return 0, 0, None

    # Restore agent step / epoch
    agent_step, epoch = checkpoint.agent_step, checkpoint.epoch

    # Restore optimizer state if possible
    if optimizer and checkpoint.optimizer_state_dict:
        try:
            if hasattr(optimizer, "optimizer"):
                optimizer.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
            else:
                optimizer.load_state_dict(checkpoint.optimizer_state_dict)  # type: ignore[arg-type]
        except ValueError:
            pass  # Ignore mismatch

    return agent_step, epoch, checkpoint.policy_path


def save_checkpoint(
    epoch: int,
    agent_step: int,
    agent: torch.nn.Module,
    optimizer: Optional[Optimizer],
    policy_store: PolicyStore,
    checkpoint_path: str,
    checkpoint_interval: int,
    stats: Optional[Dict[str, Any]] = None,
    force_save: bool = False,
) -> Optional[Any]:
    """Save checkpoint if *epoch* matches *checkpoint_interval* (or force_save)."""
    if not force_save and epoch % checkpoint_interval != 0:
        return None

    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return None  # Non-master ranks skip

    # Persist policy first
    saved_record = save_policy_with_metadata(
        policy=agent,
        policy_store=policy_store,
        epoch=epoch,
        agent_step=agent_step,
        evals=stats or {},
        timer=None,
        initial_policy_record=None,
        run_name="",
        is_master=True,
    )
    if not saved_record:
        return None

    # Save trainer state
    opt_state = None
    if optimizer:
        if hasattr(optimizer, "optimizer"):
            opt_state = optimizer.optimizer.state_dict()
        else:
            opt_state = optimizer.state_dict()  # type: ignore[attr-type]

    checkpoint = TrainerCheckpoint(
        agent_step=agent_step,
        epoch=epoch,
        optimizer_state_dict=opt_state,
        policy_path=saved_record.uri if hasattr(saved_record, "uri") else None,
        stopwatch_state=None,
    )
    checkpoint.save(checkpoint_path)

    # Prune old policies occasionally
    if epoch % 10 == 0:
        cleanup_old_policies(checkpoint_path, keep_last_n=5)

    return saved_record


def ensure_initial_policy(
    agent: torch.nn.Module,
    policy_store: PolicyStore,
    checkpoint_path: str,
    loaded_policy_path: Optional[str],
    device: torch.device,
) -> None:
    """Load *loaded_policy_path* into *agent* if provided and ensure sync across ranks."""
    if loaded_policy_path and os.path.exists(loaded_policy_path):
        try:
            policy_pr = policy_store.policy_record(loaded_policy_path)
            agent.load_state_dict(policy_pr.policy.state_dict())
        except Exception:
            pass  # ignore

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
