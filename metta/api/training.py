"""Training utilities for Metta API."""

import os
from typing import Any, Dict, Optional, Tuple

import torch
from heavyball import ForeachMuon
from omegaconf import DictConfig

from metta.agent.policy_store import PolicyStore
from metta.common.wandb.wandb_context import WandbContext
from metta.rl.functions import cleanup_old_policies, save_policy_with_metadata
from metta.rl.trainer_checkpoint import TrainerCheckpoint


class Optimizer:
    """Wrapper class for optimizer - provides a unified interface for Adam and Muon."""

    def __init__(
        self,
        optimizer_type: str,
        policy: torch.nn.Module,
        learning_rate: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
    ):
        opt_cls = torch.optim.Adam if optimizer_type == "adam" else ForeachMuon
        # ForeachMuon expects int for weight_decay, Adam expects float
        if optimizer_type != "adam":
            weight_decay = int(weight_decay)

        self.optimizer = opt_cls(
            policy.parameters(),
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.max_grad_norm = max_grad_norm
        self.param_groups = self.optimizer.param_groups

    def step(self, loss: torch.Tensor, epoch: int, accumulate_minibatches: int = 1):
        """Optimizer step with gradient accumulation."""
        self.optimizer.zero_grad()
        loss.backward()

        if (epoch + 1) % accumulate_minibatches == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for group in self.optimizer.param_groups for p in group["params"]], self.max_grad_norm
            )
            self.optimizer.step()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


def initialize_wandb(
    run_name: str,
    run_dir: str,
    enabled: bool = True,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    job_type: str = "train",
    tags: Optional[list] = None,
    notes: Optional[str] = None,
) -> Tuple[Any, WandbContext]:
    """Initialize wandb logging."""
    if enabled:
        wandb_config = {
            "enabled": True,
            "project": project or os.environ.get("WANDB_PROJECT", "metta"),
            "entity": entity or os.environ.get("WANDB_ENTITY", "metta-research"),
            "group": run_name,
            "name": run_name,
            "run_id": run_name,
            "data_dir": run_dir,
            "job_type": job_type,
            "tags": tags or [],
            "notes": notes or "",
        }
    else:
        wandb_config = {"enabled": False}

    global_config = {
        "run": run_name,
        "run_dir": run_dir,
        "cmd": job_type,
        "wandb": wandb_config,
    }

    if config:
        global_config.update(config)

    wandb_ctx = WandbContext(DictConfig(wandb_config), DictConfig(global_config))
    wandb_run = wandb_ctx.__enter__()

    return wandb_run, wandb_ctx


def cleanup_wandb(wandb_ctx: Optional[WandbContext]):
    """Clean up wandb context."""
    if wandb_ctx is not None:
        wandb_ctx.__exit__(None, None, None)


def cleanup_distributed():
    """Clean up distributed training."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def load_checkpoint(
    checkpoint_dir: str,
    agent: torch.nn.Module,
    optimizer: Optional[Optimizer],
    policy_store: PolicyStore,
    device: torch.device,
) -> Tuple[int, int, Optional[str]]:
    """Load checkpoint and return agent_step, epoch, policy_path."""
    # Try to load existing checkpoint
    existing_checkpoint = TrainerCheckpoint.load(checkpoint_dir)

    if existing_checkpoint:
        agent_step = existing_checkpoint.agent_step
        epoch = existing_checkpoint.epoch

        # Load optimizer state if provided
        if optimizer is not None and existing_checkpoint.optimizer_state_dict:
            try:
                if hasattr(optimizer, "optimizer"):
                    optimizer.optimizer.load_state_dict(existing_checkpoint.optimizer_state_dict)
                elif hasattr(optimizer, "load_state_dict"):
                    optimizer.load_state_dict(existing_checkpoint.optimizer_state_dict)
            except ValueError:
                pass  # Ignore optimizer state mismatch

        return agent_step, epoch, existing_checkpoint.policy_path

    return 0, 0, None


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
    """Save checkpoint if needed."""
    should_save = force_save or (epoch % checkpoint_interval == 0)
    if not should_save:
        return None

    # Only master saves in distributed mode
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return None

    # Save policy
    saved_record = save_policy_with_metadata(
        policy=agent,
        policy_store=policy_store,
        epoch=epoch,
        agent_step=agent_step,
        evals={},
        timer=None,
        initial_policy_record=None,
        run_name="",
        is_master=True,
    )

    if saved_record:
        # Save training state
        optimizer_state_dict = None
        if optimizer is not None:
            if hasattr(optimizer, "optimizer"):
                optimizer_state_dict = optimizer.optimizer.state_dict()
            elif hasattr(optimizer, "state_dict"):
                optimizer_state_dict = optimizer.state_dict()

        checkpoint = TrainerCheckpoint(
            agent_step=agent_step,
            epoch=epoch,
            optimizer_state_dict=optimizer_state_dict,
            policy_path=saved_record.uri if hasattr(saved_record, "uri") else None,
            stopwatch_state=None,
        )
        checkpoint.save(checkpoint_path)

        # Clean up old policies periodically
        if epoch % 10 == 0:
            cleanup_old_policies(checkpoint_path, keep_last_n=5)

        return saved_record

    return None


def ensure_initial_policy(
    agent: torch.nn.Module,
    policy_store: PolicyStore,
    checkpoint_path: str,
    loaded_policy_path: Optional[str],
    device: torch.device,
):
    """Ensure initial policy exists - loads policy if path provided."""
    # If we loaded a policy, load it into the agent
    if loaded_policy_path and os.path.exists(loaded_policy_path):
        try:
            policy_pr = policy_store.policy_record(loaded_policy_path)
            agent.load_state_dict(policy_pr.policy.state_dict())
        except Exception:
            pass  # Ignore errors

    # Ensure all ranks synchronize
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
