"""Checkpoint management for Metta training."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def load_checkpoint(
    checkpoint_dir: str,
    agent: Any,
    optimizer: Optional[Any] = None,
    policy_store: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Tuple[int, int, Optional[str]]:
    """Load training checkpoint if it exists.

    This is a simplified version that just loads the checkpoint without
    handling distributed coordination or initial policy creation.

    Args:
        checkpoint_dir: Directory containing checkpoints
        agent: The agent/policy to potentially update
        optimizer: Optional optimizer to restore state to
        policy_store: Optional PolicyStore (not used in simplified version)
        device: Device (not used in simplified version)

    Returns:
        Tuple of (agent_step, epoch, policy_path)
        - agent_step: Current training step (0 if no checkpoint)
        - epoch: Current epoch (0 if no checkpoint)
        - policy_path: Path to loaded policy (None if no checkpoint)
    """
    from metta.rl.trainer_checkpoint import TrainerCheckpoint

    # Try to load existing checkpoint
    existing_checkpoint = TrainerCheckpoint.load(checkpoint_dir)

    if existing_checkpoint:
        # Restore training state
        agent_step = existing_checkpoint.agent_step
        epoch = existing_checkpoint.epoch

        # Load policy state if agent provided and policy path exists
        if agent is not None and existing_checkpoint.policy_path and policy_store is not None:
            try:
                policy_pr = policy_store.policy_record(existing_checkpoint.policy_path)
                agent.load_state_dict(policy_pr.policy.state_dict())
            except Exception as e:
                logger.warning(f"Failed to load policy state: {e}")

        # Load optimizer state if provided
        if optimizer is not None and existing_checkpoint.optimizer_state_dict:
            try:
                if hasattr(optimizer, "optimizer"):
                    # Handle our Optimizer wrapper
                    optimizer.optimizer.load_state_dict(existing_checkpoint.optimizer_state_dict)
                elif hasattr(optimizer, "load_state_dict"):
                    # Handle raw PyTorch optimizer
                    optimizer.load_state_dict(existing_checkpoint.optimizer_state_dict)
            except ValueError:
                logger.warning("Optimizer state dict doesn't match. Starting with fresh optimizer state.")

        return agent_step, epoch, existing_checkpoint.policy_path

    # No checkpoint found
    return 0, 0, None


def save_checkpoint(
    epoch: int,
    agent_step: int,
    agent: Any,
    optimizer: Any,
    policy_store: Any,
    checkpoint_path: str,
    checkpoint_interval: int,
    stats: Optional[Dict[str, Any]] = None,
    force_save: bool = False,
) -> Optional[Any]:
    """Save a training checkpoint including policy and training state.

    In distributed mode, only the master process saves. Callers are responsible
    for adding barriers if synchronization is needed.

    Args:
        epoch: Current training epoch
        agent_step: Current agent step count
        agent: The agent/policy to save
        optimizer: The optimizer with state to save
        policy_store: PolicyStore instance for saving policies
        checkpoint_path: Directory path for saving checkpoints
        checkpoint_interval: How often to save checkpoints
        stats: Optional statistics dictionary to include in metadata
        force_save: Force save even if not on checkpoint interval

    Returns:
        The saved policy record if saved, None otherwise
    """
    from metta.rl.trainer_checkpoint import TrainerCheckpoint

    should_save = force_save or (epoch % checkpoint_interval == 0)
    if not should_save:
        return None

    # Only master saves in distributed mode
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return None

    # Master (or single GPU) saves the checkpoint
    logger.info(f"Saving checkpoint at epoch {epoch}")

    # Extract the actual policy module from distributed wrapper if needed
    from torch.nn.parallel import DistributedDataParallel

    from metta.agent.metta_agent import DistributedMettaAgent

    policy_to_save = agent
    if isinstance(agent, DistributedMettaAgent):
        policy_to_save = agent.module
    elif isinstance(agent, DistributedDataParallel):
        policy_to_save = agent.module

    # Create policy record directly
    name = policy_store.make_model_name(epoch)
    policy_record = policy_store.create_empty_policy_record(name)
    policy_record.metadata = {
        "agent_step": agent_step,
        "epoch": epoch,
        "stats": dict(stats) if stats else {},
        "final": force_save,  # Mark if this is the final checkpoint
    }
    policy_record.policy = policy_to_save

    # Save through policy store
    saved_policy_record = policy_store.save(policy_record)

    # Save training state
    # Get optimizer state dict
    optimizer_state_dict = None
    if optimizer is not None:
        if hasattr(optimizer, "optimizer"):
            # Handle our Optimizer wrapper
            optimizer_state_dict = optimizer.optimizer.state_dict()
        elif hasattr(optimizer, "state_dict"):
            # Handle raw PyTorch optimizer
            optimizer_state_dict = optimizer.state_dict()

    trainer_checkpoint = TrainerCheckpoint(
        agent_step=agent_step,
        epoch=epoch,
        total_agent_step=agent_step,
        optimizer_state_dict=optimizer_state_dict,
        policy_path=saved_policy_record.uri if hasattr(saved_policy_record, "uri") else None,
        stopwatch_state=None,
    )
    trainer_checkpoint.save(checkpoint_path)

    # Clean up old policies to prevent disk space issues
    if epoch % 10 == 0:
        cleanup_old_policies(checkpoint_path, keep_last_n=5)

    return saved_policy_record


def wrap_agent_distributed(agent: Any, device: torch.device) -> Any:
    """Wrap agent in DistributedMettaAgent if distributed training is initialized.

    Args:
        agent: The agent to potentially wrap
        device: The device to use

    Returns:
        The agent, possibly wrapped in DistributedMettaAgent
    """
    if torch.distributed.is_initialized():
        from torch.nn.parallel import DistributedDataParallel

        from metta.agent.metta_agent import DistributedMettaAgent

        # For CPU, we need to handle DistributedDataParallel differently
        if device.type == "cpu":
            # Convert BatchNorm to SyncBatchNorm
            agent = torch.nn.SyncBatchNorm.convert_sync_batchnorm(agent)
            # For CPU, don't pass device_ids
            agent = DistributedDataParallel(agent)
        else:
            # For GPU, use the custom DistributedMettaAgent wrapper
            agent = DistributedMettaAgent(agent, device)

    return agent


def ensure_initial_policy(
    agent: Any,
    policy_store: Any,
    checkpoint_path: str,
    loaded_policy_path: Optional[str],
    device: torch.device,
) -> None:
    """Ensure all ranks have the same initial policy in distributed training.

    If no checkpoint exists, master creates and saves the initial policy,
    then all ranks synchronize. In single GPU mode, just saves the initial policy.

    Args:
        agent: The agent to initialize
        policy_store: PolicyStore instance
        checkpoint_path: Directory for checkpoints
        loaded_policy_path: Path to already loaded policy (None if no checkpoint)
        device: Training device
    """
    # If we already loaded a policy, nothing to do
    if loaded_policy_path is not None:
        return

    # Get distributed info
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        is_master = rank == 0
    else:
        rank = 0
        is_master = True

    if torch.distributed.is_initialized():
        if is_master:
            # Master creates and saves initial policy
            save_checkpoint(
                epoch=0,
                agent_step=0,
                agent=agent,
                optimizer=None,
                policy_store=policy_store,
                checkpoint_path=checkpoint_path,
                checkpoint_interval=1,  # Force save
                stats={},
                force_save=True,
            )
            # Master waits at barrier after saving
            torch.distributed.barrier()
        else:
            # Non-master ranks wait at barrier first
            torch.distributed.barrier()

            # Then load the policy master created
            default_policy_path = os.path.join(checkpoint_path, policy_store.make_model_name(0))
            from metta.common.util.fs import wait_for_file

            if not wait_for_file(default_policy_path, timeout=300):
                raise RuntimeError(f"Rank {rank}: Timeout waiting for policy at {default_policy_path}")

            # Load the policy
            policy_pr = policy_store.policy_record(default_policy_path)
            agent.load_state_dict(policy_pr.policy.state_dict())  # type: ignore
    else:
        # Single GPU mode creates and saves initial policy
        save_checkpoint(
            epoch=0,
            agent_step=0,
            agent=agent,
            optimizer=None,
            policy_store=policy_store,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=1,
            stats={},
            force_save=True,
        )


def cleanup_old_policies(checkpoint_dir: str, keep_last_n: int = 5) -> None:
    """Clean up old saved policies to prevent memory accumulation.

    Args:
        checkpoint_dir: Directory containing policy checkpoints
        keep_last_n: Number of most recent policies to keep
    """
    try:
        # Get checkpoint directory
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            return

        # List all policy files
        policy_files = sorted(checkpoint_path.glob("policy_*.pt"))

        # Keep only the most recent ones
        if len(policy_files) > keep_last_n:
            files_to_remove = policy_files[:-keep_last_n]
            for file_path in files_to_remove:
                try:
                    file_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove old policy file {file_path}: {e}")

    except Exception as e:
        logger.warning(f"Error during policy cleanup: {e}")
