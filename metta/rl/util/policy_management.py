"""Policy management utilities for Metta."""

import logging
import os
from pathlib import Path
from typing import Any, Tuple

import torch
from torch.nn.parallel import DistributedDataParallel

from metta.agent.metta_agent import DistributedMettaAgent, MettaAgent, make_policy
from metta.common.util.fs import wait_for_file
from metta.rl.trainer_checkpoint import TrainerCheckpoint

logger = logging.getLogger(__name__)


def cleanup_old_policies(checkpoint_dir: str, keep_last_n: int = 5) -> None:
    """Clean up old policy checkpoints, keeping only the most recent ones."""
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


def validate_policy_environment_match(policy: Any, env: Any) -> None:
    """Validate that policy's observation shape matches environment's."""
    # Extract agent from distributed wrapper if needed
    if isinstance(policy, MettaAgent):
        agent = policy
    elif isinstance(policy, DistributedMettaAgent):
        agent = policy.module
    else:
        raise ValueError(f"Policy must be of type MettaAgent or DistributedMettaAgent, got {type(policy)}")

    _env_shape = env.single_observation_space.shape
    environment_shape = tuple(_env_shape) if isinstance(_env_shape, list) else _env_shape

    # The rest of the validation logic continues to work with duck typing
    if hasattr(agent, "components"):
        found_match = False
        for component_name, component in agent.components.items():
            if hasattr(component, "_obs_shape"):
                found_match = True
                component_shape = (
                    tuple(component._obs_shape) if isinstance(component._obs_shape, list) else component._obs_shape
                )
                if component_shape != environment_shape:
                    raise ValueError(
                        f"Observation space mismatch error:\n"
                        f"[policy] component_name: {component_name}\n"
                        f"[policy] component_shape: {component_shape}\n"
                        f"environment_shape: {environment_shape}\n"
                    )

        if not found_match:
            raise ValueError(
                "No component with observation shape found in policy. "
                f"Environment observation shape: {environment_shape}"
            )


def wrap_agent_distributed(agent: Any, device: torch.device) -> Any:
    """Wrap agent in DistributedMettaAgent if distributed training is initialized.

    Args:
        agent: The agent to potentially wrap
        device: The device to use

    Returns:
        The agent, possibly wrapped in DistributedMettaAgent
    """
    if torch.distributed.is_initialized():
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


def maybe_load_checkpoint(
    run_dir: str,
    policy_store: Any,
    trainer_cfg: Any,
    metta_grid_env: Any,
    cfg: Any,
    device: torch.device,
    is_master: bool,
    rank: int,
) -> Tuple[Any | None, Any, int, int]:
    """Load checkpoint and policy if they exist, or create new ones.

    This unifies the checkpoint loading logic from trainer.py and run.py.

    Args:
        run_dir: Directory containing checkpoints
        policy_store: PolicyStore instance
        trainer_cfg: TrainerConfig with checkpoint settings
        metta_grid_env: MettaGridEnv instance for policy creation
        cfg: Full config for policy creation
        device: Device to load on
        is_master: Whether this is the master process
        rank: Process rank for distributed training

    Returns:
        Tuple of (checkpoint, policy_record, agent_step, epoch)
    """
    # Try to load checkpoint
    checkpoint = TrainerCheckpoint.load(run_dir)
    agent_step = 0
    epoch = 0

    if checkpoint:
        agent_step = checkpoint.agent_step
        epoch = checkpoint.epoch
        logger.info(f"Restored from checkpoint at {agent_step} steps")

    # Try to load policy from checkpoint
    if checkpoint and checkpoint.policy_path:
        logger.info(f"Loading policy from checkpoint: {checkpoint.policy_path}")
        policy_record = policy_store.policy_record(checkpoint.policy_path)

        # Restore original_feature_mapping from metadata if available
        if (
            hasattr(policy_record.policy, "restore_original_feature_mapping")
            and "original_feature_mapping" in policy_record.metadata
        ):
            policy_record.policy.restore_original_feature_mapping(policy_record.metadata["original_feature_mapping"])
            logger.info("Restored original_feature_mapping from checkpoint")

        return checkpoint, policy_record, agent_step, epoch

    # Try to load initial policy from config
    if trainer_cfg.initial_policy and trainer_cfg.initial_policy.uri:
        logger.info(f"Loading initial policy URI: {trainer_cfg.initial_policy.uri}")
        policy_record = policy_store.policy_record(trainer_cfg.initial_policy.uri)

        # Restore original_feature_mapping from metadata if available
        if (
            hasattr(policy_record.policy, "restore_original_feature_mapping")
            and "original_feature_mapping" in policy_record.metadata
        ):
            policy_record.policy.restore_original_feature_mapping(policy_record.metadata["original_feature_mapping"])
            logger.info("Restored original_feature_mapping from initial policy")

        return checkpoint, policy_record, agent_step, epoch

    # Check for existing policy at default path
    default_path = os.path.join(trainer_cfg.checkpoint.checkpoint_dir, policy_store.make_model_name(0))
    if os.path.exists(default_path):
        logger.info(f"Loading policy from default path: {default_path}")
        policy_record = policy_store.policy_record(default_path)

        # Restore original_feature_mapping from metadata if available
        if (
            hasattr(policy_record.policy, "restore_original_feature_mapping")
            and "original_feature_mapping" in policy_record.metadata
        ):
            policy_record.policy.restore_original_feature_mapping(policy_record.metadata["original_feature_mapping"])
            logger.info("Restored original_feature_mapping from default path")

        return checkpoint, policy_record, agent_step, epoch

    # Create new policy with distributed coordination
    if torch.distributed.is_initialized() and not is_master:
        # Non-master waits for master to create
        logger.info(f"Rank {rank}: Waiting for master to create policy at {default_path}")
        torch.distributed.barrier()

        if not wait_for_file(default_path, timeout=300):
            raise RuntimeError(f"Rank {rank}: Timeout waiting for policy at {default_path}")

        policy_record = policy_store.policy_record(default_path)

        # Restore original_feature_mapping from metadata if available
        if (
            hasattr(policy_record.policy, "restore_original_feature_mapping")
            and "original_feature_mapping" in policy_record.metadata
        ):
            policy_record.policy.restore_original_feature_mapping(policy_record.metadata["original_feature_mapping"])
            logger.info(f"Rank {rank}: Restored original_feature_mapping")

        return checkpoint, policy_record, agent_step, epoch
    else:
        # Master creates new policy
        name = policy_store.make_model_name(0)
        pr = policy_store.create_empty_policy_record(name)
        pr.policy = make_policy(metta_grid_env, cfg)
        saved_pr = policy_store.save(pr)
        logger.info(f"Created and saved new policy to {saved_pr.uri}")

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        return checkpoint, saved_pr, agent_step, epoch


def ensure_initial_policy(
    agent: Any,
    policy_store: Any,
    checkpoint_path: str,
    loaded_policy_path: str | None,
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
            # Extract the actual policy module from distributed wrapper if needed
            policy_to_save = agent
            if isinstance(agent, DistributedMettaAgent):
                policy_to_save = agent.module

            # Create policy record directly
            name = policy_store.make_model_name(0)
            policy_record = policy_store.create_empty_policy_record(name)
            policy_record.metadata = {
                "agent_step": 0,
                "epoch": 0,
                "initial": True,
            }
            policy_record.policy = policy_to_save

            # Save through policy store
            saved_policy_record = policy_store.save(policy_record)
            logger.info(f"Master saved initial policy to {saved_policy_record.uri}")

            # Master waits at barrier after saving
            torch.distributed.barrier()
        else:
            # Non-master ranks wait at barrier first
            torch.distributed.barrier()

            # Then load the policy master created
            default_policy_path = os.path.join(checkpoint_path, policy_store.make_model_name(0))
            if not wait_for_file(default_policy_path, timeout=300):
                raise RuntimeError(f"Rank {rank}: Timeout waiting for policy at {default_policy_path}")

            # Load the policy
            policy_pr = policy_store.policy_record(default_policy_path)
            agent.load_state_dict(policy_pr.policy.state_dict())  # type: ignore
    else:
        # Single GPU mode creates and saves initial policy
        # Create policy record directly
        name = policy_store.make_model_name(0)
        policy_record = policy_store.create_empty_policy_record(name)
        policy_record.metadata = {
            "agent_step": 0,
            "epoch": 0,
            "initial": True,
        }
        policy_record.policy = agent

        # Save through policy store
        saved_policy_record = policy_store.save(policy_record)
        logger.info(f"Saved initial policy to {saved_policy_record.uri}")


def load_or_initialize_policy(
    cfg: Any,
    checkpoint: Any | None,
    policy_store: Any,
    metta_grid_env: Any,
    device: torch.device,
    is_master: bool,
    rank: int,
) -> Tuple[Any, Any, Any]:
    """
    Load or initialize policy with distributed coordination.

    Returns:
        Tuple of (policy, initial_policy_record, latest_saved_policy_record)
    """
    trainer_cfg = cfg.trainer

    # Non-master ranks in distributed training
    if torch.distributed.is_initialized() and not is_master:
        # Non-master ranks wait for master to create and save the policy
        default_policy_path = os.path.join(trainer_cfg.checkpoint.checkpoint_dir, policy_store.make_model_name(0))
        logger.info(f"Rank {rank}: Waiting for master to create policy at {default_policy_path}")

        # Synchronize with master before attempting to load
        torch.distributed.barrier()

        def log_progress(elapsed: float, status: str) -> None:
            if status == "waiting" and int(elapsed) % 10 == 0 and elapsed > 0:
                logger.info(f"Rank {rank}: Still waiting for policy file... ({elapsed:.0f}s elapsed)")
            elif status == "found":
                logger.info(f"Rank {rank}: Policy file found, waiting for write to complete...")
            elif status == "stable":
                logger.info(f"Rank {rank}: Policy file stable after {elapsed:.1f}s")

        if not wait_for_file(default_policy_path, timeout=300, progress_callback=log_progress):
            raise RuntimeError(f"Rank {rank}: Timeout waiting for policy at {default_policy_path}")

        try:
            policy_record = policy_store.policy_record(default_policy_path)
        except Exception as e:
            raise RuntimeError(f"Rank {rank}: Failed to load policy from {default_policy_path}: {e}") from e

        policy = policy_record.policy
        initial_policy_record = policy_record
        latest_saved_policy_record = policy_record

        logger.info(f"Rank {rank}: Successfully loaded policy from {default_policy_path}")

    # Master rank or single GPU
    else:
        policy_record = None

        # Try checkpoint first
        if checkpoint and checkpoint.policy_path:
            logger.info(f"Loading policy from checkpoint: {checkpoint.policy_path}")
            policy_record = policy_store.policy_record(checkpoint.policy_path)

        # Try initial_policy from config
        elif trainer_cfg.initial_policy and trainer_cfg.initial_policy.uri:
            logger.info(f"Loading initial policy URI: {trainer_cfg.initial_policy.uri}")
            policy_record = policy_store.policy_record(trainer_cfg.initial_policy.uri)

        # Try default checkpoint path
        else:
            default_path = os.path.join(trainer_cfg.checkpoint.checkpoint_dir, policy_store.make_model_name(0))
            if os.path.exists(default_path):
                logger.info(f"Loading policy from default path: {default_path}")
                policy_record = policy_store.policy_record(default_path)

        if policy_record:
            # Restore original_feature_mapping from metadata if available
            if (
                hasattr(policy_record.policy, "restore_original_feature_mapping")
                and "original_feature_mapping" in policy_record.metadata
            ):
                policy_record.policy.restore_original_feature_mapping(
                    policy_record.metadata["original_feature_mapping"]
                )
                logger.info("Restored original_feature_mapping")

            policy = policy_record.policy
            initial_policy_record = policy_record
            latest_saved_policy_record = policy_record
        else:
            # Create new policy
            logger.info("No existing policy found, creating new one")
            name = policy_store.make_model_name(0)
            pr = policy_store.create_empty_policy_record(name)
            pr.policy = make_policy(metta_grid_env, cfg)
            saved_pr = policy_store.save(pr)
            logger.info(f"Created and saved new policy to {saved_pr.uri}")

            policy = saved_pr.policy
            initial_policy_record = saved_pr
            latest_saved_policy_record = saved_pr

            # Synchronize with non-master ranks after saving
            if torch.distributed.is_initialized():
                logger.info(f"Master rank: Policy saved to {saved_pr.uri}, synchronizing with other ranks")
                torch.distributed.barrier()

    # Don't initialize policy to environment here - it should be done after distributed wrapping
    logger.info(f"Rank {rank}: USING {initial_policy_record.uri if initial_policy_record else 'new policy'}")

    return policy, initial_policy_record, latest_saved_policy_record
