"""Policy management utilities for Metta."""

import logging
import os
from pathlib import Path
from typing import Any, Optional, Tuple

import torch

from metta.agent.metta_agent import DistributedMettaAgent, MettaAgent
from metta.agent.policy_metadata import PolicyMetadata
from metta.agent.policy_record import PolicyRecord

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


def save_policy_with_metadata(
    policy: Any,
    policy_store: Any,
    epoch: int,
    agent_step: int,
    evals: Any,  # EvalRewardSummary
    timer: Any,
    initial_policy_record: Optional[Any],
    run_name: str,
    is_master: bool = True,
) -> Optional[Any]:
    """Save policy with metadata.

    Returns:
        Saved policy record or None if not master
    """
    if not is_master:
        return None

    logger.info(f"Saving policy at epoch {epoch}")

    # Extract the actual policy module from distributed wrapper if needed
    policy_to_save = policy
    if isinstance(policy, DistributedMettaAgent):
        policy_to_save = policy.module

    # Build metadata
    name = policy_store.make_model_name(epoch)

    # Extract average reward from evals
    # Handle both EvalRewardSummary object and dict
    avg_reward = 0.0
    evals_dict = {}
    if evals:
        if hasattr(evals, "avg_category_score"):
            # It's an EvalRewardSummary object
            avg_reward = evals.avg_category_score if evals.avg_category_score is not None else 0.0
            evals_dict = {
                "category_scores": evals.category_scores,
                "simulation_scores": {f"{cat}/{sim}": score for (cat, sim), score in evals.simulation_scores.items()},
                "avg_category_score": evals.avg_category_score,
                "avg_simulation_score": evals.avg_simulation_score,
            }
        else:
            # It's a dict
            avg_reward = sum(v for k, v in evals.items() if k.endswith("/score")) / max(
                1, len([k for k in evals.keys() if k.endswith("/score")])
            )
            evals_dict = evals

    metadata = {
        "epoch": epoch,
        "agent_step": agent_step,
        "total_time": timer.get_elapsed(),
        "total_train_time": timer.get_all_elapsed().get("_rollout", 0) + timer.get_all_elapsed().get("_train", 0),
        "run": run_name,
        "initial_pr": initial_policy_record.uri if initial_policy_record else None,
        "generation": initial_policy_record.metadata.get("generation", 0) + 1 if initial_policy_record else 0,
        "evals": evals_dict,
        "avg_reward": avg_reward,
    }

    # Save original feature mapping
    if hasattr(policy_to_save, "get_original_feature_mapping"):
        original_feature_mapping = policy_to_save.get_original_feature_mapping()
        if original_feature_mapping is not None:
            metadata["original_feature_mapping"] = original_feature_mapping
            logger.info(f"Saving original_feature_mapping with {len(original_feature_mapping)} features to metadata")

    # Create and save policy record
    policy_record = policy_store.create_empty_policy_record(name)
    policy_record.metadata = metadata
    policy_record.policy = policy_to_save

    saved_policy_record = policy_store.save(policy_record)
    logger.info(f"Successfully saved policy at epoch {epoch}")

    return saved_policy_record


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


def maybe_load_checkpoint(
    run_dir: str,
    policy_store: Any,
    trainer_cfg: Any,
    metta_grid_env: Any,
    cfg: Any,
    device: torch.device,
    is_master: bool,
    rank: int,
) -> Tuple[Optional[Any], Any, int, int]:
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
    from metta.agent.metta_agent import make_policy
    from metta.common.util.fs import wait_for_file
    from metta.rl.trainer_checkpoint import TrainerCheckpoint

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
    from metta.agent.metta_agent import DistributedMettaAgent
    from metta.common.util.fs import wait_for_file

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
    checkpoint: Optional[Any],
    policy_store: Any,
    metta_grid_env: Any,
    device: torch.device,
    is_master: bool,
    rank: int,
) -> Tuple[Any, Any, Any]:
    """Load or initialize policy with distributed coordination."""
    # Use existing maybe_load_checkpoint to handle core loading/creation
    checkpoint, policy_record, _, _ = maybe_load_checkpoint(
        run_dir=cfg.run_dir,
        policy_store=policy_store,
        trainer_cfg=cfg.trainer,
        metta_grid_env=metta_grid_env,
        cfg=cfg,
        device=device,
        is_master=is_master,
        rank=rank,
    )

    policy = policy_record.policy

    # Broadcast metadata from master to workers
    if torch.distributed.is_initialized():
        broadcast_obj = [None, None, None]
        if is_master:
            broadcast_obj = [dict(policy_record.metadata), policy_record.uri, policy_record.run_name]
        torch.distributed.broadcast_object_list(broadcast_obj, src=0)
        metadata_dict, uri, run_name = broadcast_obj

        if run_name is None:
            raise RuntimeError("Failed to receive run_name from master broadcast")
        if uri is None:
            raise RuntimeError("Failed to receive uri from master broadcast")

        metadata = PolicyMetadata.from_dict(metadata_dict) if metadata_dict else PolicyMetadata()
        if not is_master:
            policy_record = PolicyRecord(policy_store, run_name, uri, metadata)
    else:
        metadata = policy_record.metadata if policy_record else PolicyMetadata()

    # Restore feature mapping
    mapping = metadata.get("original_feature_mapping")
    if mapping and hasattr(policy, "restore_original_feature_mapping"):
        policy.restore_original_feature_mapping(mapping)

    # Initialize policy to environment
    _initialize_policy_to_environment(policy, metta_grid_env, device)

    initial_policy_record = policy_record
    latest_saved_policy_record = policy_record

    if torch.distributed.is_initialized() and is_master:
        torch.distributed.barrier()

    logger.info(f"Rank {rank}: USING {initial_policy_record.uri}")

    return policy, initial_policy_record, latest_saved_policy_record


def _initialize_policy_to_environment(policy, metta_grid_env, device):
    """Helper method to initialize a policy to the environment using the appropriate interface."""
    if hasattr(policy, "initialize_to_environment"):
        features = metta_grid_env.get_observation_features()
        policy.initialize_to_environment(features, metta_grid_env.action_names, metta_grid_env.max_action_args, device)
    else:
        policy.activate_actions(metta_grid_env.action_names, metta_grid_env.max_action_args, device)
