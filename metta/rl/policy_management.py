"""Policy management utilities for Metta."""

import logging
import os
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig

from metta.agent.metta_agent import DistributedMettaAgent, MettaAgent, PolicyAgent, make_policy
from metta.agent.policy_record import PolicyRecord
from metta.agent.policy_store import PolicyStore
from metta.common.util.fs import wait_for_file
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.rl.trainer_checkpoint import TrainerCheckpoint
from metta.rl.trainer_config import TrainerConfig

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


def validate_policy_environment_match(policy: PolicyAgent, env: MettaGridEnv) -> None:
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


def wrap_agent_distributed(agent: PolicyAgent, device: torch.device) -> PolicyAgent:
    if torch.distributed.is_initialized():
        # Always use DistributedMettaAgent for its __getattr__ forwarding
        agent = DistributedMettaAgent(agent, device)

    return agent


def maybe_load_checkpoint(
    run_dir: str,
    policy_store: PolicyStore,
    trainer_cfg: TrainerConfig,
    metta_grid_env: MettaGridEnv,
    cfg: DictConfig,
    is_master: bool,
    rank: int,
) -> tuple[TrainerCheckpoint | None, PolicyRecord, int, int]:
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


def load_or_initialize_policy(
    cfg: DictConfig,
    checkpoint: TrainerCheckpoint | None,
    policy_store: PolicyStore,
    metta_grid_env: MettaGridEnv,
    is_master: bool,
    rank: int,
) -> tuple[MettaAgent | DistributedMettaAgent, PolicyRecord | None, PolicyRecord | None]:
    """
    Load or initialize policy with distributed coordination.
    This is called from all ranks.
    """
    policy, initial_policy_record, latest_saved_policy_record = None, None, None

    if is_master:
        policy, initial_policy_record, latest_saved_policy_record = load_or_initialize_policy_master(
            cfg, checkpoint, policy_store, metta_grid_env
        )

    if torch.distributed.is_initialized():
        # Non-master ranks create a throwaway policy instance to receive the broadcasted state
        # during the DDP wrapping of the policy.
        policy = make_policy(metta_grid_env, cfg)

    # cast to the correct type
    policy = cast(MettaAgent | DistributedMettaAgent, policy)
    initial_policy_record = cast(PolicyRecord | None, initial_policy_record)
    latest_saved_policy_record = cast(PolicyRecord | None, latest_saved_policy_record)

    return policy, initial_policy_record, latest_saved_policy_record


def load_or_initialize_policy_master(
    cfg: Any,
    checkpoint: Any | None,
    policy_store: PolicyStore,
    metta_grid_env: Any,
) -> tuple[MettaAgent | DistributedMettaAgent, PolicyRecord, PolicyRecord]:
    """
    Load or initialize policy with distributed coordination.
    This is only called from the master rank.

    Returns:
        Tuple of (policy, initial_policy_record, latest_saved_policy_record)
    """
    trainer_cfg = cfg.trainer

    # Check if policy already exists at default path - all ranks check this
    default_path = os.path.join(trainer_cfg.checkpoint.checkpoint_dir, policy_store.make_model_name(0))

    # First priority: checkpoint
    policy_record: PolicyRecord | None = None
    if checkpoint and checkpoint.policy_path:
        logger.info(f"Loading policy from checkpoint: {checkpoint.policy_path}")
        policy_record = policy_store.policy_record(checkpoint.policy_path)
    # Second priority: initial_policy from config
    elif trainer_cfg.initial_policy and trainer_cfg.initial_policy.uri:
        logger.info(f"Loading initial policy URI: {trainer_cfg.initial_policy.uri}")
        policy_record = policy_store.policy_record(trainer_cfg.initial_policy.uri)
    # Third priority: existing default path
    elif os.path.exists(default_path):
        logger.info(f"Loading policy from default path: {default_path}")
        policy_record = policy_store.policy_record(default_path)
    else:
        policy_record = None

    # If we found an existing policy, all ranks use it
    if policy_record:
        # Restore original_feature_mapping from metadata if available
        if isinstance(policy_record.policy, MettaAgent) and "original_feature_mapping" in policy_record.metadata:
            policy_record.policy.restore_original_feature_mapping(policy_record.metadata["original_feature_mapping"])
            logger.info("Restored original_feature_mapping")

        policy = policy_record.policy
        initial_policy_record = policy_record
        latest_saved_policy_record = policy_record

        return policy, initial_policy_record, latest_saved_policy_record

    # Master creates new policy
    logger.info("No existing policy found, creating new one")
    name = policy_store.make_model_name(0)
    pr = policy_store.create_empty_policy_record(name)
    pr.policy = make_policy(metta_grid_env, cfg)
    saved_pr = policy_store.save(pr)
    logger.info(f"Created and saved new policy to {saved_pr.uri}")

    policy = saved_pr.policy
    initial_policy_record = saved_pr
    latest_saved_policy_record = saved_pr

    return policy, initial_policy_record, latest_saved_policy_record
