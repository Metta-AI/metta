"""Policy management utilities for Metta."""

import logging
import os
from pathlib import Path

import torch
from omegaconf import DictConfig

from metta.agent.metta_agent import DistributedMettaAgent, MettaAgent, PolicyAgent, make_policy
from metta.agent.policy_record import PolicyRecord
from metta.agent.policy_store import PolicyStore
from metta.common.util.fs import wait_for_file
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.rl.env_config import EnvConfig
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

    elif type(policy).__name__ == "Recurrent":
        agent = policy
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


def load_or_initialize_policy(
    agent_cfg: DictConfig,
    env_cfg: EnvConfig,
    trainer_cfg: TrainerConfig,
    checkpoint: TrainerCheckpoint | None,
    policy_store: PolicyStore,
    metta_grid_env: MettaGridEnv,
    is_master: bool,
    rank: int,
) -> PolicyRecord:
    """
    Load or initialize policy with distributed coordination.

    First, checks if there is an existing policy at any of:
        - checkpoint.policy_path
        - trainer_cfg.initial_policy.uri
        - default_path (checkpoint_dir/model_{epoch}.pt)
    If so, restore the original feature mapping (if indicated in policy metadata), and return

    If not, then distributed workers wait until the master creates the policy at default_path,
    and the master creates a new policy record and saves it to default_path.
    """

    # Check if policy already exists at default path - all ranks check this
    default_model_name = policy_store.make_model_name(0)
    default_path = os.path.join(trainer_cfg.checkpoint.checkpoint_dir, default_model_name)

    # First priority: checkpoint
    policy_record: PolicyRecord | None = None
    policy_path: str | None = (
        (checkpoint and checkpoint.policy_path)
        or (trainer_cfg.initial_policy and trainer_cfg.initial_policy.uri)
        or (default_path if os.path.exists(default_path) else None)
    )
    if policy_path:
        logger.info(f"Loading policy from {policy_path}")
        policy_record = policy_store.policy_record(policy_path)

        # Restore original_feature_mapping from metadata if available
        if isinstance(policy_record.policy, MettaAgent) and "original_feature_mapping" in policy_record.metadata:
            policy_record.policy.restore_original_feature_mapping(policy_record.metadata["original_feature_mapping"])
            logger.info("Restored original_feature_mapping")
        return policy_record

    # No existing policy found - need to create one with distributed coordination
    if torch.distributed.is_initialized() and not is_master:
        # Non-master waits for master to create
        logger.info(f"Rank {rank}: Waiting for master to create policy at {default_path}")
        torch.distributed.barrier()

        def log_progress(elapsed: float, status: str) -> None:
            if status == "waiting" and int(elapsed) % 10 == 0 and elapsed > 0:
                logger.info(f"Rank {rank}: Still waiting for policy file... ({elapsed:.0f}s elapsed)")
            elif status == "found":
                logger.info(f"Rank {rank}: Policy file found, waiting for write to complete...")
            elif status == "stable":
                logger.info(f"Rank {rank}: Policy file stable after {elapsed:.1f}s")

        if not wait_for_file(default_path, timeout=300, progress_callback=log_progress):
            raise RuntimeError(f"Rank {rank}: Timeout waiting for policy at {default_path}")

        try:
            policy_record = policy_store.policy_record(default_path)
        except Exception as e:
            raise RuntimeError(f"Rank {rank}: Failed to load policy from {default_path}: {e}") from e
    else:
        # Master creates new policy
        logger.info("No existing policy found, creating new one")
        new_policy_record = policy_store.create_empty_policy_record(
            checkpoint_dir=trainer_cfg.checkpoint.checkpoint_dir, name=default_model_name
        )
        new_policy_record.policy = make_policy(metta_grid_env, env_cfg, agent_cfg)
        policy_record = policy_store.save(new_policy_record)
        logger.info(f"Created and saved new policy to {policy_record.uri}")

        # Synchronize with non-master ranks after saving
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    return policy_record
