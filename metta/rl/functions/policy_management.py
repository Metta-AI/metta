"""Policy management utilities for Metta."""

import logging
import os
from pathlib import Path
from typing import Any, Optional

import torch

from metta.eval.eval_request_config import EvalRewardSummary

logger = logging.getLogger(__name__)


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


def save_policy_with_metadata(
    policy: Any,
    policy_store: Any,
    epoch: int,
    agent_step: int,
    evals: EvalRewardSummary,
    timer: Any,
    vecenv: Any,
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
    from torch.nn.parallel import DistributedDataParallel

    from metta.agent.metta_agent import DistributedMettaAgent

    policy_to_save = policy
    if isinstance(policy, DistributedMettaAgent):
        policy_to_save = policy.module
    elif isinstance(policy, DistributedDataParallel):
        policy_to_save = policy.module

    # Build metadata
    name = policy_store.make_model_name(epoch)
    # Get curriculum task distribution from vecenv
    task_probs = vecenv.driver_env.curriculum.get_task_probs() if hasattr(vecenv, "driver_env") else {}

    # Extract average reward from evals
    avg_reward = evals.avg_category_score if evals else 0.0

    metadata = {
        "epoch": epoch,
        "agent_step": agent_step,
        "total_time": timer.get_elapsed(),
        "total_train_time": timer.get_all_elapsed().get("_rollout", 0) + timer.get_all_elapsed().get("_train", 0),
        "run": run_name,
        "initial_pr": initial_policy_record.uri if initial_policy_record else None,
        "generation": initial_policy_record.metadata.get("generation", 0) + 1 if initial_policy_record else 0,
        "curriculum_task_probs": task_probs,
        # Convert EvalRewardSummary to dict for metadata storage
        "evals": {
            "category_scores": evals.category_scores,
            "simulation_scores": {f"{cat}/{sim}": score for (cat, sim), score in evals.simulation_scores.items()},
            "avg_category_score": evals.avg_category_score,
            "avg_simulation_score": evals.avg_simulation_score,
        }
        if evals
        else {},
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
    from metta.agent.metta_agent import DistributedMettaAgent, MettaAgent

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
    from metta.api.checkpoint import save_checkpoint

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
