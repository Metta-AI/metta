"""Policy management functions for Metta training."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch

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
    evals: Dict[str, float],
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

    from metta.agent.metta_agent import DistributedMettaAgent
    from metta.agent.policy_metadata import PolicyMetadata
    from metta.mettagrid.mettagrid_env import MettaGridEnv

    name = policy_store.make_model_name(epoch)

    metta_grid_env: MettaGridEnv = vecenv.driver_env
    assert isinstance(metta_grid_env, MettaGridEnv), "vecenv.driver_env must be a MettaGridEnv"

    training_time = timer.get_elapsed("_rollout") + timer.get_elapsed("_train")

    category_scores_map = {key.split("/")[0]: value for key, value in evals.items() if key.endswith("/score")}
    category_score_values = [v for k, v in category_scores_map.items()]
    overall_score = sum(category_score_values) / len(category_score_values) if category_score_values else 0

    metadata = PolicyMetadata(
        agent_step=agent_step,
        epoch=epoch,
        run=run_name,
        action_names=metta_grid_env.action_names,
        generation=initial_policy_record.metadata.get("generation", 0) + 1 if initial_policy_record else 0,
        initial_uri=initial_policy_record.uri if initial_policy_record else None,
        train_time=training_time,
        score=overall_score,
        eval_scores=category_scores_map,
    )

    # Extract actual policy from distributed wrapper
    policy_to_save = policy
    if isinstance(policy, DistributedMettaAgent):
        policy_to_save = policy.module

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

    saved_record = policy_store.save(policy_record)
    logger.info(f"Successfully saved policy at epoch {epoch}")

    return saved_record


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
                component_shape = component._obs_shape
                # Convert both shapes to tuples for comparison
                component_shape_tuple = tuple(component_shape) if isinstance(component_shape, list) else component_shape
                environment_shape_tuple = (
                    tuple(environment_shape) if isinstance(environment_shape, list) else environment_shape
                )

                if component_shape_tuple != environment_shape_tuple:
                    raise ValueError(
                        f"Component '{component_name}' observation shape {component_shape} "
                        f"does not match environment shape {environment_shape}"
                    )

        if not found_match:
            logger.warning("No components with _obs_shape found for validation")
    else:
        logger.warning("Agent has no components attribute for shape validation")


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
