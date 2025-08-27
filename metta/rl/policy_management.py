"""Policy management utilities for Metta."""

import logging
from pathlib import Path

import torch

from metta.agent.metta_agent import DistributedMettaAgent, MettaAgent, PolicyAgent
from metta.mettagrid.mettagrid_env import MettaGridEnv

logger = logging.getLogger(__name__)


def initialize_policy_for_environment(
    policy: PolicyAgent,
    metta_grid_env: MettaGridEnv,
    device: torch.device,
    restore_feature_mapping: bool = True,
    metadata: dict | None = None,
) -> None:
    # Restore original_feature_mapping from metadata if available
    if restore_feature_mapping and hasattr(policy, "restore_original_feature_mapping"):
        if metadata and "original_feature_mapping" in metadata:
            policy.restore_original_feature_mapping(metadata["original_feature_mapping"])
            logger.info("Restored original_feature_mapping")

    # Initialize policy to environment
    features = metta_grid_env.get_observation_features()
    policy.initialize_to_environment(features, metta_grid_env.action_names, metta_grid_env.max_action_args, device)


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
    agent = _validate_and_get_agent(policy)
    env_shape = _to_tuple_shape(env.single_observation_space.shape)
    validate_components_match(agent, env_shape)


def validate_components_match(agent: PolicyAgent, environment_shape) -> None:
    """Validate that policy's components match environment's."""
    if not hasattr(agent, "components"):
        return

    for component_name, component in agent.components.items():
        if not hasattr(component, "_obs_shape"):
            continue

        component_shape = _to_tuple_shape(component._obs_shape)
        if component_shape != environment_shape:
            raise ValueError(
                f"Observation space mismatch: component '{component_name}' has shape {component_shape}, "
                f"but environment expects {environment_shape}"
            )
        return  # Found a matching component

    # No component with observation shape found
    raise ValueError(f"No component with observation shape found in policy. Environment shape: {environment_shape}")


def _validate_and_get_agent(policy: PolicyAgent) -> PolicyAgent:
    """Extract the underlying agent from distributed wrappers."""
    if isinstance(policy, MettaAgent):
        return policy
    if isinstance(policy, DistributedMettaAgent):
        return policy.module
    raise ValueError(f"Policy must be MettaAgent or DistributedMettaAgent, got {type(policy)}")


def _to_tuple_shape(shape) -> tuple:
    """Convert shape to tuple, handling both list and tuple inputs."""
    return tuple(shape) if isinstance(shape, list) else shape


def wrap_agent_distributed(agent: PolicyAgent, device: torch.device) -> PolicyAgent:
    if torch.distributed.is_initialized():
        # Always use DistributedMettaAgent for its __getattr__ forwarding
        agent = DistributedMettaAgent(agent, device)

    return agent
