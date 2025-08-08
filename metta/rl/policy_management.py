"""Policy management utilities for Metta."""

import logging
from pathlib import Path

import torch

from metta.agent.metta_agent import DistributedMettaAgent, MettaAgent, PolicyAgent
from metta.agent.metta_agent_builder import MettaAgentBuilder
from metta.agent.policy_record import PolicyRecord
from metta.mettagrid.mettagrid_env import MettaGridEnv

logger = logging.getLogger(__name__)


def initialize_policy_for_environment(
    policy_record: PolicyRecord,
    metta_grid_env: MettaGridEnv,
    device: torch.device,
    restore_feature_mapping: bool = True,
) -> None:
    policy = policy_record.policy

    # Restore original_feature_mapping from metadata if available
    if restore_feature_mapping and hasattr(policy, "restore_original_feature_mapping"):
        if "original_feature_mapping" in policy_record.metadata:
            policy.restore_original_feature_mapping(policy_record.metadata["original_feature_mapping"])
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

    logger.info(f"Env Match {env}")

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
