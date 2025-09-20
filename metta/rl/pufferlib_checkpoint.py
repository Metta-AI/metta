"""
PufferLib checkpoint loading and conversion utilities.

This module handles the integration between PufferLib checkpoints (state_dict format)
and Metta agents, providing a clean abstraction for checkpoint format detection
and conversion.
"""

import logging
from typing import Any, Dict, TypeGuard

import torch

logger = logging.getLogger(__name__)


def _is_state_dict(loaded_obj: Any) -> TypeGuard[Dict[str, torch.Tensor]]:
    """Check if object is a PufferLib state_dict."""
    if not isinstance(loaded_obj, dict) or not loaded_obj:
        return False

    # Check first 5 items to see if they're parameter name -> tensor mappings
    sample_items = list(loaded_obj.items())[:5]
    for key, value in sample_items:
        if not (isinstance(key, str) and torch.is_tensor(value)):
            return False
    return True


def _is_metta_agent(loaded_obj: Any) -> bool:
    """Check if object is a Metta agent."""
    return hasattr(loaded_obj, "policy") and hasattr(loaded_obj, "forward")


def _preprocess_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove PufferLib-specific prefixes from state dict keys."""
    processed = {}
    for key, value in state_dict.items():
        # Remove PufferLib-specific prefixes
        if key.startswith("fast_policy."):
            key = key[len("fast_policy.") :]
        processed[key] = value

    logger.info(f"Preprocessed state_dict: {len(state_dict)} -> {len(processed)} parameters")
    return processed


def _create_metta_agent(device: str | torch.device = "cpu") -> Any:
    """Create a MettaAgent with default configuration."""
    from metta.agent.agent_config import AgentConfig
    from metta.agent.metta_agent import MettaAgent
    from metta.mettagrid.builder.envs import make_arena
    from metta.mettagrid.mettagrid_env import MettaGridEnv
    from metta.rl.system_config import SystemConfig

    # Create minimal environment for agent initialization
    env_cfg = make_arena(num_agents=60)
    temp_env = MettaGridEnv(env_cfg, render_mode="rgb_array")

    system_cfg = SystemConfig(device=str(device))
    agent_cfg = AgentConfig(name="pytorch/fast")

    # Create the agent
    agent = MettaAgent(temp_env, system_cfg, agent_cfg)
    temp_env.close()

    return agent


def _load_state_dict_into_agent(agent: Any, state_dict: Dict[str, torch.Tensor]) -> Any:
    """Load state dict into agent, handling compatibility issues."""
    agent_state = agent.policy.state_dict()
    compatible_state = {}

    keys_matched = 0
    for key, value in state_dict.items():
        if key in agent_state:
            compatible_state[key] = value
            keys_matched += 1
        else:
            logger.debug(f"Skipping incompatible parameter: {key}")

    logger.info(f"Loaded {keys_matched}/{len(state_dict)} compatible parameters")

    if compatible_state:
        agent.policy.load_state_dict(compatible_state, strict=False)
    else:
        logger.warning("No compatible parameters found - proceeding with random initialization")

    return agent


class PufferLibCheckpoint:
    """Simple checkpoint loader supporting both Metta and PufferLib formats."""

    def load_checkpoint(self, checkpoint_data: Any, device: str | torch.device = "cpu") -> Any:
        """Load checkpoint data and return an agent, auto-detecting format."""

        # If it's already a Metta agent, return it directly
        if _is_metta_agent(checkpoint_data):
            logger.info("Loading native Metta checkpoint format")
            return checkpoint_data

        # If it's a PufferLib state_dict, create agent and load weights
        if _is_state_dict(checkpoint_data):
            logger.info("Loading PufferLib checkpoint format (state_dict)")
            agent = _create_metta_agent(device)
            processed_state_dict = _preprocess_state_dict(checkpoint_data)
            return _load_state_dict_into_agent(agent, processed_state_dict)

        raise ValueError(
            f"Unrecognized checkpoint format. Expected Metta agent or PufferLib state_dict, got {type(checkpoint_data)}"
        )

    def is_pufferlib_format(self, checkpoint_data: Any) -> bool:
        """Check if checkpoint data is in PufferLib format."""
        return _is_state_dict(checkpoint_data)

    def is_metta_format(self, checkpoint_data: Any) -> bool:
        """Check if checkpoint data is in native Metta format."""
        return _is_metta_agent(checkpoint_data)
