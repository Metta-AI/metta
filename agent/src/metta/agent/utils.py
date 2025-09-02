"""Utility functions for agent operations."""

import logging

import numpy as np
import torch
from tensordict import TensorDict
from torch import nn

logger = logging.getLogger(__name__)


def log_on_master(*args, **argv):
    """Log messages only on the master process in distributed training."""
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger.info(*args, **argv)


def obs_to_td(obs: np.ndarray, device: str | torch.device = "cpu") -> TensorDict:
    """Convert numpy observations to TensorDict with 'env_obs' key."""
    return TensorDict({"env_obs": torch.from_numpy(obs).to(device)}, batch_size=obs.shape[0])


def is_old_checkpoint_format(state: dict) -> bool:
    """Check if this is an old checkpoint format (has components but no policy)."""
    has_components = "components" in state or ("_modules" in state and "components" in state.get("_modules", {}))
    has_policy = "policy" in state or ("_modules" in state and "policy" in state.get("_modules", {}))
    return has_components and not has_policy


def convert_old_checkpoint(state: dict) -> tuple[dict, object]:
    """Convert old checkpoint format to new ComponentPolicy structure.

    Returns:
        tuple: (agent_state, policy) - cleaned agent state and converted policy
    """
    logger.info("Detected old checkpoint format - converting to new ComponentPolicy structure")

    # Remove circular reference if it exists
    if "policy" in state and state.get("policy") is state:
        del state["policy"]
        log_on_master("Removed circular reference: state['policy'] = state")

    # Create policy with components
    policy = _create_policy_from_old_checkpoint(state)

    # Extract clean agent state (excluding components)
    agent_state = _extract_agent_state(state)

    log_on_master("Successfully converted old checkpoint to new structure")
    return agent_state, policy


def _create_policy_from_old_checkpoint(state: dict):
    """Create policy object from old checkpoint state."""
    from metta.agent.component_policies.fast import Fast

    # Create policy without calling __init__ to avoid rebuilding components
    policy = Fast.__new__(Fast)
    nn.Module.__init__(policy)

    # Extract and transfer components
    components = _extract_components(state)
    policy.components = components

    # Transfer component-related attributes to policy
    # Note: cum_action_max_params and action_index_tensor are also needed by MettaAgent
    # and will be restored there separately in _extract_agent_state
    component_attrs = {
        "components_with_memory": [],
        "clip_range": 0,
        "agent_attributes": {},
        "cum_action_max_params": None,
        "action_index_tensor": None,
        "cfg": None,
    }

    for attr, default in component_attrs.items():
        setattr(policy, attr, state.get(attr, default))

    logger.info("Converting old checkpoint to Fast agent")
    return policy


def _extract_components(state: dict):
    """Extract components from old checkpoint state."""
    if "components" in state:
        return state["components"]
    elif "_modules" in state and "components" in state["_modules"]:
        return state["_modules"]["components"]
    else:
        return nn.ModuleDict()


def _extract_agent_state(state: dict) -> dict:
    """Extract MettaAgent-specific state from old checkpoint, excluding components."""
    # Component-related keys that should be excluded from MettaAgent
    excluded_keys = {"components", "_modules"}

    # MettaAgent-specific attributes
    agent_keys = {
        # Core agent attributes
        "obs_width",
        "obs_height",
        "action_space",
        "feature_normalizations",
        "device",
        "obs_space",
        "_total_params",
        "cfg",
        "training",
        # Feature mapping attributes (used by MettaAgent.initialize_to_environment)
        "feature_id_to_name",
        "original_feature_mapping",
        # Action attributes (used by MettaAgent action conversion methods)
        "action_names",
        "action_max_params",
        "cum_action_max_params",
        "action_index_tensor",
    }

    # PyTorch module internals
    pytorch_internals = {k for k in state.keys() if k.startswith("_") and k not in excluded_keys}
    agent_keys.update(pytorch_internals)

    return {k: v for k, v in state.items() if k in agent_keys}
