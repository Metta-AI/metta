"""
PufferLib checkpoint loading utilities.
"""

import logging
from typing import Any, Dict, TypeGuard

import torch

from metta.agent.policies.puffer import PufferPolicy, PufferPolicyConfig

logger = logging.getLogger(__name__)


def _is_puffer_state_dict(loaded_obj: Any) -> TypeGuard[Dict[str, torch.Tensor]]:
    """Return True if the object appears to be a PufferLib state_dict."""
    return isinstance(loaded_obj, dict) and bool(loaded_obj) and any(key.startswith("policy.") for key in loaded_obj)


def _create_metta_agent(device: str | torch.device = "cpu") -> Any:
    """Instantiate a PufferLib-compatible Metta policy for checkpoint loading."""

    from mettagrid import MettaGridEnv
    from mettagrid.builder.envs import make_arena

    # Minimal environment for initialization
    env_cfg = make_arena(num_agents=60)
    temp_env = MettaGridEnv(env_cfg, render_mode="rgb_array")

    policy_cfg = PufferPolicyConfig()
    policy = PufferPolicy(temp_env, policy_cfg).to(device)
    print("Policy keys:", sorted(policy.state_dict().keys()))

    temp_env.close()
    return policy


def load_pufferlib_checkpoint(checkpoint_data: Any, device: str | torch.device = "cpu") -> Any:
    """Load a PufferLib checkpoint into a Metta policy."""
    logger.info("Loading checkpoint in PufferLib state_dict format")
    if not isinstance(checkpoint_data, dict):
        raise TypeError("Expected checkpoint_data to be a dict (state_dict format)")

    logger.debug(f"Checkpoint sample keys: {list(checkpoint_data.keys())[:10]}")
    policy = _create_metta_agent(device)

    # Load directly - policy structure now matches checkpoint exactly
    policy.load_state_dict(checkpoint_data, strict=False)
    logger.info("Successfully loaded checkpoint into Metta policy")

    return policy
