"""
PufferLib checkpoint loading utilities.
"""

import logging
from typing import Dict, TypeGuard

import torch

from metta.agent.policies.puffer import PufferPolicy, PufferPolicyConfig

logger = logging.getLogger("puffer_policy")


def _is_puffer_state_dict(loaded_obj) -> TypeGuard[Dict[str, torch.Tensor]]:
    """Return True if the object appears to be a PufferLib state_dict."""
    return isinstance(loaded_obj, dict) and bool(loaded_obj) and any(key.startswith("policy.") for key in loaded_obj)


def _create_metta_agent(device: str | torch.device = "cpu"):
    """Instantiate a PufferLib-compatible Metta policy for checkpoint loading."""

    from mettagrid import MettaGridEnv
    from mettagrid.builder.envs import make_arena

    env_cfg = make_arena(num_agents=60)
    temp_env = MettaGridEnv(env_cfg, render_mode="rgb_array")

    policy_cfg = PufferPolicyConfig()
    policy = PufferPolicy(temp_env, policy_cfg).to(device)
    temp_env.close()
    return policy


def load_pufferlib_checkpoint(checkpoint_data, device: str | torch.device = "cpu"):
    """Load a PufferLib checkpoint into a Metta policy."""
    logger.info("Loading checkpoint in PufferLib state_dict format")
    if not isinstance(checkpoint_data, dict):
        raise TypeError("Expected checkpoint_data to be a dict (state_dict format)")
    policy = _create_metta_agent(device)

    policy.load_state_dict(checkpoint_data, strict=False)
    logger.info("Successfully loaded PufferLib checkpoint into Metta policy")

    return policy
