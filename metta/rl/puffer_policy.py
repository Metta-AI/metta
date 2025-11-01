"""
PufferLib checkpoint loading utilities.
"""

import logging
from typing import Dict, TypeGuard

import torch

from metta.agent.policies.puffer import PufferPolicy, PufferPolicyConfig
from mettagrid.builder.envs import make_arena
from mettagrid.policy.policy_env_interface import PolicyEnvInterface

logger = logging.getLogger("puffer_policy")


def _is_puffer_state_dict(loaded_obj) -> TypeGuard[Dict[str, torch.Tensor]]:
    if not isinstance(loaded_obj, dict) or not loaded_obj:
        return False

    keys = loaded_obj.keys()
    puffer_keys = ["policy.conv1.weight", "lstm.weight_ih_l0", "policy.actor.0.weight"]
    return all(key in keys for key in puffer_keys)


def _create_metta_agent(device: str | torch.device = "cpu"):
    env_cfg = make_arena(num_agents=60)

    policy_cfg = PufferPolicyConfig()
    policy = PufferPolicy(PolicyEnvInterface.from_mg_cfg(env_cfg), policy_cfg).to(device)
    return policy


def load_pufferlib_checkpoint(checkpoint_data, device: str | torch.device = "cpu"):
    logger.info("Loading checkpoint in PufferLib state_dict format")
    if not isinstance(checkpoint_data, dict):
        raise TypeError("Expected checkpoint_data to be a dict (state_dict format)")
    policy = _create_metta_agent(device)

    policy.load_state_dict(checkpoint_data, strict=False)
    logger.info("Successfully loaded PufferLib checkpoint into Metta policy")

    return policy
