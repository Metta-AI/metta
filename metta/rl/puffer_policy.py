"""
PufferLib checkpoint loading utilities.
"""

import logging
import typing

import torch

import metta.agent.policies.puffer
import mettagrid.builder.envs
import mettagrid.policy.policy_env_interface

logger = logging.getLogger("puffer_policy")


def _is_puffer_state_dict(loaded_obj) -> typing.TypeGuard[typing.Dict[str, torch.Tensor]]:
    if not isinstance(loaded_obj, dict) or not loaded_obj:
        return False

    keys = loaded_obj.keys()
    puffer_keys = ["policy.conv1.weight", "lstm.weight_ih_l0", "policy.actor.0.weight"]
    return all(key in keys for key in puffer_keys)


def _create_metta_agent(device: str | torch.device = "cpu"):
    env_cfg = mettagrid.builder.envs.make_arena(num_agents=60)

    policy_cfg = metta.agent.policies.puffer.PufferPolicyConfig()
    policy = metta.agent.policies.puffer.PufferPolicy(
        mettagrid.policy.policy_env_interface.PolicyEnvInterface.from_mg_cfg(env_cfg), policy_cfg
    ).to(device)
    return policy


def load_pufferlib_checkpoint(checkpoint_data, device: str | torch.device = "cpu"):
    logger.info("Loading checkpoint in PufferLib state_dict format")
    if not isinstance(checkpoint_data, dict):
        raise TypeError("Expected checkpoint_data to be a dict (state_dict format)")
    policy = _create_metta_agent(device)

    policy.load_state_dict(checkpoint_data, strict=False)
    logger.info("Successfully loaded PufferLib checkpoint into Metta policy")

    return policy
