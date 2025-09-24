"""
PufferLib checkpoint loading and conversion utilities.

This module provides integration between PufferLib checkpoints (state_dict format)
and Metta agents. It detects checkpoint formats, preprocesses state dictionaries,
and loads them into compatible Metta agents.
"""

import logging
from typing import Any, Dict, TypeGuard

import torch

from metta.agent.policies.puffer import PufferPolicyConfig, PufferPolicy

logger = logging.getLogger(__name__)


def _is_puffer_state_dict(loaded_obj: Any) -> TypeGuard[Dict[str, torch.Tensor]]:
    """Return True if the object appears to be a PufferLib state_dict."""
    print("state dict keys", loaded_obj.keys())
    return isinstance(loaded_obj, dict) and bool(loaded_obj) and any(key.startswith("policy.") for key in loaded_obj)


def _preprocess_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Map PufferLib-specific keys to Metta-compatible keys."""
    processed = {}

    key_mappings = {
        # For PufferSimplePolicy, keep the "policy." prefix since it wraps PufferLibPolicy
        # Max vec
        "policy.max_vec": "policy.max_vec",
        # Convolution layers
        "policy.conv1.weight": "policy.conv1.weight",
        "policy.conv1.bias": "policy.conv1.bias",
        "policy.conv2.weight": "policy.conv2.weight",
        "policy.conv2.bias": "policy.conv2.bias",
        # Fully connected layers
        "policy.network.0.weight": "policy.network.0.weight",
        "policy.network.0.bias": "policy.network.0.bias",
        "policy.network.2.weight": "policy.network.2.weight",
        "policy.network.2.bias": "policy.network.2.bias",
        "policy.network.5.weight": "policy.network.5.weight",
        "policy.network.5.bias": "policy.network.5.bias",
        # Self encoder
        "policy.self_encoder.0.weight": "policy.self_encoder.0.weight",
        "policy.self_encoder.0.bias": "policy.self_encoder.0.bias",
        # LSTM mappings - PufferSimplePolicy doesn't have LSTM wrapper, so map directly
        "lstm.weight_ih_l0": "policy.lstm.weight_ih_l0",
        "lstm.weight_hh_l0": "policy.lstm.weight_hh_l0",
        "lstm.bias_ih_l0": "policy.lstm.bias_ih_l0",
        "lstm.bias_hh_l0": "policy.lstm.bias_hh_l0",
        # Alternate cell mappings (duplicates in checkpoint)
        "cell.weight_ih": "policy.lstm.weight_ih_l0",
        "cell.weight_hh": "policy.lstm.weight_hh_l0",
        "cell.bias_ih": "policy.lstm.bias_ih_l0",
        "cell.bias_hh": "policy.lstm.bias_hh_l0",
        # Value head
        "policy.value.weight": "policy.value.weight",
        "policy.value.bias": "policy.value.bias",
        # Actor head (expanded to handle more layers)
        "policy.actor.0.weight": "policy.actor.0.weight",
        "policy.actor.0.bias": "policy.actor.0.bias",
        "policy.actor.1.weight": "policy.actor.1.weight",
        "policy.actor.1.bias": "policy.actor.1.bias",
        "policy.actor.2.weight": "policy.actor.2.weight",
        "policy.actor.2.bias": "policy.actor.2.bias",
        "policy.actor.3.weight": "policy.actor.3.weight",
        "policy.actor.3.bias": "policy.actor.3.bias",
        "policy.actor.4.weight": "policy.actor.4.weight",
        "policy.actor.4.bias": "policy.actor.4.bias",
    }

    for src_key, dst_key in key_mappings.items():
        if src_key in state_dict:
            processed[dst_key] = state_dict[src_key]
        else:
            logger.debug(f"Missing expected key in checkpoint: {src_key}")

    logger.info(f"Preprocessed checkpoint: {len(state_dict)} -> {len(processed)} parameters")
    return processed


def _create_metta_agent(device: str | torch.device = "cpu") -> Any:
    """Instantiate a PufferLib-compatible Metta policy for checkpoint loading."""

    from mettagrid import MettaGridEnv
    from mettagrid.builder.envs import make_arena

    # Minimal environment for initialization
    env_cfg = make_arena(num_agents=60)
    temp_env = MettaGridEnv(env_cfg, render_mode="rgb_array")

    policy_cfg = PufferPolicyConfig()
    policy = PufferPolicy(temp_env, policy_cfg).to(device)
    print("policy keys", policy.state_dict().keys())

    temp_env.close()
    return policy


def _load_state_dict_into_agent(policy: Any, state_dict: Dict[str, torch.Tensor]) -> Any:
    """Load a state_dict into a policy, handling key and shape mismatches."""
    policy_state = policy.state_dict()
    compatible_state = {}
    shape_mismatches = []
    missing_keys = []

    keys_matched = 0
    for key, value in state_dict.items():
        if key in policy_state:
            target_param = policy_state[key]
            if target_param.shape == value.shape:
                compatible_state[key] = value
                keys_matched += 1
            else:
                shape_mismatches.append(f"{key}: checkpoint {value.shape} vs policy {target_param.shape}")
                print(f"Shape mismatch for {key}: checkpoint {value.shape} vs policy {target_param.shape}")
        else:
            missing_keys.append(key)
            print(f"Missing key in policy: {key}")

    if shape_mismatches:
        logger.warning(f"Shape mismatches found for {len(shape_mismatches)} parameters")
        for mismatch in shape_mismatches:
            print(f"SHAPE MISMATCH: {mismatch}")

    if missing_keys:
        print(f"Missing keys in policy: {missing_keys}")

    logger.info(f"Loaded {keys_matched}/{len(state_dict)} compatible parameters")

    if not compatible_state:
        raise RuntimeError("No compatible parameters found in checkpoint")

    try:
        policy.load_state_dict(compatible_state, strict=False)
        logger.info("Successfully loaded checkpoint into Metta policy")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise

    return policy


def load_pufferlib_checkpoint(checkpoint_data: Any, device: str | torch.device = "cpu") -> Any:
    """Load a PufferLib checkpoint into a Metta policy."""
    logger.info("Loading checkpoint in PufferLib state_dict format")
    if not isinstance(checkpoint_data, dict):
        raise TypeError("Expected checkpoint_data to be a dict (state_dict format)")

    logger.debug(f"Checkpoint sample keys: {list(checkpoint_data.keys())[:10]}")
    policy = _create_metta_agent(device)
    processed_state = _preprocess_state_dict(checkpoint_data)
    return _load_state_dict_into_agent(policy, processed_state)
