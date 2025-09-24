"""
PufferLib checkpoint loading and conversion utilities.

This module provides integration between PufferLib checkpoints (state_dict format)
and Metta agents. It detects checkpoint formats, preprocesses state dictionaries,
and loads them into compatible Metta agents.
"""

import logging
from typing import Any, Dict, TypeGuard

import torch

from metta.agent.policies.puffer import PufferPolicy, PufferPolicyConfig

logger = logging.getLogger(__name__)


def _is_puffer_state_dict(loaded_obj: Any) -> TypeGuard[Dict[str, torch.Tensor]]:
    """Return True if the object appears to be a PufferLib state_dict."""
    return isinstance(loaded_obj, dict) and bool(loaded_obj) and any(key.startswith("policy.") for key in loaded_obj)


def _preprocess_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Preprocess PufferLib state dict - handle LSTM key mapping only."""
    processed = {}

    # Most keys can be passed through directly now that policy has 'policy.' prefix
    # Only need to handle LSTM keys which have different structure
    for key, value in state_dict.items():
        if key.startswith("lstm."):
            # Map LSTM keys: lstm.* -> lstm.net.*
            new_key = key.replace("lstm.", "lstm.net.")
            processed[new_key] = value
        elif key.startswith("cell."):
            # Skip cell keys - they're duplicates of lstm keys
            # cell.weight_ih is same as lstm.weight_ih_l0, etc.
            continue
        else:
            # Pass through all other keys unchanged (including policy.*)
            processed[key] = value

    logger.info(f"Preprocessed checkpoint: {len(state_dict)} -> {len(processed)} parameters")
    print("Processed keys:", sorted(processed.keys()))
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
    print("Policy keys:", sorted(policy.state_dict().keys()))

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
                print(f"✓ Loaded: {key} {value.shape}")
            else:
                shape_mismatches.append(f"{key}: checkpoint {value.shape} vs policy {target_param.shape}")
                print(f"✗ Shape mismatch: {key} checkpoint {value.shape} vs policy {target_param.shape}")
        else:
            missing_keys.append(key)
            print(f"✗ Missing in policy: {key}")

    if shape_mismatches:
        logger.warning(f"Shape mismatches found for {len(shape_mismatches)} parameters")

    if missing_keys:
        print(f"Missing keys in policy: {missing_keys}")

    # Show which policy keys weren't loaded
    policy_keys_not_loaded = set(policy_state.keys()) - set(compatible_state.keys())
    if policy_keys_not_loaded:
        print(f"Policy parameters not loaded: {sorted(policy_keys_not_loaded)}")

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
