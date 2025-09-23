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

    # Check for obvious Metta agent attributes first
    if hasattr(loaded_obj, "policy") or "policy" in loaded_obj:
        return False

    # Check if it looks like a TorchRL agent
    if any(hasattr(loaded_obj, attr) for attr in ["obs_spec", "action_spec", "reward_spec"]):
        return False

    # Check if all items are parameter name -> tensor mappings
    sample_items = list(loaded_obj.items())[:10]  # Check more items for better confidence
    tensor_count = 0
    for key, value in sample_items:
        if not isinstance(key, str):
            return False
        if torch.is_tensor(value):
            tensor_count += 1
        elif not isinstance(value, (int, float, bool, str)):  # Allow simple types
            return False

    # Require at least 80% of sampled items to be tensors for state_dict detection
    return tensor_count >= len(sample_items) * 0.8


def _preprocess_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove PufferLib-specific prefixes from state dict keys and handle key mappings."""
    processed = {}
    print(f"state_dict keys: {state_dict.keys()}")
    print(f"state_dict keys: {len(state_dict)}")
    for key, value in state_dict.items():
        # Remove common PufferLib-specific prefixes
        if key.startswith("policy."):
            key = key[len("policy.") :]
        elif key.startswith("recurrent.policy."):
            key = key[len("recurrent.policy.") :]
        elif key.startswith("recurrent."):
            key = key[len("recurrent.") :]

        # Handle specific PufferLib to Metta key mappings
        # Map PufferLib Policy structure to Metta structure
        key = _map_pufferlib_key_to_metta(key)

        processed[key] = value

    logger.info(f"Preprocessed state_dict: {len(state_dict)} -> {len(processed)} parameters")
    return processed


def _map_pufferlib_key_to_metta(key: str) -> str:
    """Map PufferLib checkpoint keys to Metta agent keys."""
    # Handle PufferLib Policy -> Metta mappings based on actual checkpoint analysis
    # Based on the actual keys from policy vs state_dict, we need to map:
    # - PufferLib checkpoint keys (after policy. prefix removal) to Metta policy keys
    # - The PufferLibCompatiblePolicy uses direct PyTorch modules, not Metta components
    key_mappings = {

        #Conv layers
        "policy.conv1.weight": "conv1.weight",
        "policy.conv1.bias": "conv1.bias",
        "policy.conv2.weight": "conv2.weight",
        "policy.conv2.bias": "conv2.bias",

        # network param
        "policy.network.0.weight": "network.0.weight",
        "policy.network.0.bias": "network.0.bias",
        "policy.network.2.weight": "network.2.weight",
        "policy.network.2.bias": "network.2.bias",
        "policy.network.5.weight": "network.5.weight",
        "policy.network.5.bias": "network.5.bias",

        # self encoder params
     
        # LSTM mappings - PufferLib checkpoint has different structure
        "lstm.weight_ih_l0": "lstm.net.weight_ih_l0",
        "lstm.weight_hh_l0": "lstm.net.weight_hh_l0", 
        "lstm.bias_ih_l0": "lstm.net.bias_ih_l0",
        "lstm.bias_hh_l0": "lstm.net.bias_hh_l0",
        
        # Cell mappings (duplicates in checkpoint) - map to same LSTM targets
        "cell.weight_ih": "lstm.net.weight_ih_l0",
        "cell.weight_hh": "lstm.net.weight_hh_l0",
        "cell.bias_ih": "lstm.net.bias_ih_l0", 
        "cell.bias_hh": "lstm.net.bias_hh_l0",

        # value
        "policy.value.weight": "value.weight",
        "policy.value.bias": "value.bias",
        
        # actor
        "policy.actor.0.weight": "actor.0.weight",
        "policy.actor.0.bias": "actor.0.bias",
        "policy.actor.1.weight": "actor.1.weight",
        "policy.actor.1.bias": "actor.1.bias",

    }


    for puffer_key, metta_key in key_mappings.items():
        if puffer_key in key:
            key = key.replace(puffer_key, metta_key)
            break

    return key


def _create_metta_agent(device: str | torch.device = "cpu") -> Any:
    """Create a Policy with PufferLib-compatible configuration for checkpoint loading."""
    from metta.agent.policies.pufferlib_compatible import PufferLibCompatibleConfig, PufferLibCompatiblePolicy
    from mettagrid import MettaGridEnv
    from mettagrid.builder.envs import make_arena

    # Create minimal environment for agent initialization
    env_cfg = make_arena(num_agents=60)
    temp_env = MettaGridEnv(env_cfg, render_mode="rgb_array")

    # Create a PufferLib-compatible policy configuration
    policy_cfg = PufferLibCompatibleConfig()

    # Create the policy directly
    policy = PufferLibCompatiblePolicy(temp_env, policy_cfg)
    print(f"policy: {policy.state_dict().keys()}")
    print(f"policy: {len(policy.state_dict())}")
    temp_env.close()

    # Move policy to the specified device
    if device != "cpu":
        policy = policy.to(device)

    return policy


def _load_state_dict_into_agent(policy: Any, state_dict: Dict[str, torch.Tensor]) -> Any:
    """Load state dict into policy, handling compatibility issues."""
    policy_state = policy.state_dict()
    compatible_state = {}
    shape_mismatches = []

    keys_matched = 0
    matched_keys = []
    not_matched_keys = []
    for key, value in state_dict.items():
        if key in policy_state:
            policy_param = policy_state[key]
            if policy_param.shape == value.shape:
                compatible_state[key] = value
                keys_matched += 1
                matched_keys.append({"pufferlib": key, "metta": key})
            else:
                shape_mismatches.append(f"{key}: PufferLib {value.shape} vs Metta {policy_param.shape}")
                logger.debug(f"Shape mismatch for {key}: PufferLib {value.shape} vs Metta {policy_param.shape}")
                not_matched_keys.append({"pufferlib": key, "metta": key})
        else:
            logger.debug(f"Skipping incompatible parameter: {key}")

    if shape_mismatches:
        truncated = shape_mismatches[:5] + (["..."] if len(shape_mismatches) > 5 else [])
        logger.warning(f"Found {len(shape_mismatches)} shape mismatches: {truncated}")

    logger.info(f"Loaded {keys_matched}/{len(state_dict)} compatible parameters")

    if compatible_state:
        try:
            policy.load_state_dict(compatible_state, strict=False)
            logger.info("Successfully loaded PufferLib checkpoint into Metta policy")
        except Exception as e:
            logger.error(f"Failed to load state dict: {e}")
            logger.warning("Proceeding with random initialization")
    else:
        logger.warning("No compatible parameters found - proceeding with random initialization")

    return policy


class PufferLibCheckpoint:
    """Simple checkpoint loader supporting both Metta and PufferLib formats."""

    def load_checkpoint(self, checkpoint_data: Any, device: str | torch.device = "cpu") -> Any:
        """Load checkpoint data and return an agent, auto-detecting format."""

        # If it's a PufferLib state_dict, create policy and load weights
        if _is_state_dict(checkpoint_data):
            logger.info("Loading PufferLib checkpoint format (state_dict)")
            logger.debug(f"Checkpoint keys sample: {list(checkpoint_data.keys())[:10]}")
            policy = _create_metta_agent(device)
            processed_state_dict = _preprocess_state_dict(checkpoint_data)
            return _load_state_dict_into_agent(policy, processed_state_dict)

        # Otherwise assume it's a Metta agent and return it directly
        logger.info("Loading native Metta checkpoint format")
        logger.debug(f"Checkpoint type: {type(checkpoint_data)}")
        return checkpoint_data

    def is_pufferlib_format(self, checkpoint_data: Any) -> bool:
        """Check if checkpoint data is in PufferLib format."""
        return _is_state_dict(checkpoint_data)

    def debug_checkpoint_info(self, checkpoint_data: Any) -> Dict[str, Any]:
        """Debug function to analyze checkpoint structure."""
        info = {
            "type": str(type(checkpoint_data)),
            "is_dict": isinstance(checkpoint_data, dict),
            "is_state_dict": _is_state_dict(checkpoint_data) if isinstance(checkpoint_data, dict) else False,
        }

        if isinstance(checkpoint_data, dict):
            info["keys"] = list(checkpoint_data.keys())[:20]  # Show first 20 keys
            info["num_keys"] = len(checkpoint_data)

            # Analyze key patterns
            key_patterns = {}
            for key in checkpoint_data.keys():
                if "." in key:
                    prefix = key.split(".")[0]
                    key_patterns[prefix] = key_patterns.get(prefix, 0) + 1

            info["key_patterns"] = key_patterns
            info["tensor_keys"] = [k for k, v in checkpoint_data.items() if torch.is_tensor(v)][:10]

        return info
