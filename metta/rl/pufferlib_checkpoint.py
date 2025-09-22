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
    for key, value in state_dict.items():
        # Remove common PufferLib-specific prefixes
        if key.startswith("policy."):
            key = key[len("policy.") :]
        elif key.startswith("learner."):
            key = key[len("learner.") :]
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
    # Handle PufferLib Policy -> Metta mappings
    key_mappings = {
        # CNN layers mapping
        "conv1.": "cnn_encoder.cnn1.",
        "conv2.": "cnn_encoder.cnn2.",
        "network.0.": "cnn_encoder.cnn1.",  # First conv layer
        "network.2.": "cnn_encoder.cnn2.",  # Second conv layer
        "network.4.": "cnn_encoder.fc1.",  # First linear layer
        # Self encoder mapping
        "self_encoder.": "obs_shim.",
        # Actor/critic mappings
        "actor.": "action_probs.",
        "value.": "critic.",
        # LSTM mappings (if present)
        "lstm.": "core.lstm.",
        "hidden_": "core.hidden_",
        "cell_": "core.cell_",
    }

    for puffer_key, metta_key in key_mappings.items():
        if puffer_key in key:
            key = key.replace(puffer_key, metta_key)
            break

    return key


def _create_metta_agent(device: str | torch.device = "cpu") -> Any:
    """Create a Policy with default configuration suitable for PufferLib checkpoint loading."""
    from metta.agent.policies.fast import FastConfig, FastPolicy
    from mettagrid import MettaGridEnv
    from mettagrid.builder.envs import make_arena

    # Create minimal environment for agent initialization
    env_cfg = make_arena(num_agents=60)
    temp_env = MettaGridEnv(env_cfg, render_mode="rgb_array")

    # Create a basic policy configuration
    policy_cfg = FastConfig()

    # Create the policy directly
    policy = FastPolicy(temp_env, policy_cfg)
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
    for key, value in state_dict.items():
        if key in policy_state:
            policy_param = policy_state[key]
            if policy_param.shape == value.shape:
                compatible_state[key] = value
                keys_matched += 1
            else:
                shape_mismatches.append(f"{key}: PufferLib {value.shape} vs Metta {policy_param.shape}")
                logger.debug(f"Shape mismatch for {key}: PufferLib {value.shape} vs Metta {policy_param.shape}")
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
