"""
Backwards compatibility adapter for observation format changes.

This module contains adapters to handle breaking changes in observation formats
between different versions of the codebase, allowing older models to work with
newer observation formats.
"""

import logging
import os
from typing import Union

import gymnasium as gym
import torch
from omegaconf import DictConfig, ListConfig

logger = logging.getLogger(__name__)


def adapter_for_commit_hash_1af886ae0093c528b0e3e18537db730547115086(x: torch.Tensor) -> torch.Tensor:
    """
    Backwards compatibility adapter for observation format changes in commit 1af886ae.

    This commit consolidated inventory features from separate agent:inv:* and inv:* slots
    into shared inv:* slots, reducing the feature dimension from 34 to 26.

    This adapter converts the new 26-feature format back to the old 34-feature format
    by duplicating inventory features into both agent-specific and shared slots.

    Args:
        x: Input tensor with shape [..., H, W, 26] (new format)

    Returns:
        Tensor with shape [..., H, W, 34] (old format)
    """
    if x.shape[-1] != 26:
        # Not the format we need to adapt
        return x

    # Map from new consolidated format (26 features) to old format (34 features)
    # New format inventory indices
    inv_ore_red_idx = 6  # "inv:ore.red"
    inv_ore_blue_idx = 7  # "inv:ore.blue"
    inv_ore_green_idx = 8  # "inv:ore.green"
    inv_battery_idx = 9  # "inv:battery"
    inv_heart_idx = 10  # "inv:heart"
    inv_armor_idx = 11  # "inv:armor"
    inv_laser_idx = 12  # "inv:laser"
    inv_blueprint_idx = 13  # "inv:blueprint"

    # Create the expanded tensor for old format
    batch_dims = x.shape[:-1]
    expanded_shape = batch_dims + (34,)
    expanded_x = torch.zeros(expanded_shape, dtype=x.dtype, device=x.device)

    # Copy features to old format positions:

    # Features 0-5: agent, agent:group, hp, agent:frozen, agent:orientation, agent:color
    expanded_x[..., 0:6] = x[..., 0:6]

    # Features 6-13: Copy inventory to agent:inv:* slots (old format)
    expanded_x[..., 6] = x[..., inv_ore_red_idx]  # agent:inv:ore.red
    expanded_x[..., 7] = x[..., inv_ore_blue_idx]  # agent:inv:ore.blue
    expanded_x[..., 8] = x[..., inv_ore_green_idx]  # agent:inv:ore.green
    expanded_x[..., 9] = x[..., inv_battery_idx]  # agent:inv:battery
    expanded_x[..., 10] = x[..., inv_heart_idx]  # agent:inv:heart
    expanded_x[..., 11] = x[..., inv_armor_idx]  # agent:inv:armor
    expanded_x[..., 12] = x[..., inv_laser_idx]  # agent:inv:laser
    expanded_x[..., 13] = x[..., inv_blueprint_idx]  # agent:inv:blueprint

    # Features 14-18: wall, swappable, mine, color, converting
    expanded_x[..., 14:19] = x[..., 14:19]

    # Features 19-26: Copy inventory to shared inv:* slots (old format)
    expanded_x[..., 19:27] = x[..., 6:14]  # inv:ore.red through inv:blueprint

    # Features 27-33: generator, altar, armory, lasery, lab, factory, temple
    expanded_x[..., 27:34] = x[..., 19:26]

    return expanded_x


def apply_backwards_compatibility_adapters(x: torch.Tensor) -> torch.Tensor:
    """
    Apply all backwards compatibility adapters in sequence.

    This function serves as the main entry point for applying all necessary
    backwards compatibility transformations to observation tensors.

    Args:
        x: Input observation tensor

    Returns:
        Transformed tensor with backwards compatibility applied
    """
    # Apply adapters in chronological order (oldest first)
    x = adapter_for_commit_hash_1af886ae0093c528b0e3e18537db730547115086(x)

    # Future adapters can be added here:
    # x = adapter_for_commit_hash_xyz(x)

    return x


def should_update_observation_space(cfg: Union[DictConfig, ListConfig]) -> bool:
    """
    Detect if we should use old 34-feature format for backwards compatibility.

    This checks if we're loading an old model that was trained with 34 features.

    Args:
        cfg: Configuration object that might contain model loading paths

    Returns:
        True if old format should be used, False for new format
    """
    # Check common config fields for model loading paths
    load_paths = []

    # Direct model loading paths
    if hasattr(cfg, "load_model_path") and cfg.load_model_path:
        load_paths.append(cfg.load_model_path)

    if hasattr(cfg, "model_path") and cfg.model_path:
        load_paths.append(cfg.model_path)

    if hasattr(cfg, "checkpoint_path") and cfg.checkpoint_path:
        load_paths.append(cfg.checkpoint_path)

    if hasattr(cfg, "resume_from") and cfg.resume_from:
        load_paths.append(cfg.resume_from)

    # Check trainer config for model paths
    if hasattr(cfg, "trainer") and cfg.trainer:
        trainer_cfg = cfg.trainer
        if hasattr(trainer_cfg, "load_model_path") and trainer_cfg.load_model_path:
            load_paths.append(trainer_cfg.load_model_path)

    # Check any of the found paths
    for path in load_paths:
        if path and _is_old_model_file(path):
            return True

    return False


def update_observation_space(obs_space: gym.spaces.Space) -> gym.spaces.Space:
    """
    Update observation space from 26-feature to 34-feature format for old models.

    Args:
        obs_space: Original gym observation space (26 features)

    Returns:
        Updated observation space (34 features) for old model compatibility
    """
    if hasattr(obs_space, "shape") and len(obs_space.shape) == 3:
        # Check if this is the 26-feature format that needs updating
        if obs_space.shape[2] == 26:
            # Update to 34-feature format
            updated_shape = (obs_space.shape[0], obs_space.shape[1], 34)
            return gym.spaces.Box(
                low=obs_space.low.flat[0],  # Use the same bounds
                high=obs_space.high.flat[0],
                shape=updated_shape,
                dtype=obs_space.dtype,
            )

    # No changes needed
    return obs_space


def _is_old_model_file(model_path: str) -> bool:
    """
    Check if a model file contains old 34-feature format weights.

    Loads just the state dict to inspect for 34-feature signatures.
    """
    try:
        if not os.path.exists(model_path):
            logger.warning(f"Model path does not exist: {model_path}")
            return False

        # Load state dict to check format
        state_dict = torch.load(model_path, map_location="cpu")

        # Look for telltale signs of 34-feature format
        for key, tensor in state_dict.items():
            # Check for obs_norm with 34 features
            if "obs_norm" in key and tensor.numel() == 34:
                logger.info(f"Detected old 34-feature format in {model_path} (found {key} with 34 features)")
                return True

            # Check for linear layers with 34*11*11 = 4114 inputs
            if "weight" in key and len(tensor.shape) == 2:
                if tensor.shape[1] == 34 * 11 * 11:
                    logger.info(f"Detected old 34-feature format in {model_path} (found {key} with {tensor.shape})")
                    return True

        logger.info(f"Model {model_path} appears to use new 26-feature format")
        return False

    except Exception as e:
        logger.warning(f"Could not check model format for {model_path}: {e}")
        return False  # Default to new format if we can't check
