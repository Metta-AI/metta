"""
Backwards compatibility adapters for observation format changes.

This module contains adapters to handle breaking changes in observation formats
between different versions of the codebase, allowing older models to work with
newer observation formats.
"""

import torch


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


def update_agent_attributes_for_backwards_compatibility(agent_attributes: dict) -> dict:
    """
    Update agent attributes to match the backwards compatibility transformations.

    This function updates obs_shape and related attributes to match what the
    adapters will produce, ensuring consistency throughout the agent.

    Args:
        agent_attributes: Dictionary containing agent configuration

    Returns:
        Updated agent_attributes dictionary
    """
    if "obs_shape" in agent_attributes:
        obs_shape = agent_attributes["obs_shape"]
        if isinstance(obs_shape, (list, tuple)) and len(obs_shape) == 3:
            # If we have the new 26-feature format, update to expect 34 features
            if obs_shape[2] == 26:
                updated_obs_shape = list(obs_shape)
                updated_obs_shape[2] = 34
                agent_attributes["obs_shape"] = updated_obs_shape
                agent_attributes["num_objects"] = 34

    return agent_attributes
