import numpy as np


def revert_observations(observations: np.ndarray) -> np.ndarray:
    """
    Revert observations from new 26-channel format to old 34-channel format.

    This function operates directly on numpy arrays, converting the consolidated
    26-feature format back to the old 34-feature format by duplicating inventory
    features into both agent-specific and shared slots.

    Args:
        observations: Numpy array with shape [..., H, W, 26] (new format)

    Returns:
        Numpy array with shape [..., H, W, 34] (old format)
    """
    if observations.shape[-1] != 26:
        # Not the format we need to adapt
        return observations

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

    # Create the expanded array for old format
    batch_dims = observations.shape[:-1]
    expanded_shape = batch_dims + (34,)
    expanded_obs = np.zeros(expanded_shape, dtype=observations.dtype)

    # Copy features to old format positions:

    # Features 0-5: agent, agent:group, hp, agent:frozen, agent:orientation, agent:color
    expanded_obs[..., 0:6] = observations[..., 0:6]

    # Features 6-13: Copy inventory to agent:inv:* slots (old format)
    expanded_obs[..., 6] = observations[..., inv_ore_red_idx]  # agent:inv:ore.red
    expanded_obs[..., 7] = observations[..., inv_ore_blue_idx]  # agent:inv:ore.blue
    expanded_obs[..., 8] = observations[..., inv_ore_green_idx]  # agent:inv:ore.green
    expanded_obs[..., 9] = observations[..., inv_battery_idx]  # agent:inv:battery
    expanded_obs[..., 10] = observations[..., inv_heart_idx]  # agent:inv:heart
    expanded_obs[..., 11] = observations[..., inv_armor_idx]  # agent:inv:armor
    expanded_obs[..., 12] = observations[..., inv_laser_idx]  # agent:inv:laser
    expanded_obs[..., 13] = observations[..., inv_blueprint_idx]  # agent:inv:blueprint

    # Features 14-18: wall, swappable, mine, color, converting
    expanded_obs[..., 14:19] = observations[..., 14:19]

    # Features 19-26: Copy inventory to shared inv:* slots (old format)
    expanded_obs[..., 19:27] = observations[..., 6:14]  # inv:ore.red through inv:blueprint

    # Features 27-33: generator, altar, armory, lasery, lab, factory, temple
    expanded_obs[..., 27:34] = observations[..., 19:26]

    return expanded_obs
