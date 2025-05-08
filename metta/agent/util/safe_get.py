from typing import Any, Optional, Union

import gymnasium as gym


def safe_get_obs_property(
    obs_space: Union[gym.spaces.Space, gym.spaces.Dict],
    obs_key: str,
    property_index: Optional[int] = None,
    property_name: str = "shape",
) -> Any:
    """
    Safely extract properties from observation spaces with comprehensive error handling.

    Args:
        obs_space: The observation space to extract properties from
        obs_key: The key to access in the observation space
        property_index: Optional index to access in the property (e.g., 1 for shape[1:], 2 for shape[2])
        property_name: The name of the property to extract (default: "shape")

    Returns:
        The extracted property value

    Raises:
        ValueError: If the property cannot be safely extracted
    """
    try:
        if isinstance(obs_space, gym.spaces.Dict):
            if obs_key in obs_space.spaces:
                space = obs_space.spaces[obs_key]
                if space is None:
                    raise ValueError(f"Space for key '{obs_key}' is None")
                if not hasattr(space, property_name):
                    raise ValueError(f"Space for key '{obs_key}' has no {property_name} attribute")

                prop = getattr(space, property_name)
                if prop is None:
                    raise ValueError(f"{property_name.capitalize()} for space '{obs_key}' is None")

                if property_index is not None:
                    if len(prop) <= property_index:
                        raise ValueError(
                            f"{property_name.capitalize()} for space '{obs_key}' "
                            f"does not have an index {property_index}"
                        )
                    if property_index == 1:
                        return prop[1:]  # Special case for obs_input_shape
                    return prop[property_index]
                return prop
            else:
                raise ValueError(
                    f"Key '{obs_key}' not found in observation space. Available keys: {list(obs_space.spaces.keys())}"
                )
        elif hasattr(obs_space, property_name):
            prop = getattr(obs_space, property_name)
            if prop is None:
                raise ValueError(f"Observation space {property_name} is None")

            if property_index is not None:
                if len(prop) <= property_index:
                    raise ValueError(f"Observation space {property_name} does not have an index {property_index}")
                if property_index == 1:
                    return prop[1:]  # Special case for obs_input_shape
                return prop[property_index]
            return prop
        else:
            raise ValueError(f"Observation space doesn't have a {property_name} attribute")
    except (TypeError, AttributeError, IndexError) as e:
        raise ValueError(f"Error accessing {property_name} from observation space: {e}") from e
