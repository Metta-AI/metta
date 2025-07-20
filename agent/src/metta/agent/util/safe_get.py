from typing import Any, Union

import gymnasium as gym


def safe_get_from_obs_space(
    obs_space: Union[gym.spaces.Space, gym.spaces.Dict],
    obs_key: str,
    property_name: str,
) -> Any:
    """
    Safely extract properties from observation spaces with comprehensive error handling.

    Args:
        obs_space: the observation space to search
        obs_key: The obs_key to access in the observation space
        property_name: The name of the property to extract

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
                    raise ValueError(f"Space for obs_key '{obs_key}' is None")
                if not hasattr(space, property_name):
                    raise ValueError(f"Space for obs_key '{obs_key}' has no {property_name} attribute")

                prop = getattr(space, property_name)

                if prop is None:
                    raise ValueError(f"Space for obs_key '{obs_key}' has {property_name} attribute = None")

                return prop
            else:
                raise ValueError(
                    f"Key '{obs_key}' not found in observation space. Available keys: {list(obs_space.spaces.keys())}"
                )
        elif hasattr(obs_space, property_name):
            prop = getattr(obs_space, property_name)
            if prop is None:
                raise ValueError(f"Observation space {property_name} is None")
            return prop
        else:
            raise ValueError(f"Observation space doesn't have a {property_name} attribute")

    except (TypeError, AttributeError, IndexError) as e:
        raise ValueError(f"Error accessing {property_name} from observation space: {e}") from e
