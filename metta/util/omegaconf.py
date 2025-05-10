from typing import Any, Dict, TypeVar

from omegaconf import OmegaConf, SCMode

# Define a type variable for dictionary keys
KT = TypeVar("KT")


def convert_to_dict(config, resolve: bool = True) -> Dict[str, Any]:
    """
    Convert OmegaConf config to a standard Python dictionary with string keys.
    Raises ValueError if the config structure resolves to a list or other non-dict type.

    Args:
        config: The OmegaConf config to convert
        resolve: Whether to resolve interpolations

    Returns:
        Dict[str, Any]: A standard Python dictionary with string keys

    Raises:
        ValueError: If the config doesn't resolve to a dictionary
    """
    result = OmegaConf.to_container(config, resolve=resolve, enum_to_str=True, structured_config_mode=SCMode.DICT)

    if not isinstance(result, dict):
        raise ValueError(f"Expected dictionary configuration, got {type(result).__name__}")

    # Convert all keys to strings to ensure consistent typing
    string_keyed_dict: Dict[str, Any] = {}
    for key, value in result.items():
        string_keyed_dict[str(key)] = value

    return string_keyed_dict
