from typing import Any, Dict, List, TypeVar, Union

from omegaconf import DictConfig, ListConfig

# Define a TypeVar for scalar values, properly bound in the function signature
T = TypeVar("T")

# Define types for the structures we're handling
ScalarType = Union[str, int, float, bool, None]
ConfigType = Union[Dict[str, Any], List[Any], DictConfig, ListConfig, ScalarType]


def flatten_config(obj: ConfigType, parent_key: str = "", sep: str = ".") -> Dict[str, ScalarType]:
    """
    Recursively flatten a nested structure of DictConfig, ListConfig, dict, and list
    using dot notation, including indices for list items.

    Example:

      Input:
      {
          "foo": {
              "bar": [
                  {"a": 1},
                  {"b": 2}
              ]
          }
      }

      Output:
      {
          "foo.bar.0.a": 1,
          "foo.bar.1.b": 2
      }

    Args:
        obj: The structure to flatten (DictConfig, ListConfig, dict, list, or a scalar).
        parent_key: The base key (used for recursion).
        sep: The separator to use between levels in the flattened key.

    Returns:
        A flattened dictionary with keys in dot notation.
    """
    items: Dict[str, ScalarType] = {}

    # If obj is a dictionary-like (DictConfig or dict)
    if isinstance(obj, (DictConfig, dict)):
        for k, v in obj.items():
            # Compose a new key, ensuring k is a string
            key_str = str(k)
            new_key = f"{parent_key}{sep}{key_str}" if parent_key else key_str
            # Recurse
            if isinstance(v, (DictConfig, dict, ListConfig, list)):
                items.update(flatten_config(v, new_key, sep=sep))
            else:
                items[new_key] = v

    # If obj is a list-like (ListConfig or list)
    elif isinstance(obj, (ListConfig, list)):
        for i, v in enumerate(obj):
            # Use the list index as part of the key
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            if isinstance(v, (DictConfig, dict, ListConfig, list)):
                items.update(flatten_config(v, new_key, sep=sep))
            else:
                items[new_key] = v

    # If obj is a scalar (int, str, float, etc.), just assign it
    else:
        # This handles the corner case where flatten_config is called
        # on a direct scalar, without a parent key.
        items[parent_key] = obj

    return items
