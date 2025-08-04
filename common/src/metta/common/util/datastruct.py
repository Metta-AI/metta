from collections import Counter
from typing import Any, Dict, Hashable, Iterable, List

from omegaconf import DictConfig, ListConfig


def flatten_config(obj, parent_key="", sep="."):
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
        parent_key (str): The base key (used for recursion).
        sep (str): The separator to use between levels in the flattened key.

    Returns:
        dict: A flattened dictionary with keys in dot notation.
    """
    items = {}

    # If obj is a dictionary-like (DictConfig or dict)
    if isinstance(obj, (DictConfig, dict)):
        for k, v in obj.items():
            # Compose a new key
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
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


def convert_dict_to_cli_args(suggestion: Dict[str, Any], prefix: str = "") -> list[str]:
    """Convert a nested dictionary to Hydra command-line arguments."""
    args = []

    for key, value in suggestion.items():
        # Build the full key path
        full_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            # Recursively handle nested dictionaries
            args.extend(convert_dict_to_cli_args(value, full_key))
        else:
            # Convert the value to a string appropriate for command line
            if isinstance(value, bool):
                str_value = "true" if value else "false"
            elif isinstance(value, (int, float)):
                # Use scientific notation for very small/large numbers
                if isinstance(value, float) and (abs(value) < 1e-6 or abs(value) > 1e6):
                    str_value = f"{value:.6e}"
                else:
                    str_value = str(value)
            elif value is None:
                str_value = "null"
            else:
                # Quote strings if they contain spaces or special characters
                str_value = str(value)
                if " " in str_value or "=" in str_value:
                    str_value = f"'{str_value}'"

            # Use ++ prefix to force add-or-override keys regardless of struct mode
            # This works for both existing keys and new keys
            args.append(f"++{full_key}={str_value}")

    return args


def duplicates(iterable: Iterable[Hashable]) -> List[Hashable]:
    """Returns a list of items that appear more than once in the iterable."""
    counts = Counter(iterable)
    return [item for item, count in counts.items() if count > 1]
