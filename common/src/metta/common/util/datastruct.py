from collections import Counter
from typing import Hashable, Iterable, List

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


def duplicates(iterable: Iterable[Hashable]) -> List[Hashable]:
    """Returns a list of items that appear more than once in the iterable."""
    counts = Counter(iterable)
    return [item for item, count in counts.items() if count > 1]
