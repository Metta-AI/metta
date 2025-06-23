"""Utility functions that were previously in pufferlib.utils"""

from typing import Any, Iterator, Tuple


def unroll_nested_dict(d: dict, parent_key: str = "", sep: str = "/") -> Iterator[Tuple[str, Any]]:
    """
    Recursively flatten a nested dictionary, yielding (key, value) pairs.

    This function takes a nested dictionary and yields tuples of (flattened_key, value)
    where the flattened key uses the specified separator to join nested keys.

    Args:
        d: The dictionary to flatten
        parent_key: The prefix key (used for recursion)
        sep: The separator to use between nested keys

    Yields:
        Tuples of (flattened_key, value) for each leaf value in the nested dict

    Example:
        >>> d = {'a': {'b': 1, 'c': 2}, 'd': 3}
        >>> list(unroll_nested_dict(d))
        [('a/b', 1), ('a/c', 2), ('d', 3)]
    """
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            yield from unroll_nested_dict(v, new_key, sep)
        else:
            yield (new_key, v)


def profile(func):
    """
    Decorator for profiling functions.
    Currently just returns the function as-is, but can be extended
    to add actual profiling functionality.
    """
    return func
