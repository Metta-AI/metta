"""
Type stubs for metta.mettagrid.util.hash
"""

from typing import List

DEFAULT_SEED: int

def hash_string(str: str, seed: int = 0) -> int:
    """
    Hash a string using rapidhash algorithm.

    Args:
        str: The string to hash
        seed: Optional seed value (default: 0)

    Returns:
        64-bit hash value
    """
    ...

def hash_mettagrid_map(map: List[List[str]]) -> int:
    """
    Calculate a deterministic hash of a MettaGrid map.

    Args:
        map: A list of lists of strings representing the grid

    Returns:
        64-bit hash value representing the map configuration
    """
    ...
