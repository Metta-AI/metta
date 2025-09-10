"""Map compression utilities for converting string grids to byte grids."""

from typing import List, Optional, Set, Tuple

import numpy as np


class MapCompressor:
    """Handles compression of string grids to byte grids with validation."""

    def __init__(self, valid_objects: Optional[Set[str]] = None):
        """
        Args:
            valid_objects: Set of valid object names for validation.
                          If None, no validation is performed.
        """
        self.valid_objects = valid_objects

    def compress(self, string_grid: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Convert string grid to byte grid + object key.

        Args:
            string_grid: 2D numpy array of strings (e.g., "wall", "agent.team_1")

        Returns:
            byte_grid: 2D numpy array of uint8 indices
            object_key: List mapping indices to object names

        Raises:
            ValueError: If grid contains unknown objects (when valid_objects is set)
        """
        # Get unique objects in the grid
        unique_objects = np.unique(string_grid)

        # Validate if validator is present
        if self.valid_objects is not None:
            unknown_objects = set(unique_objects) - self.valid_objects - {"empty", ".", " "}
            if unknown_objects:
                raise ValueError(
                    f"Map contains unknown objects: {unknown_objects}. Valid objects: {sorted(self.valid_objects)}"
                )

        # Create object key (sorted for consistency)
        object_key = sorted(unique_objects.tolist())

        # Create mapping dictionary
        str_to_idx = {obj: idx for idx, obj in enumerate(object_key)}

        # Convert to byte grid using vectorized operation
        byte_grid = np.zeros(string_grid.shape, dtype=np.uint8)
        for obj_str, idx in str_to_idx.items():
            mask = string_grid == obj_str
            byte_grid[mask] = idx

        return byte_grid, object_key

    def decompress(self, byte_grid: np.ndarray, object_key: List[str]) -> np.ndarray:
        """
        Convert byte grid + object key back to string grid.

        Args:
            byte_grid: 2D numpy array of uint8 indices
            object_key: List mapping indices to object names

        Returns:
            string_grid: 2D numpy array of strings
        """
        # Handle empty grid
        if byte_grid.size == 0:
            return np.empty(byte_grid.shape, dtype="<U20")

        # Validate indices
        max_idx = byte_grid.max()
        if max_idx >= len(object_key):
            raise ValueError(f"Invalid byte grid: index {max_idx} >= key length {len(object_key)}")

        # Convert using vectorized operation
        string_grid = np.empty(byte_grid.shape, dtype="<U20")
        for idx, obj_str in enumerate(object_key):
            mask = byte_grid == idx
            string_grid[mask] = obj_str

        return string_grid


def compress_map(string_grid: np.ndarray, valid_objects: Optional[Set[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """Convenience function for one-off compression."""
    compressor = MapCompressor(valid_objects)
    return compressor.compress(string_grid)


def decompress_map(byte_grid: np.ndarray, object_key: List[str]) -> np.ndarray:
    """Convenience function for one-off decompression."""
    compressor = MapCompressor()
    return compressor.decompress(byte_grid, object_key)
