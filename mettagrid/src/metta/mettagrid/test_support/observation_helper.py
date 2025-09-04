from typing import Optional

import numpy as np

from metta.mettagrid.mettagrid_c import PackedCoordinate


class ObservationHelper:
    """Helper class for observation-related operations."""

    @staticmethod
    def find_tokens(
        obs: np.ndarray,
        location: None | tuple[int, int] = None,
        feature_id: None | int = None,
        value: None | int = None,
    ) -> np.ndarray:
        """Filter tokens by location, feature id, and value."""
        tokens = obs
        if location is not None:
            tokens = tokens[tokens[:, 0] == PackedCoordinate.pack(location[1], location[0])]
        if feature_id is not None:
            tokens = tokens[tokens[:, 1] == feature_id]
        if value is not None:
            tokens = tokens[tokens[:, 2] == value]
        return tokens

    @staticmethod
    def find_token_values(
        obs: np.ndarray,
        location: None | tuple[int, int] = None,
        feature_id: None | int = None,
        value: None | int = None,
    ) -> np.ndarray:
        """Find the values of tokens by location, feature id, and value."""
        tokens = ObservationHelper.find_tokens(obs, location, feature_id, value)
        return tokens[:, 2]

    @staticmethod
    def find_token_value_at_location(obs: np.ndarray, x: int, y: int, token_type: int) -> Optional[int]:
        """Get the value of a specific token type at a location.

        Returns None if no token of that type exists at the location.
        """
        location_tokens = ObservationHelper.find_tokens(obs, location=(x, y))
        type_tokens = location_tokens[location_tokens[:, 1] == token_type]
        return int(type_tokens[0, 2]) if len(type_tokens) > 0 else None

    @staticmethod
    def get_positions_from_tokens(tokens: np.ndarray) -> list[tuple[int, int]]:
        """Extract (x, y) positions from tokens."""
        positions = []
        for token in tokens:
            coords = PackedCoordinate.unpack(token[0])
            if coords:
                row, col = coords
                positions.append((col, row))  # Return as (x, y)
        return positions
