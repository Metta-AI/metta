from typing import Optional

import numpy as np

from metta.mettagrid.mettagrid_c import PackedCoordinate
from metta.mettagrid.test_support.token_types import TokenTypes


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

    @staticmethod
    def count_walls(obs: np.ndarray) -> int:
        """Count the number of wall tokens in an observation."""
        return len(
            ObservationHelper.find_tokens(obs, feature_id=TokenTypes.TYPE_ID_FEATURE, value=TokenTypes.WALL_TYPE_ID)
        )

    @staticmethod
    def has_wall_at(obs: np.ndarray, x: int, y: int) -> bool:
        """Check if there's a wall at the given location."""
        wall_tokens = ObservationHelper.find_tokens(
            obs, location=(x, y), feature_id=TokenTypes.TYPE_ID_FEATURE, value=TokenTypes.WALL_TYPE_ID
        )
        return len(wall_tokens) > 0

    @staticmethod
    def count_features_by_type(obs: np.ndarray, feature_type_id: int) -> int:
        """Count the number of features with a specific type ID."""
        return len(ObservationHelper.find_tokens(obs, feature_id=TokenTypes.TYPE_ID_FEATURE, value=feature_type_id))
