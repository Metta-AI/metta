from typing import Optional

import numpy as np

from metta.mettagrid.mettagrid_c import PackedCoordinate
from metta.mettagrid.test_support.token_types import TokenTypes


class ObservationHelper:
    """Helper class for observation-related operations."""

    @staticmethod
    def find_tokens_at_location(obs: np.ndarray, x: int, y: int) -> np.ndarray:
        """Find all tokens at a specific location."""
        location = PackedCoordinate.pack(y, x)
        return obs[obs[:, 0] == location]

    @staticmethod
    def find_tokens_by_type(obs: np.ndarray, type_id: int) -> np.ndarray:
        """Find all tokens of a specific type."""
        return obs[obs[:, 1] == type_id]

    @staticmethod
    def find_tokens_by_value(obs: np.ndarray, value: int) -> np.ndarray:
        """Find all tokens with a specific value."""
        return obs[obs[:, 2] == value]

    @staticmethod
    def find_features_by_type(obs: np.ndarray, feature_type_id: int) -> np.ndarray:
        """Find all feature tokens with a specific feature type ID."""
        feature_tokens = ObservationHelper.find_tokens_by_type(obs, TokenTypes.TYPE_ID_FEATURE)
        return feature_tokens[feature_tokens[:, 2] == feature_type_id]

    @staticmethod
    def find_feature_at_location(obs: np.ndarray, x: int, y: int, feature_type_id: int) -> np.ndarray:
        """Find tokens of a specific feature type at a location."""
        location_tokens = ObservationHelper.find_tokens_at_location(obs, x, y)
        feature_tokens = location_tokens[location_tokens[:, 1] == TokenTypes.TYPE_ID_FEATURE]
        return feature_tokens[feature_tokens[:, 2] == feature_type_id]

    @staticmethod
    def find_token_value_at_location(obs: np.ndarray, x: int, y: int, token_type: int) -> Optional[int]:
        """Get the value of a specific token type at a location.

        Returns None if no token of that type exists at the location.
        """
        location_tokens = ObservationHelper.find_tokens_at_location(obs, x, y)
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
    def get_wall_positions(obs: np.ndarray) -> list[tuple[int, int]]:
        """Get all wall positions from an observation."""
        wall_tokens = ObservationHelper.find_features_by_type(obs, TokenTypes.WALL_TYPE_ID)
        return ObservationHelper.get_positions_from_tokens(wall_tokens)

    @staticmethod
    def count_walls(obs: np.ndarray) -> int:
        """Count the number of wall tokens in an observation."""
        return len(ObservationHelper.find_features_by_type(obs, TokenTypes.WALL_TYPE_ID))

    @staticmethod
    def has_wall_at(obs: np.ndarray, x: int, y: int) -> bool:
        """Check if there's a wall at the given location."""
        return len(ObservationHelper.find_feature_at_location(obs, x, y, TokenTypes.WALL_TYPE_ID)) > 0

    @staticmethod
    def count_features_by_type(obs: np.ndarray, feature_type_id: int) -> int:
        """Count the number of features with a specific type ID."""
        return len(ObservationHelper.find_features_by_type(obs, feature_type_id))

    @staticmethod
    def has_feature_at(obs: np.ndarray, x: int, y: int, feature_type_id: int) -> bool:
        """Check if there's a specific feature type at the given location."""
        return len(ObservationHelper.find_feature_at_location(obs, x, y, feature_type_id)) > 0

    @staticmethod
    def has_token_at(obs: np.ndarray, x: int, y: int, token_type: int) -> bool:
        """Check if there's a specific token type at the given location."""
        return ObservationHelper.find_token_value_at_location(obs, x, y, token_type) is not None
