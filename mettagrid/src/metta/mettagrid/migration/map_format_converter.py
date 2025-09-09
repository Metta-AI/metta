"""
Map format conversion utilities for migrating between legacy and int-based storage.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from metta.mettagrid.map_builder.map_builder import (
    GameMap,
    MapGridInt,
    MapGridLegacy,
    create_int_grid,
    create_legacy_grid,
)
from metta.mettagrid.mettagrid_config import GameConfig
from metta.mettagrid.object_types import ObjectTypes
from metta.mettagrid.type_mapping import TypeMapping


class MapFormatConverter:
    """
    Utility class for converting between legacy string-based and int-based map formats.

    This class provides bidirectional conversion capabilities and validation to ensure
    data integrity during the migration process.
    """

    def __init__(self, game_config: Optional[GameConfig] = None):
        """
        Initialize the converter with optional GameConfig.

        Args:
            game_config: GameConfig instance for validation. If None, uses standard ObjectTypes.
        """
        self.game_config = game_config
        self.type_mapping = TypeMapping(game_config)

    def legacy_to_int(self, legacy_map: Union[List[List[str]], MapGridLegacy]) -> Tuple[MapGridInt, List[str]]:
        """
        Convert a legacy string-based map to int-based format.

        Args:
            legacy_map: Legacy map as nested lists or numpy array of strings

        Returns:
            Tuple of (int_grid, decoder_key)

        Raises:
            ValueError: If map contains unknown object types
        """
        # Convert input to numpy array if needed
        if isinstance(legacy_map, list):
            legacy_array = np.array(legacy_map, dtype=np.str_)
        else:
            legacy_array = legacy_map

        height, width = legacy_array.shape
        int_grid = create_int_grid(height, width)

        # Convert each cell
        for r in range(height):
            for c in range(width):
                object_name = str(legacy_array[r, c])
                try:
                    type_id = self.type_mapping.get_type_id(object_name)
                    int_grid[r, c] = type_id
                except KeyError as e:
                    raise ValueError(f"Unknown object type '{object_name}' at position ({r}, {c})") from e

        return int_grid, self.type_mapping.decoder_key

    def int_to_legacy(self, int_grid: MapGridInt, decoder_key: List[str]) -> MapGridLegacy:
        """
        Convert an int-based map to legacy string format.

        Args:
            int_grid: Int-based map grid
            decoder_key: List mapping type_ids to object names

        Returns:
            Legacy string-based map grid

        Raises:
            ValueError: If int_grid contains invalid type_ids
        """
        height, width = int_grid.shape
        legacy_grid = create_legacy_grid(height, width)

        # Convert each cell
        for r in range(height):
            for c in range(width):
                type_id = int(int_grid[r, c])

                if type_id >= len(decoder_key):
                    raise ValueError(
                        f"Invalid type_id {type_id} at position ({r}, {c}). "
                        f"Decoder key only has {len(decoder_key)} entries."
                    )

                object_name = decoder_key[type_id]
                legacy_grid[r, c] = object_name

        return legacy_grid

    def convert_game_map_to_int(self, game_map: GameMap) -> GameMap:
        """
        Convert a GameMap from legacy to int-based format.

        Args:
            game_map: GameMap instance (must be legacy format)

        Returns:
            New GameMap instance with int-based grid

        Raises:
            ValueError: If input is not legacy format or conversion fails
        """
        if not game_map.is_legacy():
            raise ValueError("Input GameMap is not in legacy format")

        legacy_grid = game_map.get_legacy_grid()
        int_grid, decoder_key = self.legacy_to_int(legacy_grid)

        return GameMap(grid=int_grid, decoder_key=decoder_key)

    def convert_game_map_to_legacy(self, game_map: GameMap) -> GameMap:
        """
        Convert a GameMap from int-based to legacy format.

        Args:
            game_map: GameMap instance (must be int-based format)

        Returns:
            New GameMap instance with legacy string-based grid

        Raises:
            ValueError: If input is not int-based format or conversion fails
        """
        if not game_map.is_int_based():
            raise ValueError("Input GameMap is not in int-based format")

        int_grid = game_map.grid
        decoder_key = game_map.decoder_key

        if decoder_key is None:
            raise ValueError("GameMap missing decoder_key for int-based conversion")

        legacy_grid = self.int_to_legacy(int_grid, decoder_key)

        return GameMap(grid=legacy_grid, decoder_key=None)

    def validate_conversion_integrity(self, original: GameMap, converted: GameMap) -> Dict[str, Any]:
        """
        Validate that conversion preserves map integrity.

        Args:
            original: Original GameMap
            converted: Converted GameMap

        Returns:
            Dictionary with validation results
        """
        results = {"shape_match": False, "content_match": False, "errors": [], "stats": {}}

        try:
            # Check shapes
            orig_shape = original.grid.shape
            conv_shape = converted.grid.shape
            results["shape_match"] = orig_shape == conv_shape

            if not results["shape_match"]:
                results["errors"].append(f"Shape mismatch: {orig_shape} vs {conv_shape}")
                return results

            # Convert both to int format for semantic comparison
            if original.is_legacy():
                orig_int, _ = self.legacy_to_int(original.grid)
            else:
                orig_int = original.grid

            if converted.is_legacy():
                conv_int, _ = self.legacy_to_int(converted.grid)
            else:
                conv_int = converted.grid

            # Compare semantic content (type_ids)
            content_match = np.array_equal(orig_int, conv_int)
            results["content_match"] = content_match

            if not content_match:
                # Find semantic differences
                diff_positions = np.where(orig_int != conv_int)
                results["errors"].append(f"Semantic differences at {len(diff_positions[0])} positions")

                # Sample some differences
                for i in range(min(5, len(diff_positions[0]))):
                    r, c = diff_positions[0][i], diff_positions[1][i]
                    orig_type_id = orig_int[r, c]
                    conv_type_id = conv_int[r, c]
                    results["errors"].append(f"  Position ({r}, {c}): type_id {orig_type_id} vs {conv_type_id}")

            # Generate statistics
            height, width = orig_shape
            total_cells = height * width

            # Count object types using decoder key and int format
            flat_int = orig_int.flatten()
            unique_ids, counts = np.unique(flat_int, return_counts=True)

            # Convert type_ids back to names for readable stats
            unique_names = []
            for type_id in unique_ids:
                if converted.decoder_key and type_id < len(converted.decoder_key):
                    unique_names.append(converted.decoder_key[type_id])
                else:
                    unique_names.append(f"type_id_{type_id}")
            unique = np.array(unique_names)

            results["stats"] = {
                "total_cells": total_cells,
                "unique_objects": len(unique),
                "object_counts": dict(zip(unique, counts.tolist(), strict=False)),
                "empty_cells": dict(zip(unique, counts.tolist(), strict=False)).get("empty", 0),
            }

        except Exception as e:
            results["errors"].append(f"Validation error: {str(e)}")

        return results

    def batch_convert_maps(self, maps: List[GameMap], target_format: str = "int") -> List[GameMap]:
        """
        Convert multiple GameMaps to the target format.

        Args:
            maps: List of GameMap instances to convert
            target_format: "int" for int-based, "legacy" for string-based

        Returns:
            List of converted GameMap instances

        Raises:
            ValueError: If target_format is invalid or conversion fails
        """
        if target_format not in ["int", "legacy"]:
            raise ValueError("target_format must be 'int' or 'legacy'")

        converted_maps = []

        for i, game_map in enumerate(maps):
            try:
                if target_format == "int":
                    if game_map.is_legacy():
                        converted = self.convert_game_map_to_int(game_map)
                    else:
                        converted = game_map  # Already int-based
                else:  # target_format == "legacy"
                    if game_map.is_int_based():
                        converted = self.convert_game_map_to_legacy(game_map)
                    else:
                        converted = game_map  # Already legacy

                converted_maps.append(converted)

            except Exception as e:
                raise ValueError(f"Failed to convert map {i}: {str(e)}") from e

        return converted_maps

    def get_conversion_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the type mapping used by this converter.

        Returns:
            Dictionary with mapping statistics
        """
        decoder_key = self.type_mapping.decoder_key

        return {
            "total_types": len(decoder_key),
            "decoder_key": decoder_key,
            "name_to_id_mapping": dict(self.type_mapping.name_to_type_id),
            "uses_game_config": self.game_config is not None,
            "standard_types": {
                "empty": ObjectTypes.EMPTY,
                "wall": ObjectTypes.WALL,
                "agent_base": ObjectTypes.AGENT_BASE,
            },
        }
