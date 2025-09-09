"""
Validation utilities for map format correctness and consistency.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from metta.mettagrid.map_builder.map_builder import GameMap, MapGridInt, MapGridLegacy
from metta.mettagrid.mettagrid_config import GameConfig
from metta.mettagrid.object_types import ObjectTypes
from metta.mettagrid.type_mapping import TypeMapping


class ValidationError(Exception):
    """Custom exception for validation errors."""

    pass


class MapFormatValidator:
    """
    Comprehensive validation utilities for map formats.

    This class provides validation for both legacy and int-based map formats,
    ensuring data integrity and consistency across format conversions.
    """

    def __init__(self, game_config: Optional[GameConfig] = None):
        """
        Initialize the validator with optional GameConfig.

        Args:
            game_config: GameConfig instance for validation. If None, uses standard ObjectTypes.
        """
        self.game_config = game_config
        self.type_mapping = TypeMapping(game_config)

    def validate_legacy_map(self, legacy_map: MapGridLegacy) -> Dict[str, Any]:
        """
        Validate a legacy string-based map for correctness.

        Args:
            legacy_map: Legacy map grid to validate

        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {},
            "unknown_objects": set(),
        }

        try:
            height, width = legacy_map.shape
            results["stats"]["dimensions"] = (height, width)
            results["stats"]["total_cells"] = height * width

            # Track object types and positions
            object_counts = {}
            agent_positions = []
            empty_positions = []

            for r in range(height):
                for c in range(width):
                    object_name = str(legacy_map[r, c])

                    # Count occurrences
                    object_counts[object_name] = object_counts.get(object_name, 0) + 1

                    # Check if object type is known
                    try:
                        self.type_mapping.get_type_id(object_name)
                    except KeyError:
                        results["unknown_objects"].add(object_name)
                        results["errors"].append(f"Unknown object '{object_name}' at ({r}, {c})")
                        results["valid"] = False

                    # Track special positions
                    if object_name in ["empty", ".", " "]:
                        empty_positions.append((r, c))
                    elif "agent" in object_name.lower():
                        agent_positions.append((r, c))

            results["stats"]["object_counts"] = object_counts
            results["stats"]["unique_objects"] = len(object_counts)
            results["stats"]["agent_count"] = len(agent_positions)
            results["stats"]["empty_count"] = len(empty_positions)
            results["stats"]["agent_positions"] = agent_positions[:10]  # Sample

            # Validate agent count if GameConfig available
            if self.game_config and hasattr(self.game_config, "num_agents"):
                expected_agents = self.game_config.num_agents
                actual_agents = len(agent_positions)

                if actual_agents != expected_agents:
                    results["errors"].append(f"Agent count mismatch: expected {expected_agents}, found {actual_agents}")
                    results["valid"] = False

            # Check for empty map
            if len(empty_positions) == height * width:
                results["warnings"].append("Map contains only empty cells")

            # Check for extremely dense maps
            non_empty_ratio = (height * width - len(empty_positions)) / (height * width)
            if non_empty_ratio > 0.8:
                results["warnings"].append(f"Map is very dense ({non_empty_ratio:.1%} non-empty)")

        except Exception as e:
            results["errors"].append(f"Validation exception: {str(e)}")
            results["valid"] = False

        return results

    def validate_int_map(self, int_map: MapGridInt, decoder_key: List[str]) -> Dict[str, Any]:
        """
        Validate an int-based map for correctness.

        Args:
            int_map: Int-based map grid to validate
            decoder_key: Decoder key mapping type_ids to object names

        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {},
            "invalid_type_ids": set(),
        }

        try:
            height, width = int_map.shape
            results["stats"]["dimensions"] = (height, width)
            results["stats"]["total_cells"] = height * width
            results["stats"]["decoder_key_length"] = len(decoder_key)

            # Track type_id usage
            type_id_counts = {}
            agent_positions = []
            empty_positions = []

            # Get max valid type_id
            max_valid_type_id = len(decoder_key) - 1

            for r in range(height):
                for c in range(width):
                    type_id = int(int_map[r, c])

                    # Count occurrences
                    type_id_counts[type_id] = type_id_counts.get(type_id, 0) + 1

                    # Validate type_id range
                    if type_id < 0 or type_id > max_valid_type_id:
                        results["invalid_type_ids"].add(type_id)
                        results["errors"].append(f"Invalid type_id {type_id} at ({r}, {c})")
                        results["valid"] = False
                        continue

                    # object_name = decoder_key[type_id]  # Not used in validation

                    # Track special positions
                    if type_id == ObjectTypes.EMPTY:
                        empty_positions.append((r, c))
                    elif type_id >= ObjectTypes.AGENT_BASE:
                        agent_positions.append((r, c))

            results["stats"]["type_id_counts"] = type_id_counts
            results["stats"]["unique_type_ids"] = len(type_id_counts)
            results["stats"]["agent_count"] = len(agent_positions)
            results["stats"]["empty_count"] = len(empty_positions)
            results["stats"]["agent_positions"] = agent_positions[:10]  # Sample

            # Check for unused decoder entries
            used_type_ids = set(type_id_counts.keys())
            available_type_ids = set(range(len(decoder_key)))
            unused_type_ids = available_type_ids - used_type_ids

            if unused_type_ids:
                results["warnings"].append(f"Unused decoder entries: {sorted(unused_type_ids)}")

            # Validate against GameConfig if available
            if self.game_config and hasattr(self.game_config, "num_agents"):
                expected_agents = self.game_config.num_agents
                actual_agents = len(agent_positions)

                if actual_agents != expected_agents:
                    results["errors"].append(f"Agent count mismatch: expected {expected_agents}, found {actual_agents}")
                    results["valid"] = False

            # Check data type constraints
            if int_map.dtype != np.uint8:
                results["warnings"].append(f"Map dtype is {int_map.dtype}, expected uint8")

            # Check for type_id overflow
            max_used_type_id = max(type_id_counts.keys()) if type_id_counts else 0
            if max_used_type_id > 255:
                results["errors"].append(f"Max type_id {max_used_type_id} exceeds uint8 limit")
                results["valid"] = False

        except Exception as e:
            results["errors"].append(f"Validation exception: {str(e)}")
            results["valid"] = False

        return results

    def validate_game_map(self, game_map: GameMap) -> Dict[str, Any]:
        """
        Validate a GameMap instance for correctness.

        Args:
            game_map: GameMap instance to validate

        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "format": None,
            "stats": {},
        }

        try:
            # Determine format
            if game_map.is_legacy():
                results["format"] = "legacy"
                legacy_results = self.validate_legacy_map(game_map.grid)
                results.update(legacy_results)

            elif game_map.is_int_based():
                results["format"] = "int"
                if game_map.decoder_key is None:
                    results["errors"].append("Int-based GameMap missing decoder_key")
                    results["valid"] = False
                else:
                    int_results = self.validate_int_map(game_map.grid, game_map.decoder_key)
                    results.update(int_results)
            else:
                results["errors"].append("GameMap format could not be determined")
                results["valid"] = False

            # Additional GameMap-specific validations
            if hasattr(game_map, "grid") and game_map.grid is not None:
                results["stats"]["grid_shape"] = game_map.grid.shape
                results["stats"]["grid_dtype"] = str(game_map.grid.dtype)

        except Exception as e:
            results["errors"].append(f"GameMap validation exception: {str(e)}")
            results["valid"] = False

        return results

    def validate_conversion_consistency(
        self, original: GameMap, converted: GameMap, converted_back: GameMap
    ) -> Dict[str, Any]:
        """
        Validate round-trip conversion consistency.

        Tests: original -> converted -> converted_back should equal original

        Args:
            original: Original GameMap
            converted: GameMap converted to different format
            converted_back: GameMap converted back to original format

        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "round_trip_consistent": False,
            "stats": {},
        }

        try:
            # Validate individual maps
            orig_valid = self.validate_game_map(original)
            conv_valid = self.validate_game_map(converted)
            back_valid = self.validate_game_map(converted_back)

            if not orig_valid["valid"]:
                results["errors"].append("Original map validation failed")
                results["valid"] = False

            if not conv_valid["valid"]:
                results["errors"].append("Converted map validation failed")
                results["valid"] = False

            if not back_valid["valid"]:
                results["errors"].append("Round-trip converted map validation failed")
                results["valid"] = False

            # Check semantic consistency using type_ids
            from .map_format_converter import MapFormatConverter

            converter = MapFormatConverter()

            # Convert both to int format for semantic comparison
            if original.is_legacy():
                orig_int, _ = converter.legacy_to_int(original.grid)
            else:
                orig_int = original.grid

            if converted_back.is_legacy():
                back_int, _ = converter.legacy_to_int(converted_back.grid)
            else:
                back_int = converted_back.grid

            # Compare semantic content
            content_match = np.array_equal(orig_int, back_int)
            results["round_trip_consistent"] = content_match

            if not content_match:
                results["errors"].append("Round-trip conversion changed semantic content")
                results["valid"] = False

                # Find semantic differences
                if orig_int.shape == back_int.shape:
                    diff_mask = orig_int != back_int
                    diff_count = np.sum(diff_mask)
                    results["stats"]["differences"] = int(diff_count)

                    if diff_count < 20:  # Show all differences if few
                        diff_positions = np.where(diff_mask)
                        for i in range(len(diff_positions[0])):
                            r, c = diff_positions[0][i], diff_positions[1][i]
                            orig_type_id = orig_int[r, c]
                            back_type_id = back_int[r, c]
                            results["errors"].append(
                                f"Semantic difference at ({r}, {c}): type_id {orig_type_id} vs {back_type_id}"
                            )

            results["stats"]["formats"] = {
                "original": "legacy" if original.is_legacy() else "int",
                "converted": "legacy" if converted.is_legacy() else "int",
                "converted_back": "legacy" if converted_back.is_legacy() else "int",
            }

        except Exception as e:
            results["errors"].append(f"Consistency validation exception: {str(e)}")
            results["valid"] = False

        return results

    def validate_decoder_key(self, decoder_key: List[str]) -> Dict[str, Any]:
        """
        Validate a decoder key for correctness.

        Args:
            decoder_key: List of object names indexed by type_id

        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {},
        }

        try:
            results["stats"]["length"] = len(decoder_key)

            # Check for required entries
            if len(decoder_key) == 0:
                results["errors"].append("Decoder key is empty")
                results["valid"] = False
                return results

            if decoder_key[0] != "empty":
                results["errors"].append("Decoder key must start with 'empty' at index 0")
                results["valid"] = False

            # Check for duplicates
            unique_names = set(decoder_key)
            if len(unique_names) != len(decoder_key):
                duplicates = [name for name in decoder_key if decoder_key.count(name) > 1]
                results["errors"].append(f"Decoder key contains duplicates: {set(duplicates)}")
                results["valid"] = False

            # Check for empty strings
            empty_indices = [i for i, name in enumerate(decoder_key) if not name.strip()]
            if empty_indices:
                results["errors"].append(f"Empty object names at indices: {empty_indices}")
                results["valid"] = False

            # Statistical analysis
            results["stats"]["unique_names"] = len(unique_names)
            results["stats"]["has_empty"] = "empty" in decoder_key
            results["stats"]["has_wall"] = "wall" in decoder_key
            results["stats"]["agent_types"] = [name for name in decoder_key if "agent" in name.lower()]

            # Check for standard object types
            standard_objects = ["empty", "wall"]
            missing_standard = [obj for obj in standard_objects if obj not in decoder_key]
            if missing_standard:
                results["warnings"].append(f"Missing standard objects: {missing_standard}")

        except Exception as e:
            results["errors"].append(f"Decoder key validation exception: {str(e)}")
            results["valid"] = False

        return results
