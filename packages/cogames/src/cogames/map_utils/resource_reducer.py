"""Utility for reducing resources in ASCII maps based on difficulty levels."""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING

from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.map_builder.map_builder import MapBuilderConfig

if TYPE_CHECKING:
    pass


def reduce_map_resources(
    map_input: str | Path | MapBuilderConfig,
    resource_levels: dict[str, int],
    seed: int | None = None,
) -> AsciiMapBuilder.Config:
    """
    Reduce resources in an ASCII map based on difficulty levels.

    Args:
        map_input: Can be a file path (str/Path) to a .map file, or an existing MapBuilderConfig
        resource_levels: Dict with keys "G", "S", "O", "C", "&", "+", "=", values 0-10
            - Level 10 = 100% (no reduction)
            - Level 9 = 90% kept (10% removed)
            - Level 8 = 80% kept (20% removed)
            - etc.
        seed: Optional random seed for reproducibility

    Returns:
        AsciiMapBuilder.Config with reduced resources, ready for use in missions

    Example:
        >>> modified_map = reduce_map_resources(
        ...     "evals/diagnostic_extract_lab.map",
        ...     resource_levels={"G": 10, "S": 10, "O": 10, "C": 7, "&": 10, "+": 10, "=": 10},
        ...     seed=42
        ... )
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)

    # Load map configuration
    if isinstance(map_input, (str, Path)):
        # Load from file path
        if isinstance(map_input, str):
            map_path = Path(map_input)
        else:
            map_path = map_input

        # Handle relative paths - check if it's relative to maps directory
        if not map_path.is_absolute():
            # Try relative to cogames maps directory
            maps_dir = Path(__file__).parent.parent / "maps"
            potential_path = maps_dir / map_path
            if potential_path.exists():
                map_path = potential_path
            # If it doesn't exist, try the path as-is (might be absolute or relative to cwd)

        config = MapBuilderConfig.from_uri(str(map_path))
    else:
        # Already a MapBuilderConfig
        config = map_input

    # Ensure it's an AsciiMapBuilder config
    if not isinstance(config, AsciiMapBuilder.Config):
        raise ValueError(
            f"Expected AsciiMapBuilder.Config, got {type(config).__name__}. "
            "This function only works with ASCII map files."
        )

    # Extract map data and char mapping
    map_data = [list(row) for row in config.map_data]  # Make a mutable copy
    char_to_map_name = config.char_to_map_name

    # Resource character mapping
    resource_chars = {
        "G": "G",  # germanium_extractor
        "S": "S",  # silicon_extractor
        "O": "O",  # oxygen_extractor
        "C": "C",  # carbon_extractor
        "&": "&",  # assembler
        "+": "+",  # charger
        "=": "=",  # chest
    }

    # Process each resource type
    for resource_key, char in resource_chars.items():
        if resource_key not in resource_levels:
            continue  # Skip resources not specified in levels

        level = resource_levels[resource_key]
        if level < 0 or level > 10:
            raise ValueError(f"Resource level for '{resource_key}' must be between 0 and 10, got {level}")

        # Find all positions of this character in the map
        positions: list[tuple[int, int]] = []
        for row_idx, row in enumerate(map_data):
            for col_idx, cell in enumerate(row):
                if cell == char:
                    positions.append((row_idx, col_idx))

        total_count = len(positions)
        if total_count == 0:
            continue  # Resource not present in map, skip

        # Calculate how many to remove
        # Level 10 = 100% kept (0% removed)
        # Level 9 = 90% kept (10% removed)
        # Level 8 = 80% kept (20% removed)
        # etc.
        removal_fraction = 1.0 - (level / 10.0)
        num_to_remove = round(total_count * removal_fraction)

        # Ensure at least 1 remains
        num_to_remove = min(num_to_remove, total_count - 1)

        if num_to_remove > 0:
            # Randomly sample positions to remove
            positions_to_remove = random.sample(positions, num_to_remove)

            # Replace those positions with "." (empty)
            for row_idx, col_idx in positions_to_remove:
                map_data[row_idx][col_idx] = "."

    # Create and return new AsciiMapBuilder.Config
    # Convert map_data back from list of lists to list of strings
    map_data_strings = ["".join(row) if isinstance(row, list) else row for row in map_data]

    return AsciiMapBuilder.Config(
        map_data=map_data_strings,
        char_to_map_name=char_to_map_name,
    )
