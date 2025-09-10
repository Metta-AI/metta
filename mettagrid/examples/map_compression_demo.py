#!/usr/bin/env python
"""Demo script showing map compression and validation features."""

import numpy as np

from metta.mettagrid.map_builder.random import RandomMapBuilder
from metta.mettagrid.mapgen.utils.storable_map import StorableMap
from metta.mettagrid.mettagrid_config import GameConfig


def main():
    """Demonstrate map compression and validation."""

    # 1. Create a minimal game config with known objects
    print("Creating game configuration...")
    from metta.mettagrid.mettagrid_config import ConverterConfig, WallConfig

    # Create a simple game config manually
    game_config = GameConfig(
        num_agents=4,
        objects={
            "wall": WallConfig(type_id=1, swappable=False),
            "generator": ConverterConfig(type_id=2, output_resources={"ore": 1}, cooldown=10),
            "converter": ConverterConfig(
                type_id=3, input_resources={"ore": 1}, output_resources={"energy": 1}, cooldown=5
            ),
            # Agents are added automatically by the system based on num_agents
        },
    )

    # 2. Create a map builder with validation
    print("\nBuilding map with validation...")
    map_config = RandomMapBuilder.Config(
        width=20,
        height=20,
        objects={
            "generator": 5,
            "converter": 3,
        },
        agents=4,  # Use simple agent count
        border_width=1,
        border_object="wall",
    )

    builder = RandomMapBuilder(map_config)
    builder.set_game_config(game_config)  # Enable validation

    # Build with validation
    game_map = builder.build_validated()

    # 3. Show compression results
    print(f"\nMap dimensions: {game_map.grid.shape}")
    print(f"Unique objects in map: {np.unique(game_map.grid)}")

    if game_map.byte_grid is not None:
        print("\n✓ Map compression successful!")
        print(f"  Object key: {game_map.object_key}")
        print(f"  String grid size: {game_map.grid.nbytes} bytes")
        print(f"  Byte grid size: {game_map.byte_grid.nbytes} bytes")
        print(f"  Compression ratio: {game_map.grid.nbytes / game_map.byte_grid.nbytes:.1f}x")

    # 4. Test map storage with compression
    print("\nTesting map storage...")
    storable_map = StorableMap(grid=game_map.grid, metadata={"demo": True, "version": "1.0"}, config=map_config)

    # Convert to string (uses compression internally)
    map_str = str(storable_map)
    print(f"Stored map size: {len(map_str)} characters")

    # Show a snippet of the compressed format
    lines = map_str.split("\n")
    print("\nStored format preview:")
    print("---")
    for line in lines[:15]:  # First 15 lines
        print(line)
    print("...")

    # 5. Test validation with unknown objects
    print("\n\nTesting validation with unknown objects...")
    bad_config = RandomMapBuilder.Config(
        width=10,
        height=10,
        objects={
            "unknown_object": 2,  # This doesn't exist in game_config
            "generator": 1,
        },
        agents=1,
    )

    bad_builder = RandomMapBuilder(bad_config)
    bad_builder.set_game_config(game_config)

    # This should log a warning but still build
    print("Building map with unknown objects...")
    import logging

    logging.basicConfig(level=logging.WARNING)

    _ = bad_builder.build_validated()
    print("Map built despite validation error (backward compatibility)")


if __name__ == "__main__":
    main()
