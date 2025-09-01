#!/usr/bin/env python3
"""Simple example showing how to create radial patterns like radial_mini.map."""

from metta.map.mapgen import MapGen
from metta.map.scenes.angled_corridor_builder import (
    AngledCorridorBuilder,
    AngledCorridorBuilderParams,
    corridor,
    radial_corridors,
)


def create_radial_mini():
    """Recreate radial_mini.map with just one function call."""

    # That's it! One function call creates all the spokes
    corridors = radial_corridors(
        center=(11, 11),  # Center of map
        num_spokes=8,  # 8 spokes (N, NE, E, SE, S, SW, W, NW)
        length=8,  # Length of each spoke
        thickness=1,  # Thin corridors
    )

    # Build the map
    map_config = MapGen.Config(
        width=22,
        height=22,
        border_width=1,
        root=AngledCorridorBuilder.factory(
            params=AngledCorridorBuilderParams(
                corridors=corridors,
                objects={"altar": 3},
                place_at_ends=True,
                agent_position=(11, 11),  # Agent at center
            )
        ),
    )

    map_gen = MapGen(map_config)
    return map_gen.build()


def create_custom_radial():
    """Create a custom radial pattern with different angles."""

    center = (25, 25)
    corridors = []

    # Method 1: Use the helper function for uniform spokes
    corridors.extend(
        radial_corridors(
            center=center,
            num_spokes=6,  # 6 evenly spaced spokes
            length=15,
            thickness=2,
        )
    )

    # Method 2: Add individual corridors at specific angles
    corridors.append(corridor(center, angle=30, length=20, thickness=3))
    corridors.append(corridor(center, angle=120, length=18, thickness=2))
    corridors.append(corridor(center, angle=225, length=22, thickness=1))

    # Build the map
    map_config = MapGen.Config(
        width=50,
        height=50,
        border_width=2,
        root=AngledCorridorBuilder.factory(
            params=AngledCorridorBuilderParams(
                corridors=corridors,
                objects={"altar": 5},
                place_at_ends=True,
                agent_position=center,
            )
        ),
    )

    map_gen = MapGen(map_config)
    return map_gen.build()


def unified_corridor_api():
    """Show the unified API - one function for all corridor types."""

    # The unified corridor function handles everything:
    corridors = [
        # Horizontal (angle = 0)
        corridor(center=(10, 5), angle=0, length=40, thickness=3),
        # Vertical (angle = 270 for down, 90 for up)
        corridor(center=(5, 25), angle=270, length=35, thickness=3),
        # Diagonal
        corridor(center=(25, 25), angle=45, length=20, thickness=2),
        # Any custom angle
        corridor(center=(25, 25), angle=137, length=15, thickness=2),
        # Bidirectional (extends both ways from center)
        corridor(center=(25, 25), angle=30, length=18, thickness=2, bidirectional=True),
    ]

    # Build the map
    map_config = MapGen.Config(
        width=50,
        height=50,
        border_width=2,
        root=AngledCorridorBuilder.factory(
            params=AngledCorridorBuilderParams(
                corridors=corridors,
                objects={"altar": 4},
                place_at_ends=True,
            )
        ),
    )

    map_gen = MapGen(map_config)
    return map_gen.build()


if __name__ == "__main__":
    print("Creating radial patterns is now trivial!")
    print()
    print("For radial_mini.map, just use:")
    print("  radial_corridors(center=(11,11), num_spokes=8, length=8)")
    print()
    print("The unified corridor() function handles all cases:")
    print("  corridor(center, angle=0, length)     # Horizontal")
    print("  corridor(center, angle=90, length)    # Vertical up")
    print("  corridor(center, angle=45, length)    # Diagonal")
    print("  corridor(center, angle=137, length)   # Any angle!")
    print()
    print("Key angles to remember:")
    print("  0° = East (right)")
    print("  90° = North (up)")
    print("  180° = West (left)")
    print("  270° = South (down)")
