#!/usr/bin/env python3
"""Consolidated examples for corridor generation using the unified angle-based API.

This demonstrates how to create:
1. Simple horizontal/vertical corridors
2. Radial patterns (like radial_mini.map)
3. Complex corridor layouts (like corridors.map)
4. Custom angled corridors
"""

from metta.map.mapgen import MapGen
from metta.map.scenes.angled_corridor_builder import (
    AngledCorridorBuilder,
    AngledCorridorBuilderParams,
    corridor,
    horizontal,
    vertical,
    radial_corridors,
    star_pattern,
)


# ==============================================================================
# EXAMPLE 1: Recreate corridors.map pattern
# ==============================================================================


def create_corridors_map():
    """Recreate the corridors.map pattern with one horizontal and multiple vertical corridors."""

    corridors = [
        # Main horizontal corridor (thick band)
        horizontal(y=13, thickness=7, x_start=1, x_end=73),
        # Vertical corridors at various positions with different properties
        vertical(x=10, thickness=3, y_start=1, y_end=10),  # Stops before horizontal
        vertical(x=20, thickness=4),  # Full height
        vertical(x=30, thickness=2, y_start=17, y_end=26),  # Starts after horizontal
        vertical(x=40, thickness=3),  # Full height
        vertical(x=50, thickness=5, y_start=5, y_end=20),  # Partial, crosses horizontal
        vertical(x=60, thickness=2),  # Full height
        vertical(x=68, thickness=3, y_start=8, y_end=18),  # Short, crosses horizontal
    ]

    map_config = MapGen.Config(
        width=74,
        height=26,
        border_width=2,
        root=AngledCorridorBuilder.factory(
            params=AngledCorridorBuilderParams(
                corridors=corridors,
                objects={"altar": 3},
                place_at_ends=True,
                agent_position=(13, 37),  # Center of horizontal corridor
            )
        ),
    )

    map_gen = MapGen(map_config)
    return map_gen.build()


# ==============================================================================
# EXAMPLE 2: Create radial_mini.map pattern
# ==============================================================================


def create_radial_mini():
    """Recreate radial_mini.map with 8 spokes radiating from center."""

    corridors = radial_corridors(
        center=(11, 11),  # Center of 22x22 map
        num_spokes=8,  # 8 directions
        length=8,  # Short spokes
        thickness=1,  # Thin corridors
    )

    map_config = MapGen.Config(
        width=22,
        height=22,
        border_width=1,
        root=AngledCorridorBuilder.factory(
            params=AngledCorridorBuilderParams(
                corridors=corridors,
                objects={"altar": 3},
                place_at_ends=True,
                agent_position=(11, 11),
            )
        ),
    )

    map_gen = MapGen(map_config)
    return map_gen.build()


# ==============================================================================
# EXAMPLE 3: Custom angled corridors
# ==============================================================================


def create_custom_angles():
    """Create corridors at specific custom angles."""

    center = (25, 25)

    corridors = [
        # Cardinal directions
        corridor(center, angle=0, length=20, thickness=3),  # East
        corridor(center, angle=90, length=20, thickness=3),  # North
        corridor(center, angle=180, length=20, thickness=3),  # West
        corridor(center, angle=270, length=20, thickness=3),  # South
        # Custom angles
        corridor(center, angle=30, length=15, thickness=2),  # 30 degrees
        corridor(center, angle=150, length=15, thickness=2),  # 150 degrees
        corridor(center, angle=210, length=15, thickness=2),  # 210 degrees
        corridor(center, angle=330, length=15, thickness=2),  # 330 degrees
    ]

    map_config = MapGen.Config(
        width=50,
        height=50,
        border_width=2,
        root=AngledCorridorBuilder.factory(
            params=AngledCorridorBuilderParams(
                corridors=corridors,
                objects={"altar": 4},
                place_at_ends=True,
                agent_position=center,
            )
        ),
    )

    map_gen = MapGen(map_config)
    return map_gen.build()


# ==============================================================================
# EXAMPLE 4: Star pattern (bidirectional corridors)
# ==============================================================================


def create_star():
    """Create a star pattern with corridors extending in both directions from center."""

    corridors = star_pattern(
        center=(30, 30),
        num_spokes=6,  # 6 spokes (12 endpoints)
        length=20,
        thickness=2,
    )

    map_config = MapGen.Config(
        width=60,
        height=60,
        border_width=2,
        root=AngledCorridorBuilder.factory(
            params=AngledCorridorBuilderParams(
                corridors=corridors,
                objects={"altar": 6},
                place_at_ends=True,
                agent_position=(30, 30),
            )
        ),
    )

    map_gen = MapGen(map_config)
    return map_gen.build()


# ==============================================================================
# EXAMPLE 5: Grid pattern using loops
# ==============================================================================


def create_grid():
    """Create a grid pattern using horizontal and vertical corridors."""

    corridors = []

    # Horizontal corridors every 10 units
    for y in [10, 20, 30, 40]:
        corridors.append(horizontal(y=y, thickness=2))

    # Vertical corridors every 12 units
    for x in [12, 24, 36, 48]:
        corridors.append(vertical(x=x, thickness=2))

    map_config = MapGen.Config(
        width=60,
        height=50,
        border_width=2,
        root=AngledCorridorBuilder.factory(
            params=AngledCorridorBuilderParams(
                corridors=corridors,
                objects={"altar": 8},
                place_at_intersections=True,  # Place at intersections
                agent_position=(25, 30),
            )
        ),
    )

    map_gen = MapGen(map_config)
    return map_gen.build()


# ==============================================================================
# API REFERENCE
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CORRIDOR GENERATION API REFERENCE")
    print("=" * 60)
    print()
    print("BASIC USAGE:")
    print("  corridor(center=(y,x), angle=degrees, length=n, thickness=t)")
    print()
    print("CONVENIENCE FUNCTIONS:")
    print("  horizontal(y=10, thickness=3, x_start=1, x_end=50)")
    print("  vertical(x=10, thickness=3, y_start=1, y_end=50)")
    print()
    print("PATTERN GENERATORS:")
    print("  radial_corridors(center, num_spokes=8, length=10, thickness=2)")
    print("  star_pattern(center, num_spokes=6, length=10, thickness=2)")
    print()
    print("ANGLE REFERENCE:")
    print("  0° = East (right)")
    print("  90° = North (up)")
    print("  180° = West (left)")
    print("  270° = South (down)")
    print()
    print("EXAMPLES:")
    print("  create_corridors_map()  # corridors.map pattern")
    print("  create_radial_mini()    # radial_mini.map pattern")
    print("  create_custom_angles()  # custom angled corridors")
    print("  create_star()           # star pattern")
    print("  create_grid()           # grid pattern")
    print("=" * 60)
