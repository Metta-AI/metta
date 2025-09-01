#!/usr/bin/env python3
"""Test radial corridor generation to recreate patterns like radial_mini.map."""

import math

import matplotlib.pyplot as plt
import numpy as np
from metta.map.mapgen import MapGen
from metta.map.scenes.angled_corridor_builder import (
    AngledCorridorBuilder,
    AngledCorridorBuilderParams,
    corridor,
    horizontal,
    radial_corridors,
    star_pattern,
    vertical,
)


def visualize_map(grid: np.ndarray, title: str = "Map"):
    """Simple visualization of a map grid."""
    visual = np.zeros_like(grid, dtype=float)

    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            cell = str(grid[y, x])
            if "wall" in cell or cell == "#":
                visual[y, x] = 0  # Black for walls
            elif "agent" in cell or cell == "@":
                visual[y, x] = 0.5  # Gray for agent
            elif "altar" in cell or cell == "_":
                visual[y, x] = 1.0  # White for goals
            else:
                visual[y, x] = 0.7  # Light gray for empty

    plt.figure(figsize=(10, 10))
    plt.imshow(visual, cmap="gray", interpolation="nearest")
    plt.title(title)
    plt.axis("off")
    return plt.gcf()


def test_radial_mini():
    """Recreate something like radial_mini.map using angled corridors."""
    print("\n=== Creating radial_mini.map Pattern ===")

    # radial_mini has 8 spokes from center
    center = (11, 11)  # Center of a 22x22 map

    # Create radial spokes
    corridors = radial_corridors(
        center=center,
        num_spokes=8,  # 8 directions
        length=8,  # Short spokes for mini version
        thickness=1,  # Thin corridors
    )

    # Create the map
    map_config = MapGen.Config(
        width=22,
        height=22,
        border_width=1,
        root=AngledCorridorBuilder.factory(
            params=AngledCorridorBuilderParams(
                corridors=corridors,
                objects={"altar": 3},
                place_at_ends=True,  # Place altars at spoke ends
                agent_position=center,  # Agent at center
            )
        ),
    )

    map_gen = MapGen(map_config)
    game_map = map_gen.build()

    # Visualize
    visualize_map(game_map.grid, "Radial Mini Pattern")
    plt.show()

    return game_map


def test_custom_angles():
    """Create corridors at specific custom angles."""
    print("\n=== Custom Angle Corridors ===")

    center = (25, 25)

    # Create corridors at specific angles
    corridors = [
        # Cardinal directions
        corridor(center, angle=0, length=20, thickness=3),  # East
        corridor(center, angle=90, length=20, thickness=3),  # North
        corridor(center, angle=180, length=20, thickness=3),  # West
        corridor(center, angle=270, length=20, thickness=3),  # South
        # Diagonal directions
        corridor(center, angle=45, length=15, thickness=2),  # NE
        corridor(center, angle=135, length=15, thickness=2),  # NW
        corridor(center, angle=225, length=15, thickness=2),  # SW
        corridor(center, angle=315, length=15, thickness=2),  # SE
    ]

    # Create the map
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
    game_map = map_gen.build()

    # Visualize
    visualize_map(game_map.grid, "Custom Angle Corridors")
    plt.show()

    return game_map


def test_star_pattern():
    """Create a star pattern with bidirectional corridors."""
    print("\n=== Star Pattern ===")

    center = (30, 30)

    # Create a star with 6 points
    corridors = star_pattern(
        center=center,
        num_spokes=6,  # Creates 6 bidirectional spokes
        length=20,
        thickness=2,
    )

    # Create the map
    map_config = MapGen.Config(
        width=60,
        height=60,
        border_width=2,
        root=AngledCorridorBuilder.factory(
            params=AngledCorridorBuilderParams(
                corridors=corridors,
                objects={"altar": 6},
                place_at_ends=True,
                agent_position=center,
            )
        ),
    )

    map_gen = MapGen(map_config)
    game_map = map_gen.build()

    # Visualize
    visualize_map(game_map.grid, "Star Pattern")
    plt.show()

    return game_map


def test_mixed_corridors():
    """Mix traditional horizontal/vertical with angled corridors."""
    print("\n=== Mixed Corridor Types ===")

    # You can mix the convenience functions with angled corridors
    corridors = [
        # Traditional horizontal and vertical
        horizontal(y=25, thickness=5),  # Main horizontal
        vertical(x=30, thickness=4),  # Main vertical
        # Add some angled corridors
        corridor((25, 30), angle=30, length=20, thickness=2),
        corridor((25, 30), angle=150, length=20, thickness=2),
        corridor((25, 30), angle=210, length=20, thickness=2),
        corridor((25, 30), angle=330, length=20, thickness=2),
    ]

    # Create the map
    map_config = MapGen.Config(
        width=60,
        height=50,
        border_width=2,
        root=AngledCorridorBuilder.factory(
            params=AngledCorridorBuilderParams(
                corridors=corridors,
                objects={"altar": 5},
                place_at_ends=True,
                place_at_intersections=False,
                agent_position=(25, 30),
            )
        ),
    )

    map_gen = MapGen(map_config)
    game_map = map_gen.build()

    # Visualize
    visualize_map(game_map.grid, "Mixed Corridor Types")
    plt.show()

    return game_map


def test_spiral_approximation():
    """Create a spiral-like pattern using angled corridors."""
    print("\n=== Spiral Approximation ===")

    corridors = []
    center_y, center_x = 30, 30

    # Create spiral by gradually changing angle and distance
    num_segments = 16
    for i in range(num_segments):
        angle = i * 45  # Rotate 45 degrees each segment
        radius = 5 + i * 2  # Increase radius

        # Calculate start point for this segment
        start_angle_rad = math.radians(angle - 45)
        start_y = int(center_y - math.sin(start_angle_rad) * (radius - 2))
        start_x = int(center_x + math.cos(start_angle_rad) * (radius - 2))

        corridors.append(
            corridor(
                center=(start_y, start_x),
                angle=angle,
                length=int(radius * 0.7),
                thickness=2,
            )
        )

    # Create the map
    map_config = MapGen.Config(
        width=60,
        height=60,
        border_width=2,
        root=AngledCorridorBuilder.factory(
            params=AngledCorridorBuilderParams(
                corridors=corridors,
                objects={"altar": 4},
                place_at_ends=True,
                agent_position=(center_y, center_x),
            )
        ),
    )

    map_gen = MapGen(map_config)
    game_map = map_gen.build()

    # Visualize
    visualize_map(game_map.grid, "Spiral Approximation")
    plt.show()

    return game_map


def main():
    """Run all radial corridor tests."""
    print("=" * 60)
    print("Angled Corridor Builder Testing")
    print("=" * 60)

    print("\nThe enhanced API now supports:")
    print("1. Corridors at ANY angle (not just horizontal/vertical)")
    print("2. Radial patterns like radial_mini.map")
    print("3. Star patterns (bidirectional corridors)")
    print("4. Mixing traditional and angled corridors")
    print("5. Complex patterns like spirals")

    print("\nBasic usage:")
    print("  corridor(center=(y,x), angle=45, length=20, thickness=3)")
    print("\nConvenience functions:")
    print("  horizontal(y=10)  # Same as angle=0")
    print("  vertical(x=10)    # Same as angle=270")
    print("  radial_corridors(center, num_spokes=8, length=10)")

    # Test radial mini pattern
    test_radial_mini()

    # Test custom angles
    test_custom_angles()

    # Test star pattern
    test_star_pattern()

    # Test mixed corridors
    test_mixed_corridors()

    # Test spiral approximation
    test_spiral_approximation()

    print("\n" + "=" * 60)
    print("Summary: Now you can create corridors at any angle!")
    print("This makes radial patterns trivial to create.")
    print("=" * 60)


if __name__ == "__main__":
    main()
