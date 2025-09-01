#!/usr/bin/env python3
"""Test script to verify corridor generation in navigation suite."""

import numpy as np

from experiments.evals.navigation_with_corridors import (
    make_corridors_env,
    make_grid_maze_env,
    make_navigation_eval_suite,
    make_radial_large_env,
    make_radial_mini_env,
    make_radial_small_env,
)


def print_map(grid: np.ndarray, name: str):
    """Print a simple ASCII representation of the map."""
    print(f"\n{'=' * 40}")
    print(f"{name} ({grid.shape[0]}x{grid.shape[1]})")
    print("=" * 40)

    # Convert to simple ASCII
    ascii_map = []
    for row in grid:
        line = ""
        for cell in row:
            if cell == "wall":
                line += "#"
            elif cell == "empty":
                line += "."
            elif cell == "agent":
                line += "A"
            elif cell == "altar":
                line += "*"
            else:
                line += "?"
        ascii_map.append(line)

    # Print first 20 rows for visibility
    for line in ascii_map[:20]:
        if len(line) > 80:
            print(line[:80] + "...")
        else:
            print(line)

    if len(ascii_map) > 20:
        print(f"... ({len(ascii_map) - 20} more rows)")


def test_individual_envs():
    """Test individual corridor environment generators."""

    print("\n" + "=" * 60)
    print("TESTING INDIVIDUAL CORRIDOR ENVIRONMENTS")
    print("=" * 60)

    # Test corridors pattern
    print("\n1. Testing corridors.map pattern...")
    env_config = make_corridors_env()
    print(f"   Max steps: {env_config.game.max_steps}")
    print(
        f"   Map size: {env_config.game.map_builder.instance_map.width}x{env_config.game.map_builder.instance_map.height}"
    )

    # Test radial patterns
    print("\n2. Testing radial_mini pattern...")
    env_config = make_radial_mini_env()
    print(f"   Max steps: {env_config.game.max_steps}")
    print(
        f"   Map size: {env_config.game.map_builder.instance_map.width}x{env_config.game.map_builder.instance_map.height}"
    )

    print("\n3. Testing radial_small pattern...")
    env_config = make_radial_small_env()
    print(f"   Max steps: {env_config.game.max_steps}")
    print(
        f"   Map size: {env_config.game.map_builder.instance_map.width}x{env_config.game.map_builder.instance_map.height}"
    )

    print("\n4. Testing radial_large pattern...")
    env_config = make_radial_large_env()
    print(f"   Max steps: {env_config.game.max_steps}")
    print(
        f"   Map size: {env_config.game.map_builder.instance_map.width}x{env_config.game.map_builder.instance_map.height}"
    )

    print("\n5. Testing grid_maze pattern...")
    env_config = make_grid_maze_env()
    print(f"   Max steps: {env_config.game.max_steps}")
    print(
        f"   Map size: {env_config.game.map_builder.instance_map.width}x{env_config.game.map_builder.instance_map.height}"
    )


def test_full_suite():
    """Test the full navigation evaluation suite."""

    print("\n" + "=" * 60)
    print("TESTING FULL NAVIGATION SUITE")
    print("=" * 60)

    suite = make_navigation_eval_suite()

    print(f"\nTotal environments in suite: {len(suite)}")
    print("\nEnvironments using procedural generation:")

    procedural_envs = [
        "corridors",
        "radial_mini",
        "radial_small",
        "radial_large",
        "grid_maze",
    ]
    for sim_config in suite:
        if sim_config.name in procedural_envs:
            print(
                f"  ✓ {sim_config.name:20s} (max_steps: {sim_config.env.game.max_steps})"
            )

    print("\nEnvironments using ASCII maps:")
    for sim_config in suite:
        if (
            sim_config.name not in procedural_envs
            and sim_config.name != "emptyspace_sparse"
        ):
            print(
                f"  - {sim_config.name:20s} (max_steps: {sim_config.env.game.max_steps})"
            )

    print("\nEnvironments using other generation:")
    for sim_config in suite:
        if sim_config.name == "emptyspace_sparse":
            print(f"  - {sim_config.name:20s} (MeanDistance)")


def main():
    """Run all tests."""
    print("=" * 60)
    print("CORRIDOR GENERATION IN NAVIGATION SUITE")
    print("=" * 60)

    # Test individual environment generators
    test_individual_envs()

    # Test the full suite
    test_full_suite()

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
