#!/usr/bin/env python3
"""Test that the navigation map truncation bug is fixed."""

from experiments.recipes.navigation import make_env
from metta.map.mapgen import MapGen


def test_navigation_fix():
    print("Testing navigation map fix...")
    print("=" * 60)

    # Test with 4 agents (play mode scenario)
    print("\nTest 1: 4 agents (play mode)")
    print("-" * 40)
    env_config = make_env(num_agents=4)
    map_builder = MapGen(env_config.game.map_builder)

    # Build the map
    full_map = map_builder.build()

    print(f"Full map shape: {full_map.grid.shape}")
    print(f"Instance arrangement: {map_builder.instance_rows}x{map_builder.instance_cols}")
    print(f"Space allocated per instance: {map_builder.width}x{map_builder.height}")

    # Load a sample terrain map to check its size
    from metta.map.terrain_from_numpy import TerrainFromNumpy

    terrain_config = env_config.game.map_builder.instance_map
    terrain_builder = TerrainFromNumpy(terrain_config)

    # Get max dimensions (should use metadata if available)
    max_dims = terrain_builder.get_max_dimensions()
    if max_dims:
        print(f"Maximum terrain dimensions: {max_dims[0]}x{max_dims[1]}")

        # Check if allocated space is sufficient
        if map_builder.width >= max_dims[1] and map_builder.height >= max_dims[0]:
            print("✅ PASS: Allocated space is sufficient for all maps")
        else:
            print(
                f"❌ FAIL: Allocated space ({map_builder.width}x{map_builder.height}) "
                f"is smaller than max terrain ({max_dims[1]}x{max_dims[0]})"
            )

    # Test with 24 agents (training scenario)
    print("\n" + "=" * 60)
    print("Test 2: 24 agents (training mode)")
    print("-" * 40)
    env_config_24 = make_env(num_agents=24)
    map_builder_24 = MapGen(env_config_24.game.map_builder)

    # Build the map
    full_map_24 = map_builder_24.build()

    print(f"Full map shape: {full_map_24.grid.shape}")
    print(f"Instance arrangement: {map_builder_24.instance_rows}x{map_builder_24.instance_cols}")
    print(f"Space allocated per instance: {map_builder_24.width}x{map_builder_24.height}")

    if max_dims:
        # Check if allocated space is sufficient
        if map_builder_24.width >= max_dims[1] and map_builder_24.height >= max_dims[0]:
            print("✅ PASS: Allocated space is sufficient for all maps")
        else:
            print(
                f"❌ FAIL: Allocated space ({map_builder_24.width}x{map_builder_24.height}) "
                f"is smaller than max terrain ({max_dims[1]}x{max_dims[0]})"
            )

    print("\n" + "=" * 60)
    print("Summary:")
    if max_dims:
        print(f"  Max terrain dimensions: {max_dims[0]}x{max_dims[1]}")
        print(f"  4-agent allocation: {map_builder.width}x{map_builder.height}")
        print(f"  24-agent allocation: {map_builder_24.width}x{map_builder_24.height}")

        if (
            map_builder.width >= max_dims[1]
            and map_builder.height >= max_dims[0]
            and map_builder_24.width >= max_dims[1]
            and map_builder_24.height >= max_dims[0]
        ):
            print("\n🎉 SUCCESS: The truncation bug is fixed!")
            print("All terrain maps will fit within allocated space.")
        else:
            print("\n⚠️  WARNING: Some maps may still be truncated.")
    else:
        print("Could not determine max dimensions")


if __name__ == "__main__":
    test_navigation_fix()
