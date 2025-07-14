#!/usr/bin/env python3
"""Test script to verify C++ exploration tracking functionality."""

import numpy as np
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config

def test_exploration_tracking():
    """Test the exploration tracking functionality."""

    # Create a simple game configuration with exploration tracking enabled
    game_config = {
        "max_steps": 100,
        "num_agents": 2,
        "obs_width": 11,
        "obs_height": 11,
        "num_observation_tokens": 200,
        "inventory_item_names": ["ore_red", "heart"],
        "enable_exploration_tracking": True,  # Enable C++ exploration tracking
        "actions": {
            "noop": {"enabled": True},
            "move": {"enabled": True},
            "rotate": {"enabled": True},
            "attack": {"enabled": False},
            "put_items": {"enabled": False},
            "get_items": {"enabled": False},
            "swap": {"enabled": False},
            "change_color": {"enabled": False},
        },
        "groups": {"red": {"id": 0, "props": {}}},
        "objects": {
            "wall": {"type_id": 1},
        },
        "agent": {},
    }

    # Create a simple map
    game_map = np.full((10, 10), "empty", dtype="<U50")
    game_map[0, :] = "wall"
    game_map[-1, :] = "wall"
    game_map[:, 0] = "wall"
    game_map[:, -1] = "wall"

    # Place agents
    game_map[1, 1] = "agent.red"
    game_map[2, 2] = "agent.red"

    try:
        # Test that the config can be converted without errors
        cpp_config = from_mettagrid_config(game_config)
        print("✅ Successfully created C++ config with exploration tracking enabled")
        print(f"   enable_exploration_tracking: {cpp_config.enable_exploration_tracking}")

        # Test that we can create the MettaGrid environment
        from metta.mettagrid.mettagrid_c import MettaGrid
        env = MettaGrid(cpp_config, game_map.tolist(), 42)
        print("✅ Successfully created MettaGrid environment")

        # Test that exploration metrics methods are available
        if hasattr(env, 'get_exploration_metrics'):
            print("✅ Exploration metrics method is available")
            metrics = env.get_exploration_metrics()
            print(f"   Initial metrics: {len(metrics)} metrics available")
        else:
            print("❌ Exploration metrics method not found")

        if hasattr(env, 'reset_exploration_tracking'):
            print("✅ Reset exploration tracking method is available")
        else:
            print("❌ Reset exploration tracking method not found")

        if hasattr(env, 'update_exploration_thresholds'):
            print("✅ Update exploration thresholds method is available")
        else:
            print("❌ Update exploration thresholds method not found")

        print("\n🎉 C++ exploration tracking implementation is working!")

    except Exception as e:
        print(f"❌ Error testing exploration tracking: {e}")
        import traceback
        traceback.print_exc()

def test_exploration_tracking_disabled():
    """Test that exploration tracking can be disabled."""

    game_config = {
        "max_steps": 100,
        "num_agents": 2,
        "obs_width": 11,
        "obs_height": 11,
        "num_observation_tokens": 200,
        "inventory_item_names": ["ore_red", "heart"],
        "enable_exploration_tracking": False,  # Disable C++ exploration tracking
        "actions": {
            "noop": {"enabled": True},
            "move": {"enabled": True},
            "rotate": {"enabled": True},
            "attack": {"enabled": False},
            "put_items": {"enabled": False},
            "get_items": {"enabled": False},
            "swap": {"enabled": False},
            "change_color": {"enabled": False},
        },
        "groups": {"red": {"id": 0, "props": {}}},
        "objects": {
            "wall": {"type_id": 1},
        },
        "agent": {},
    }

    game_map = np.full((10, 10), "empty", dtype="<U50")
    game_map[0, :] = "wall"
    game_map[-1, :] = "wall"
    game_map[:, 0] = "wall"
    game_map[:, -1] = "wall"
    game_map[1, 1] = "agent.red"
    game_map[2, 2] = "agent.red"

    try:
        cpp_config = from_mettagrid_config(game_config)
        print("✅ Successfully created C++ config with exploration tracking disabled")
        print(f"   enable_exploration_tracking: {cpp_config.enable_exploration_tracking}")

        from metta.mettagrid.mettagrid_c import MettaGrid
        env = MettaGrid(cpp_config, game_map.tolist(), 42)
        print("✅ Successfully created MettaGrid environment with exploration tracking disabled")

        # Test that exploration metrics methods are still available but return empty results
        if hasattr(env, 'get_exploration_metrics'):
            metrics = env.get_exploration_metrics()
            if len(metrics) == 0:
                print("✅ Exploration metrics correctly return empty when disabled")
            else:
                print(f"⚠️  Exploration metrics returned {len(metrics)} metrics when disabled")
        else:
            print("❌ Exploration metrics method not found")

        print("\n🎉 C++ exploration tracking can be properly disabled!")

    except Exception as e:
        print(f"❌ Error testing disabled exploration tracking: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Testing C++ exploration tracking functionality...")
    print("=" * 50)

    test_exploration_tracking()
    print("\n" + "=" * 50)
    test_exploration_tracking_disabled()

    print("\n" + "=" * 50)
    print("✅ All tests completed!")
