#!/usr/bin/env python3
"""
Simple test for stochastic resource expiry using the correct configuration format.
"""

import os
import sys

import numpy as np

# Add the mettagrid module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mettagrid", "src"))

from metta.mettagrid.mettagrid_c import MettaGrid
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config


def test_agent_resource_expiry():
    """Test agent resource expiry with correct configuration format."""
    print("Testing Agent Resource Expiry...")

    # Create configuration using the correct format
    game_config_dict = {
        "max_steps": 50,
        "num_agents": 1,
        "obs_width": 3,
        "obs_height": 3,
        "num_observation_tokens": 20,
        "inventory_item_names": ["wood", "stone", "food"],
        "actions": {
            "noop": {"enabled": True},
        },
        "groups": {
            "player": {
                "id": 0,
                "sprite": 0,
                "props": {},
            }
        },
        "objects": {
            "agent.player": {
                "type_id": 1,
                "type_name": "agent.player",
                "group_id": 0,
                "group_name": "player",
                "initial_inventory": {"wood": 5, "stone": 3, "food": 2},
                "resource_loss_prob": {"wood": 0.1, "stone": 0.05, "food": 0.2},
                "resource_rewards": {"wood": 1.0, "stone": 2.0, "food": 0.5},
            },
            "mine": {
                "type_id": 2,
                "type_name": "mine",
                "input_resources": {},
                "output_resources": {"stone": 1},
                "max_output": 10,
                "max_conversions": -1,
                "conversion_ticks": 2,
                "cooldown": 1,
                "initial_resource_count": 3,
                "resource_loss_prob": {"stone": 0.15},
            },
            "wall": {"type_id": 3, "type_name": "wall"},
        },
    }

    # Convert to C++ config
    game_config, map_data = from_mettagrid_config(game_config_dict)

    # Create simple map
    map_data = [["wall", "wall", "wall"], ["wall", "agent.player", "mine"], ["wall", "wall", "wall"]]

    # Create environment
    env = MettaGrid(game_config, map_data, 42)

    # Get initial state
    obs, rewards, terminals, truncations, infos = env.reset()

    print(f"Initial agent inventory: {infos[0].get('inventory', {})}")

    # Track inventory changes
    initial_inventory = infos[0].get("inventory", {}).copy()

    # Run simulation for 30 steps
    for step in range(30):
        # Agent does nothing
        actions = np.array([1], dtype=np.int32)  # noop action

        obs, rewards, terminals, truncations, infos = env.step(actions)

        # Print every 10 steps
        if step % 10 == 0:
            inventory = infos[0].get("inventory", {})
            print(f"Step {step}: Agent inventory = {inventory}")

        if terminals[0] or truncations[0]:
            break

    final_inventory = infos[0].get("inventory", {})
    print(f"Final agent inventory: {final_inventory}")

    # Check if resources were lost
    resources_lost = False
    for resource, initial_count in initial_inventory.items():
        final_count = final_inventory.get(resource, 0)
        if final_count < initial_count:
            print(f"✓ {resource}: {initial_count} -> {final_count} (lost {initial_count - final_count})")
            resources_lost = True
        else:
            print(f"  {resource}: {initial_count} -> {final_count} (no loss)")

    if resources_lost:
        print("✓ Agent resource expiry test PASSED")
    else:
        print("✗ Agent resource expiry test FAILED - no resources were lost")

    return resources_lost


def test_converter_resource_expiry():
    """Test converter resource expiry."""
    print("\nTesting Converter Resource Expiry...")

    # Create configuration
    game_config_dict = {
        "max_steps": 50,
        "num_agents": 1,
        "obs_width": 3,
        "obs_height": 3,
        "num_observation_tokens": 20,
        "inventory_item_names": ["wood", "stone"],
        "actions": {
            "noop": {"enabled": True},
        },
        "groups": {
            "player": {
                "id": 0,
                "sprite": 0,
                "props": {},
            }
        },
        "objects": {
            "agent.player": {
                "type_id": 1,
                "type_name": "agent.player",
                "group_id": 0,
                "group_name": "player",
                "initial_inventory": {},
                "resource_loss_prob": {},
            },
            "mine": {
                "type_id": 2,
                "type_name": "mine",
                "input_resources": {},
                "output_resources": {"stone": 1},
                "max_output": 10,
                "max_conversions": -1,
                "conversion_ticks": 2,
                "cooldown": 1,
                "initial_resource_count": 5,  # Start with 5 stone
                "resource_loss_prob": {"stone": 0.2},  # 20% loss rate
            },
            "wall": {"type_id": 3, "type_name": "wall"},
        },
    }

    # Convert to C++ config
    game_config, map_data = from_mettagrid_config(game_config_dict)

    # Create simple map
    map_data = [["wall", "wall", "wall"], ["wall", "agent.player", "mine"], ["wall", "wall", "wall"]]

    # Create environment
    env = MettaGrid(game_config, map_data, 42)

    # Get initial state
    obs, rewards, terminals, truncations, infos = env.reset()

    # Get initial converter state
    grid_objects = env.grid_objects()
    print(f"Initial grid objects: {grid_objects}")

    # Find mine inventory
    initial_mine_inventory = {}
    if "mine" in grid_objects and grid_objects["mine"]:
        initial_mine_inventory = grid_objects["mine"][0].get("inventory", {}).copy()
        print(f"Initial mine inventory: {initial_mine_inventory}")

    # Run simulation for 30 steps
    for step in range(30):
        # Agent does nothing
        actions = np.array([1], dtype=np.int32)  # noop action

        obs, rewards, terminals, truncations, infos = env.step(actions)

        # Print every 10 steps
        if step % 10 == 0:
            grid_objects = env.grid_objects()
            if "mine" in grid_objects and grid_objects["mine"]:
                mine_inventory = grid_objects["mine"][0].get("inventory", {})
                print(f"Step {step}: Mine inventory = {mine_inventory}")

        if terminals[0] or truncations[0]:
            break

    # Get final converter state
    grid_objects = env.grid_objects()
    final_mine_inventory = {}
    if "mine" in grid_objects and grid_objects["mine"]:
        final_mine_inventory = grid_objects["mine"][0].get("inventory", {})
        print(f"Final mine inventory: {final_mine_inventory}")

    # Check if resources were lost
    resources_lost = False
    for resource, initial_count in initial_mine_inventory.items():
        final_count = final_mine_inventory.get(resource, 0)
        if final_count < initial_count:
            print(f"✓ {resource}: {initial_count} -> {final_count} (lost {initial_count - final_count})")
            resources_lost = True
        else:
            print(f"  {resource}: {initial_count} -> {final_count} (no loss)")

    if resources_lost:
        print("✓ Converter resource expiry test PASSED")
    else:
        print("✗ Converter resource expiry test FAILED - no resources were lost")

    return resources_lost


def test_zero_loss_rates():
    """Test that resources don't expire when loss rates are zero."""
    print("\nTesting Zero Loss Rates...")

    # Create configuration with no resource loss
    game_config_dict = {
        "max_steps": 30,
        "num_agents": 1,
        "obs_width": 3,
        "obs_height": 3,
        "num_observation_tokens": 20,
        "inventory_item_names": ["wood", "stone"],
        "actions": {
            "noop": {"enabled": True},
        },
        "groups": {
            "player": {
                "id": 0,
                "sprite": 0,
                "props": {},
            }
        },
        "objects": {
            "agent.player": {
                "type_id": 1,
                "type_name": "agent.player",
                "group_id": 0,
                "group_name": "player",
                "initial_inventory": {"wood": 5, "stone": 3},
                "resource_loss_prob": {},  # No loss rates
                "resource_rewards": {"wood": 1.0, "stone": 2.0},
            },
            "wall": {"type_id": 2, "type_name": "wall"},
        },
    }

    # Convert to C++ config
    game_config, map_data = from_mettagrid_config(game_config_dict)

    # Create simple map
    map_data = [["wall", "wall", "wall"], ["wall", "agent.player", "wall"], ["wall", "wall", "wall"]]

    # Create environment
    env = MettaGrid(game_config, map_data, 42)

    # Get initial state
    obs, rewards, terminals, truncations, infos = env.reset()

    initial_inventory = infos[0].get("inventory", {}).copy()
    print(f"Initial inventory: {initial_inventory}")

    # Run simulation for 20 steps
    for step in range(20):
        actions = np.array([1], dtype=np.int32)  # noop action
        obs, rewards, terminals, truncations, infos = env.step(actions)

        if step % 10 == 0:
            inventory = infos[0].get("inventory", {})
            print(f"Step {step}: {inventory}")

        if terminals[0] or truncations[0]:
            break

    final_inventory = infos[0].get("inventory", {})
    print(f"Final inventory: {final_inventory}")

    # Check that inventory didn't change
    if initial_inventory == final_inventory:
        print("✓ Resources did not expire (as expected)")
        return True
    else:
        print("✗ Resources expired when they shouldn't have")
        return False


def main():
    """Run all tests."""
    print("Resource Expiry Test Suite")
    print("=" * 50)

    try:
        # Test agent resource expiry
        agent_test_passed = test_agent_resource_expiry()

        # Test converter resource expiry
        converter_test_passed = test_converter_resource_expiry()

        # Test zero loss rates
        zero_loss_test_passed = test_zero_loss_rates()

        print("\n" + "=" * 50)
        print("Test Results:")
        print(f"Agent resource expiry: {'PASS' if agent_test_passed else 'FAIL'}")
        print(f"Converter resource expiry: {'PASS' if converter_test_passed else 'FAIL'}")
        print(f"Zero loss rates: {'PASS' if zero_loss_test_passed else 'FAIL'}")

        if agent_test_passed and converter_test_passed and zero_loss_test_passed:
            print("\n🎉 All tests PASSED!")
            return 0
        else:
            print("\n❌ Some tests FAILED!")
            return 1

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
