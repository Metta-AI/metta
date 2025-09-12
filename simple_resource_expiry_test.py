#!/usr/bin/env python3
"""
Simple test script for stochastic resource expiry.
This script focuses on basic functionality testing.
"""

import os
import sys

import numpy as np

# Add the mettagrid module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mettagrid", "src"))

from metta.mettagrid.mettagrid_c import MettaGrid
from metta.mettagrid.mettagrid_config import AgentConfig, ConverterConfig, GameConfig


def test_basic_resource_expiry():
    """Test basic resource expiry functionality."""
    print("Testing basic resource expiry...")

    # Create a simple agent config with resource loss
    agent_config = AgentConfig(
        type_id=1,
        type_name="agent.player",
        initial_inventory={"wood": 5, "stone": 3},
        resource_loss_prob={"wood": 0.2, "stone": 0.1},  # 20% and 10% loss rates
        group_id=0,
        group_reward_pct=1.0,
        resource_rewards={"wood": 1.0, "stone": 2.0},
    )

    # Create a simple converter config with resource loss
    mine_config = ConverterConfig(
        type_id=2,
        type_name="mine",
        input_resources={},
        output_resources={"stone": 1},
        max_output=10,
        max_conversions=-1,
        conversion_ticks=2,
        cooldown=1,
        initial_resource_count=2,
        resource_loss_prob={"stone": 0.15},  # 15% loss rate
    )

    # Create game config
    game_config = GameConfig(
        num_agents=1,
        max_steps=50,
        inventory_item_names=["wood", "stone"],
        objects={"agent.player": agent_config, "mine": mine_config, "wall": {"type_id": 3, "type_name": "wall"}},
        actions={"noop": {"type_id": 1, "type_name": "noop"}},
    )

    # Create simple map
    map_data = [["wall", "wall", "wall"], ["wall", "agent.player", "mine"], ["wall", "wall", "wall"]]

    # Create environment
    env = MettaGrid(game_config, map_data, 42)

    # Get initial state
    obs, rewards, terminals, truncations, infos = env.reset()

    print(f"Initial agent inventory: {infos[0].get('inventory', {})}")

    # Get initial converter state
    grid_objects = env.grid_objects()
    print(f"Initial grid objects: {grid_objects}")

    # Run simulation for 20 steps
    for step in range(20):
        # Agent does nothing
        actions = np.array([1], dtype=np.int32)  # noop action

        obs, rewards, terminals, truncations, infos = env.step(actions)

        # Print every 5 steps
        if step % 5 == 0:
            inventory = infos[0].get("inventory", {})
            print(f"Step {step}: Agent inventory = {inventory}")

            # Check converter state
            grid_objects = env.grid_objects()
            if "mine" in grid_objects and grid_objects["mine"]:
                mine_inventory = grid_objects["mine"][0].get("inventory", {})
                print(f"Step {step}: Mine inventory = {mine_inventory}")

        if terminals[0] or truncations[0]:
            break

    print(f"Final agent inventory: {infos[0].get('inventory', {})}")

    # Check final converter state
    grid_objects = env.grid_objects()
    if "mine" in grid_objects and grid_objects["mine"]:
        mine_inventory = grid_objects["mine"][0].get("inventory", {})
        print(f"Final mine inventory: {mine_inventory}")

    print("Basic resource expiry test completed!")


def test_no_loss_rates():
    """Test that resources don't expire when loss rates are zero."""
    print("\nTesting zero loss rates...")

    # Create agent config with no resource loss
    agent_config = AgentConfig(
        type_id=1,
        type_name="agent.player",
        initial_inventory={"wood": 5, "stone": 3},
        resource_loss_prob={},  # No loss rates
        group_id=0,
        group_reward_pct=1.0,
        resource_rewards={"wood": 1.0, "stone": 2.0},
    )

    # Create game config
    game_config = GameConfig(
        num_agents=1,
        max_steps=30,
        inventory_item_names=["wood", "stone"],
        objects={"agent.player": agent_config, "wall": {"type_id": 2, "type_name": "wall"}},
        actions={"noop": {"type_id": 1, "type_name": "noop"}},
    )

    # Create simple map
    map_data = [["wall", "wall", "wall"], ["wall", "agent.player", "wall"], ["wall", "wall", "wall"]]

    # Create environment
    env = MettaGrid(game_config, map_data, 42)

    # Get initial state
    obs, rewards, terminals, truncations, infos = env.reset()

    initial_inventory = infos[0].get("inventory", {})
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
    else:
        print("✗ Resources expired when they shouldn't have")

    print("Zero loss rates test completed!")


def main():
    """Run the simple tests."""
    print("Simple Resource Expiry Test")
    print("=" * 40)

    try:
        test_basic_resource_expiry()
        test_no_loss_rates()
        print("\n" + "=" * 40)
        print("All simple tests completed!")

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
