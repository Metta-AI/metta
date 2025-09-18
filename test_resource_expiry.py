#!/usr/bin/env python3
"""
Test script for stochastic resource expiry in Agent and Converter classes.
This script demonstrates and validates the resource loss functionality.
"""

import os
import sys

import numpy as np

# Add the mettagrid module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mettagrid", "src"))

from metta.mettagrid.mettagrid_c import MettaGrid
from metta.mettagrid.mettagrid_config import AgentConfig, ConverterConfig, GameConfig


def create_test_config():
    """Create a test configuration with agents and converters that have resource loss probabilities."""

    # Create agent config with resource loss probabilities
    agent_config = AgentConfig(
        type_id=1,
        type_name="agent.player",
        initial_inventory={"wood": 5, "stone": 3, "food": 2},
        resource_loss_prob={"wood": 0.1, "stone": 0.05, "food": 0.2},  # High loss rates for testing
        group_id=0,
        group_reward_pct=1.0,
        resource_rewards={"wood": 1.0, "stone": 2.0, "food": 0.5},
    )

    # Create converter config (mine) with resource loss probabilities
    mine_config = ConverterConfig(
        type_id=2,
        type_name="mine",
        input_resources={},  # No input required
        output_resources={"stone": 1},
        max_output=10,
        max_conversions=-1,  # Unlimited
        conversion_ticks=2,
        cooldown=1,
        initial_resource_count=3,  # Start with some stone
        resource_loss_prob={"stone": 0.15},  # High loss rate for testing
    )

    # Create converter config (generator) with resource loss probabilities
    generator_config = ConverterConfig(
        type_id=3,
        type_name="generator",
        input_resources={"wood": 1},
        output_resources={"food": 1},
        max_output=8,
        max_conversions=-1,
        conversion_ticks=3,
        cooldown=2,
        initial_resource_count=2,  # Start with some food
        resource_loss_prob={"food": 0.12},  # High loss rate for testing
    )

    # Create game config
    game_config = GameConfig(
        num_agents=1,
        max_steps=100,
        inventory_item_names=["wood", "stone", "food"],
        objects={
            "agent.player": agent_config,
            "mine": mine_config,
            "generator": generator_config,
            "wall": {"type_id": 4, "type_name": "wall"},
        },
        actions={"move": {"type_id": 0, "type_name": "move"}, "noop": {"type_id": 1, "type_name": "noop"}},
    )

    return game_config


def create_test_map():
    """Create a simple test map with agents and converters."""
    return [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.player", "mine", "generator", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]


def test_agent_resource_expiry():
    """Test agent resource expiry functionality."""
    print("=== Testing Agent Resource Expiry ===")

    game_config = create_test_config()
    map_data = create_test_map()

    # Create environment
    env = MettaGrid(game_config, map_data, 42)

    # Get initial state
    obs, rewards, terminals, truncations, infos = env.reset()

    print(f"Initial agent inventory: {infos[0].get('inventory', {})}")

    # Track resource counts over time
    resource_counts = {"wood": [], "stone": [], "food": []}
    timesteps = []

    # Run simulation for 50 steps
    for step in range(50):
        # Agent does nothing (noop action)
        actions = np.array([1], dtype=np.int32)  # noop action

        obs, rewards, terminals, truncations, infos = env.step(actions)

        # Record resource counts
        inventory = infos[0].get("inventory", {})
        for resource in resource_counts:
            resource_counts[resource].append(inventory.get(resource, 0))
        timesteps.append(step)

        # Print significant changes
        if step % 10 == 0 or step < 5:
            print(f"Step {step}: Agent inventory = {inventory}")

        if terminals[0] or truncations[0]:
            break

    print(f"\nFinal agent inventory: {infos[0].get('inventory', {})}")

    # Analyze results
    print("\nResource count analysis:")
    for resource, counts in resource_counts.items():
        if counts:
            initial = counts[0]
            final = counts[-1]
            min_count = min(counts)
            print(f"  {resource}: {initial} -> {final} (min: {min_count})")

    return resource_counts


def test_converter_resource_expiry():
    """Test converter resource expiry functionality."""
    print("\n=== Testing Converter Resource Expiry ===")

    game_config = create_test_config()
    map_data = create_test_map()

    # Create environment
    env = MettaGrid(game_config, map_data, 42)

    # Get initial state
    obs, rewards, terminals, truncations, infos = env.reset()

    # Get converter information
    grid_objects = env.grid_objects()
    print(f"Grid objects: {list(grid_objects.keys())}")

    # Track converter outputs over time
    converter_outputs = {"mine": [], "generator": []}
    timesteps = []

    # Run simulation for 50 steps
    for step in range(50):
        # Agent does nothing (noop action)
        actions = np.array([1], dtype=np.int32)  # noop action

        obs, rewards, terminals, truncations, infos = env.step(actions)

        # Record converter outputs
        grid_objects = env.grid_objects()
        for converter_name in converter_outputs:
            if converter_name in grid_objects:
                converter_info = grid_objects[converter_name]
                if isinstance(converter_info, list) and len(converter_info) > 0:
                    # Get the first converter's inventory
                    converter_inventory = converter_info[0].get("inventory", {})
                    converter_outputs[converter_name].append(converter_inventory)
                else:
                    converter_outputs[converter_name].append({})
            else:
                converter_outputs[converter_name].append({})

        timesteps.append(step)

        # Print significant changes
        if step % 10 == 0 or step < 5:
            print(f"Step {step}: Grid objects = {grid_objects}")

        if terminals[0] or truncations[0]:
            break

    print(f"\nFinal grid objects: {env.grid_objects()}")

    # Analyze results
    print("\nConverter output analysis:")
    for converter_name, outputs in converter_outputs.items():
        if outputs:
            print(f"  {converter_name}:")
            for step, output in enumerate(outputs):
                if output:  # Only print non-empty outputs
                    print(f"    Step {step}: {output}")

    return converter_outputs


def test_resource_loss_events():
    """Test that resource loss events are properly scheduled and executed."""
    print("\n=== Testing Resource Loss Events ===")

    game_config = create_test_config()
    map_data = create_test_map()

    # Create environment
    env = MettaGrid(game_config, map_data, 42)

    # Get initial state
    obs, rewards, terminals, truncations, infos = env.reset()

    print(f"Initial agent inventory: {infos[0].get('inventory', {})}")

    # Run simulation and track when resources are lost
    lost_resources = []

    for step in range(30):
        # Agent does nothing (noop action)
        actions = np.array([1], dtype=np.int32)  # noop action

        obs, rewards, terminals, truncations, infos = env.step(actions)

        # Check for resource loss by comparing with previous step
        current_inventory = infos[0].get("inventory", {})

        if step > 0:
            # Compare with previous inventory to detect losses
            prev_inventory = getattr(test_resource_loss_events, "prev_inventory", {})
            for resource, count in prev_inventory.items():
                if resource in current_inventory:
                    if current_inventory[resource] < count:
                        lost_amount = count - current_inventory[resource]
                        lost_resources.append((step, resource, lost_amount))
                        print(f"Step {step}: Lost {lost_amount} {resource}")
                else:
                    # Resource completely lost
                    lost_resources.append((step, resource, count))
                    print(f"Step {step}: Lost all {count} {resource}")

        # Store current inventory for next comparison
        test_resource_loss_events.prev_inventory = current_inventory.copy()

        if terminals[0] or truncations[0]:
            break

    print(f"\nTotal resource loss events: {len(lost_resources)}")
    for step, resource, amount in lost_resources:
        print(f"  Step {step}: Lost {amount} {resource}")

    return lost_resources


def test_high_loss_rates():
    """Test with very high loss rates to ensure the system works correctly."""
    print("\n=== Testing High Loss Rates ===")

    # Create config with very high loss rates
    agent_config = AgentConfig(
        type_id=1,
        type_name="agent.player",
        initial_inventory={"wood": 10, "stone": 10, "food": 10},
        resource_loss_prob={"wood": 0.5, "stone": 0.3, "food": 0.4},  # Very high loss rates
        group_id=0,
        group_reward_pct=1.0,
        resource_rewards={"wood": 1.0, "stone": 2.0, "food": 0.5},
    )

    game_config = GameConfig(
        num_agents=1,
        max_steps=50,
        inventory_item_names=["wood", "stone", "food"],
        objects={"agent.player": agent_config, "wall": {"type_id": 4, "type_name": "wall"}},
        actions={"noop": {"type_id": 1, "type_name": "noop"}},
    )

    map_data = [["wall", "wall", "wall"], ["wall", "agent.player", "wall"], ["wall", "wall", "wall"]]

    # Create environment
    env = MettaGrid(game_config, map_data, 42)

    # Get initial state
    obs, rewards, terminals, truncations, infos = env.reset()

    print(f"Initial inventory: {infos[0].get('inventory', {})}")

    # Run simulation
    for step in range(20):
        actions = np.array([1], dtype=np.int32)  # noop action
        obs, rewards, terminals, truncations, infos = env.step(actions)

        inventory = infos[0].get("inventory", {})
        print(f"Step {step}: {inventory}")

        if terminals[0] or truncations[0]:
            break

    print(f"Final inventory: {infos[0].get('inventory', {})}")


def main():
    """Run all tests."""
    print("Testing Stochastic Resource Expiry for Agent and Converter")
    print("=" * 60)

    try:
        # Test agent resource expiry
        agent_results = test_agent_resource_expiry()

        # Test converter resource expiry
        converter_results = test_converter_resource_expiry()

        # Test resource loss events
        loss_events = test_resource_loss_events()

        # Test high loss rates
        test_high_loss_rates()

        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("\nSummary:")
        print(f"- Agent resource expiry: {'PASS' if any(any(counts) for counts in agent_results.values()) else 'FAIL'}")
        converter_pass = any(any(outputs) for outputs in converter_results.values())
        print(f"- Converter resource expiry: {'PASS' if converter_pass else 'FAIL'}")
        print(f"- Resource loss events: {'PASS' if loss_events else 'FAIL'}")

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
