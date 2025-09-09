#!/usr/bin/env python3
"""
Example demonstrating how to use the new inventory reduction event system.

This example shows how to:
1. Create a MettaGrid environment with agents that have resource loss probabilities
2. Schedule inventory reduction events for specific agents
3. Observe the stochastic resource loss over time
"""

import numpy as np

from metta.mettagrid.mettagrid_c import MettaGrid
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config


def create_example_config():
    """Create a simple game configuration for testing inventory reduction."""

    # Create game configuration dictionary
    game_config = {
        "max_steps": 100,
        "num_agents": 1,
        "obs_width": 5,
        "obs_height": 5,
        "num_observation_tokens": 50,
        "resource_names": ["heart", "battery_blue"],  # Two items for testing
        "actions": {
            "noop": {"enabled": True},
            "move": {"enabled": True},
            "rotate": {"enabled": True},
            "attack": {"enabled": False},  # Disable attack action
            "put_items": {"enabled": False},
            "get_items": {"enabled": False},
            "swap": {"enabled": False},
            "change_color": {"enabled": False},
        },
        "objects": {
            "wall": {"type_id": 1},
        },
    }

    return game_config


def create_simple_map():
    """Create a simple 5x5 map with one agent."""
    # Create a 5x5 grid with walls around the edges and one agent in the center
    map_data = [
        ["wall", "wall", "wall", "wall", "wall"],  # Top wall
        ["wall", "wall", "wall", "wall", "wall"],  # Left wall, empty, empty, empty, right wall
        ["wall", "wall", "agent.player", "wall", "wall"],  # Left wall, empty, agent, empty, right wall
        ["wall", "wall", "wall", "wall", "wall"],  # Left wall, empty, empty, empty, right wall
        ["wall", "wall", "wall", "wall", "wall"],  # Bottom wall
    ]
    return map_data


def main():
    """Main example function."""
    print("Creating MettaGrid environment with inventory reduction events...")

    # Create configuration
    game_config_dict = create_example_config()
    map_data = create_simple_map()

    # Convert to proper config format
    game_config = from_mettagrid_config(game_config_dict)

    # Create environment
    env = MettaGrid(game_config, map_data, 42)

    print(f"Environment created with {env.num_agents} agents")

    # Get initial state
    observations, terminals, truncations, rewards = env.reset()

    print("Initial agent inventory:")
    grid_objects = env.grid_objects()
    for obj_id, obj_data in grid_objects.items():
        if obj_data.get("type_name") == "test_agent":
            print(f"  Agent {obj_id}: {obj_data['inventory']}")

    # Schedule inventory reduction events
    print("\nScheduling inventory reduction events...")

    # Schedule events for agent 0 at different timesteps
    env.schedule_inventory_reduction(agent_id=0, delay=5)  # Event at step 5
    env.schedule_inventory_reduction(agent_id=0, delay=10)  # Event at step 10
    env.schedule_inventory_reduction(agent_id=0, delay=15)  # Event at step 15
    env.schedule_inventory_reduction(agent_id=0, delay=20)  # Event at step 20

    print("Scheduled 4 inventory reduction events for agent 0")
    print("Running simulation for 25 steps...")

    # Run simulation
    for step in range(25):
        # Take no-op actions (since we have no actions defined)
        actions = np.array([0], dtype=np.uint8)

        observations, terminals, truncations, rewards = env.step(actions)

        # Check inventory every 5 steps
        if step % 5 == 0:
            grid_objects = env.grid_objects()
            for obj_id, obj_data in grid_objects.items():
                if obj_data.get("type_name") == "test_agent":
                    print(f"  Step {step}: Agent {obj_id} inventory: {obj_data['inventory']}")

    print("\nSimulation complete!")
    print("Final agent inventory:")
    grid_objects = env.grid_objects()
    for obj_id, obj_data in grid_objects.items():
        if obj_data.get("type_name") == "test_agent":
            print(f"  Agent {obj_id}: {obj_data['inventory']}")


if __name__ == "__main__":
    main()
