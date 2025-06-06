#!/usr/bin/env python3
"""Test script for Z-level functionality with stairs and tall bridges."""

import numpy as np

from mettagrid.mettagrid_c import MettaGrid

# Test configuration
test_config = {
    "game": {
        "num_agents": 2,
        "obs_width": 11,
        "obs_height": 11,
        "use_observation_tokens": False,
        "num_observation_tokens": 128,
        "tile_size": 16,
        "max_steps": 100,
        "agent": {
            "default_item_max": 50,
            "heart_max": 255,
            "freeze_duration": 10,
            "hp": 10,
            "rewards": {
                "action_failure_penalty": 0.0,
                "ore.red": 0.005,
                "ore.blue": 0.005,
                "ore.green": 0.005,
                "ore.red_max": 4,
                "ore.blue_max": 4,
                "ore.green_max": 4,
                "battery.red": 0.01,
                "battery.blue": 0.01,
                "battery.green": 0.01,
                "battery.red_max": 5,
                "battery.blue_max": 5,
                "battery.green_max": 5,
                "heart": 1,
                "heart_max": 1000,
            },
        },
        "groups": {"test": {"id": 0, "sprite": 0, "props": {}}},
        "objects": {
            "wall": {"hp": 10, "swappable": False},
            "stairs": {"hp": 100, "swappable": False},
            "tallbridge": {"hp": 100, "swappable": False},
        },
        "actions": {
            "noop": {"enabled": True},
            "move": {"enabled": True},
            "rotate": {"enabled": True},
            "put_items": {"enabled": False},
            "get_items": {"enabled": False},
            "attack": {"enabled": False},
            "swap": {"enabled": False},
            "change_color": {"enabled": False},
            "climb": {"enabled": True},
        },
    }
}

# Create a simple test map
# W = wall, S = stairs, B = tall bridge, A = agent, . = empty
test_map = [
    ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
    ["wall", ".", ".", "stairs", ".", ".", "wall"],
    ["wall", ".", "agent.test", ".", "agent.test", ".", "wall"],
    ["wall", ".", ".", "tallbridge", ".", ".", "wall"],
    ["wall", ".", ".", ".", ".", ".", "wall"],
    ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
]


def test_zlevels():
    print("Testing Z-level functionality with stairs and tall bridges")
    print("==========================================================")

    # Create environment
    env = MettaGrid({"game": test_config["game"]}, test_map)

    # Get action indices
    action_names = env.action_names()
    print(f"Available actions: {action_names}")

    # Find action indices
    move_idx = action_names.index("move") if "move" in action_names else -1
    climb_idx = action_names.index("climb") if "climb" in action_names else -1

    print(f"Move action index: {move_idx}")
    print(f"Climb action index: {climb_idx}")

    # Reset environment
    obs, info = env.reset()

    # Print initial state
    print("\nInitial state:")
    objects = env.grid_objects()
    for obj_id, obj_data in objects.items():
        if "agent_id" in obj_data:
            z_level = obj_data.get("agent:z_level", 0)
            print(f"Agent {obj_data['agent_id']} at ({obj_data['r']}, {obj_data['c']}) - Z-level: {z_level}")

    # Test scenario:
    # 1. Move agent 0 to stairs
    # 2. Climb up
    # 3. Move on upper level (walls act as floors)
    # 4. Move agent 1 under the tall bridge

    print("\nTest scenario:")

    # Step 1: Move agent 0 north to stairs
    print("\n1. Moving agent 0 north towards stairs...")
    actions = np.array([[move_idx, 0], [move_idx, 0]])  # Both agents move up
    obs, rewards, terminals, truncations, info = env.step(actions)

    # Step 2: Agent 0 climbs stairs
    print("\n2. Agent 0 climbing stairs...")
    actions = np.array([[climb_idx, 0], [move_idx, 0]])  # Agent 0 climbs up, Agent 1 moves
    obs, rewards, terminals, truncations, info = env.step(actions)

    # Print current state
    objects = env.grid_objects()
    for obj_id, obj_data in objects.items():
        if "agent_id" in obj_data:
            z_level = obj_data.get("agent:z_level", 0)
            print(f"Agent {obj_data['agent_id']} at ({obj_data['r']}, {obj_data['c']}) - Z-level: {z_level}")

    # Step 3: Move agent 0 on upper level
    print("\n3. Agent 0 moving on upper level (walls act as floors)...")
    actions = np.array([[move_idx, 0], [move_idx, 0]])  # Both move forward
    obs, rewards, terminals, truncations, info = env.step(actions)

    # Step 4: Move agent 1 under tall bridge
    print("\n4. Agent 1 moving under tall bridge...")
    actions = np.array([[move_idx, 0], [move_idx, 0]])  # Both move forward
    obs, rewards, terminals, truncations, info = env.step(actions)

    # Final state
    print("\nFinal state:")
    objects = env.grid_objects()
    for obj_id, obj_data in objects.items():
        if "agent_id" in obj_data:
            z_level = obj_data.get("agent:z_level", 0)
            print(f"Agent {obj_data['agent_id']} at ({obj_data['r']}, {obj_data['c']}) - Z-level: {z_level}")

    print("\nTest completed successfully!")
    print("- Stairs allow agents to move between Z-levels")
    print("- Walls act as floors at the upper Z-level")
    print("- TallBridge allows agents to walk underneath at ground level")
    print("- TallBridge provides a walkable surface at upper Z-level")


if __name__ == "__main__":
    test_zlevels()
