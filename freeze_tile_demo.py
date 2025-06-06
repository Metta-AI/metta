#!/usr/bin/env python3

import numpy as np

from mettagrid import mettagrid_c


def test_freeze_tile():
    """
    Test the freeze tile functionality.

    This creates a simple 5x5 grid with:
    - An agent at position (1, 1)
    - A freeze tile at position (2, 1)
    - Demonstrates freeze tile behavior
    """

    # Create a simple 5x5 grid
    grid = [
        ["empty", "empty", "empty", "empty", "empty"],
        ["empty", "agent.test", "empty", "empty", "empty"],
        ["empty", "freeze_tile", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty"],
        ["empty", "empty", "empty", "empty", "empty"],
    ]

    # Configuration for the environment
    config = {
        "max_steps": 100,
        "groups": {"test": {"id": 0, "props": {"rewards": {}}}},
        "agent": {"hp": 10, "freeze_duration": 3, "default_item_max": 10},
        "objects": {"freeze_tile": {"hp": 1}},
        "actions": [{"_target_": "mettagrid.actions.move.Move"}, {"_target_": "mettagrid.actions.noop.Noop"}],
    }

    # Create the environment
    env = mettagrid_c.MettaGrid(config, grid)

    # Set up observation and reward buffers
    obs_buffer = np.zeros((1, 5, 5, 10), dtype=np.uint8)
    terminals = np.zeros(1, dtype=bool)
    truncations = np.zeros(1, dtype=bool)
    rewards = np.zeros(1, dtype=np.float32)

    env.set_buffers(obs_buffer, terminals, truncations, rewards)

    # Reset the environment
    obs, info = env.reset()

    print("=== Freeze Tile Demo ===")
    print("Initial state:")
    print(f"Agent at position: ({env.grid_objects()[1]['r']}, {env.grid_objects()[1]['c']})")

    # Move agent down to step on freeze tile (action 0 = move, arg 0 = forward)
    print("\nStep 1: Moving agent down onto freeze tile...")
    actions = np.array([[0, 0]], dtype=np.int32)  # Move forward (down)
    obs, rewards, terminals, truncations, info = env.step(actions)

    agent_obj = None
    for obj_id, obj in env.grid_objects().items():
        if "agent_id" in obj:
            agent_obj = obj
            break

    if agent_obj:
        print(f"Agent now at position: ({agent_obj['r']}, {agent_obj['c']})")
        print(f"Agent on freeze tile: {agent_obj.get('on_freeze_tile', False)}")
        print(f"Freeze tile direction: {agent_obj.get('freeze_tile_direction', 'N/A')}")

    # Next step should force movement in the same direction
    print("\nStep 2: Agent should be forced to continue moving down...")
    actions = np.array([[0, 1]], dtype=np.int32)  # Try to move backward, but should be forced forward
    obs, rewards, terminals, truncations, info = env.step(actions)

    for obj_id, obj in env.grid_objects().items():
        if "agent_id" in obj:
            agent_obj = obj
            break

    if agent_obj:
        print(f"Agent now at position: ({agent_obj['r']}, {agent_obj['c']})")
        print(f"Agent on freeze tile: {agent_obj.get('on_freeze_tile', False)}")

    print("\n=== Demo completed ===")
    print("The freeze tile should have forced the agent to continue moving in the same direction!")


if __name__ == "__main__":
    test_freeze_tile()
