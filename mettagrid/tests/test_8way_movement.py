"""Tests for 8-way movement system."""

import numpy as np

from metta.mettagrid.mettagrid_c import MettaGrid
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config
from metta.mettagrid.mettagrid_env import dtype_actions
from tests.test_utils import make_test_config


def test_8way_movement_all_directions():
    """Test 8-way movement in all eight directions."""
    config = make_test_config(
        num_agents=1,
        map=[
            [".", ".", ".", ".", "."],
            [".", ".", ".", ".", "."],
            [".", ".", "agent.player", ".", "."],
            [".", ".", ".", ".", "."],
            [".", ".", ".", ".", "."],
        ],
        actions={
            "move": {"enabled": False},
            "rotate": {"enabled": False},
            "move_8way": {"enabled": True},
        },
    )

    game_map = config.pop("map")
    env = MettaGrid(from_mettagrid_config(config), game_map, 42)
    env.reset()

    objects = env.grid_objects()
    agent_id = next(id for id, obj in objects.items() if obj["type_id"] == 0)  # type_id 0 is agent
    initial_pos = (objects[agent_id]["r"], objects[agent_id]["c"])
    initial_orientation = objects[agent_id]["orientation"]
    assert initial_pos == (2, 2)

    action_names = env.action_names()
    move_8dir_idx = action_names.index("move_8way")

    # Test all 8 directions
    moves = [
        (0, (1, 2)),  # North
        (1, (0, 3)),  # Northeast (from 1,2)
        (4, (1, 3)),  # South (back to center-ish)
        (6, (1, 2)),  # West
        (3, (2, 3)),  # Southeast
        (7, (1, 2)),  # Northwest
        (2, (1, 3)),  # East
        (5, (2, 2)),  # Southwest (back to start)
    ]

    for direction, expected_pos in moves:
        actions = np.zeros((1, 2), dtype=dtype_actions)
        actions[0] = [move_8dir_idx, direction]
        env.step(actions)

        objects = env.grid_objects()
        actual_pos = (objects[agent_id]["r"], objects[agent_id]["c"])
        assert actual_pos == expected_pos, f"Direction {direction}: expected {expected_pos}, got {actual_pos}"
        
        # Verify orientation hasn't changed (atomic action principle)
        assert objects[agent_id]["orientation"] == initial_orientation


def test_8way_movement_obstacles():
    """Test that 8-way movement respects obstacles."""
    config = make_test_config(
        num_agents=1,
        map=[
            ["wall", "wall", "wall", "wall", "wall"],
            ["wall", ".", ".", ".", "wall"],
            ["wall", ".", "agent.player", ".", "wall"],
            ["wall", ".", ".", ".", "wall"],
            ["wall", "wall", "wall", "wall", "wall"],
        ],
        actions={
            "move": {"enabled": False},
            "rotate": {"enabled": False},
            "move_8way": {"enabled": True},
        },
    )

    game_map = config.pop("map")
    env = MettaGrid(from_mettagrid_config(config), game_map, 42)
    env.reset()

    objects = env.grid_objects()
    agent_id = next(id for id, obj in objects.items() if obj["type_id"] == 0)  # type_id 0 is agent

    action_names = env.action_names()
    move_8dir_idx = action_names.index("move_8way")

    # Test diagonal movements near corners
    actions = np.zeros((1, 2), dtype=dtype_actions)

    # Move to top-left corner area
    actions[0] = [move_8dir_idx, 0]  # North
    env.step(actions)
    actions[0] = [move_8dir_idx, 6]  # West
    env.step(actions)

    objects = env.grid_objects()
    assert (objects[agent_id]["r"], objects[agent_id]["c"]) == (1, 1)

    # Try to move Northwest into wall - should fail
    actions[0] = [move_8dir_idx, 7]  # Northwest
    env.step(actions)
    assert not env.action_success()[0]

    # Position should remain unchanged
    objects = env.grid_objects()
    assert (objects[agent_id]["r"], objects[agent_id]["c"]) == (1, 1)


def test_orientation_unchanged_with_8way():
    """Test that orientation is never changed by 8-way movement."""
    config = make_test_config(
        num_agents=1,
        map=[
            [".", ".", ".", ".", "."],
            [".", "agent.player", ".", ".", "."],
            [".", ".", ".", ".", "."],
        ],
        actions={
            "move": {"enabled": False},
            "rotate": {"enabled": True},  # Keep rotate to test orientation
            "move_8way": {"enabled": True},
        },
    )

    game_map = config.pop("map")
    env = MettaGrid(from_mettagrid_config(config), game_map, 42)
    env.reset()

    objects = env.grid_objects()
    agent_id = next(id for id, obj in objects.items() if obj["type_id"] == 0)  # type_id 0 is agent

    action_names = env.action_names()
    move_8dir_idx = action_names.index("move_8way")
    rotate_idx = action_names.index("rotate")

    # First rotate to face right
    actions = np.zeros((1, 2), dtype=dtype_actions)
    actions[0] = [rotate_idx, 3]  # Face right
    env.step(actions)

    objects = env.grid_objects()
    assert objects[agent_id]["orientation"] == 3  # Right

    # Now move in various directions and verify orientation stays at 3
    for direction in range(8):
        actions[0] = [move_8dir_idx, direction]
        env.step(actions)
        
        objects = env.grid_objects()
        assert objects[agent_id]["orientation"] == 3, f"Orientation changed after moving in direction {direction}"


def test_cardinal_movement_no_orientation_change():
    """Test that the updated cardinal movement doesn't change orientation."""
    config = make_test_config(
        num_agents=1,
        map=[
            [".", ".", ".", ".", "."],
            [".", "agent.player", ".", ".", "."],
            [".", ".", ".", ".", "."],
        ],
        actions={
            "move": {"enabled": False},
            "rotate": {"enabled": True},
            "move_cardinal": {"enabled": True},
        },
    )

    game_map = config.pop("map")
    env = MettaGrid(from_mettagrid_config(config), game_map, 42)
    env.reset()

    objects = env.grid_objects()
    agent_id = next(id for id, obj in objects.items() if obj["type_id"] == 0)  # type_id 0 is agent

    action_names = env.action_names()
    move_cardinal_idx = action_names.index("move_cardinal")
    rotate_idx = action_names.index("rotate")

    # Rotate to face right
    actions = np.zeros((1, 2), dtype=dtype_actions)
    actions[0] = [rotate_idx, 3]  # Face right
    env.step(actions)

    objects = env.grid_objects()
    assert objects[agent_id]["orientation"] == 3

    # Move north with cardinal movement
    actions[0] = [move_cardinal_idx, 0]  # North
    env.step(actions)

    objects = env.grid_objects()
    assert (objects[agent_id]["r"], objects[agent_id]["c"]) == (0, 1)
    # Orientation should still be right (3), not up (0)
    assert objects[agent_id]["orientation"] == 3