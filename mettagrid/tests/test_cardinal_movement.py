"""Tests for cardinal movement system."""

import numpy as np

from metta.mettagrid.mettagrid_c import MettaGrid
from metta.mettagrid.mettagrid_env import dtype_actions
from tests.test_utils import make_test_config


def test_cardinal_movement_basic():
    """Test basic cardinal movement in all four directions."""
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
            "move_cardinal": {"enabled": True},
        },
    )

    env = MettaGrid(config, config["map"], seed=42)
    env.reset()

    # Get initial agent position and orientation
    objects = env.grid_objects()
    agent_id = next(id for id, obj in objects.items() if obj["type_name"] == "agent")
    initial_pos = (objects[agent_id]["r"], objects[agent_id]["c"])
    initial_orientation = objects[agent_id]["orientation"]
    assert initial_pos == (2, 2)

    # Test movement in each cardinal direction
    action_names = env.action_names()
    move_cardinal_idx = action_names.index("move_cardinal")

    # Note: We've disabled move and rotate, so they shouldn't be present
    assert "move" not in action_names
    assert "rotate" not in action_names

    # Move North (Up)
    actions = np.zeros((1, 2), dtype=dtype_actions)
    actions[0] = [move_cardinal_idx, 0]  # 0 = North
    env.step(actions)

    objects = env.grid_objects()
    assert (objects[agent_id]["r"], objects[agent_id]["c"]) == (1, 2)
    assert objects[agent_id]["orientation"] == initial_orientation  # Orientation unchanged

    # Move South (Down)
    actions[0] = [move_cardinal_idx, 1]  # 1 = South
    env.step(actions)

    objects = env.grid_objects()
    assert (objects[agent_id]["r"], objects[agent_id]["c"]) == (2, 2)
    assert objects[agent_id]["orientation"] == initial_orientation  # Orientation unchanged

    # Move West (Left)
    actions[0] = [move_cardinal_idx, 2]  # 2 = West
    env.step(actions)

    objects = env.grid_objects()
    assert (objects[agent_id]["r"], objects[agent_id]["c"]) == (2, 1)
    assert objects[agent_id]["orientation"] == initial_orientation  # Orientation unchanged

    # Move East (Right)
    actions[0] = [move_cardinal_idx, 3]  # 3 = East
    env.step(actions)

    objects = env.grid_objects()
    assert (objects[agent_id]["r"], objects[agent_id]["c"]) == (2, 2)
    assert objects[agent_id]["orientation"] == initial_orientation  # Orientation unchanged


def test_cardinal_movement_obstacles():
    """Test that cardinal movement respects obstacles."""
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
            "move_cardinal": {"enabled": True},
        },
    )

    env = MettaGrid(config, config["map"], seed=42)
    env.reset()

    objects = env.grid_objects()
    agent_id = next(id for id, obj in objects.items() if obj["type_name"] == "agent")

    action_names = env.action_names()
    move_cardinal_idx = action_names.index("move_cardinal")

    # Try to move into walls - should fail
    actions = np.zeros((1, 2), dtype=dtype_actions)

    # North into wall
    actions[0] = [move_cardinal_idx, 0]
    env.step(actions)
    success = env.action_success()
    assert not success[0]

    # Move to corner
    actions[0] = [move_cardinal_idx, 2]  # West
    env.step(actions)
    actions[0] = [move_cardinal_idx, 0]  # North
    env.step(actions)

    objects = env.grid_objects()
    assert (objects[agent_id]["r"], objects[agent_id]["c"]) == (1, 1)

    # Now both North and West should fail (walls)
    actions[0] = [move_cardinal_idx, 0]  # North
    env.step(actions)
    assert not env.action_success()[0]

    actions[0] = [move_cardinal_idx, 2]  # West
    env.step(actions)
    assert not env.action_success()[0]


def test_orientation_preserved_in_cardinal_mode():
    """Test that agent orientation is preserved for other actions like attack."""
    config = make_test_config(
        num_agents=2,
        map=[
            [".", ".", ".", ".", "."],
            [".", "agent.player", ".", "agent.enemy", "."],
            [".", ".", ".", ".", "."],
        ],
        actions={
            "move": {"enabled": False},
            "rotate": {"enabled": False},
            "move_cardinal": {"enabled": True},
            "attack": {"enabled": True, "consumed_resources": {}},
        },
    )

    env = MettaGrid(config, config["map"], seed=42)
    env.reset()

    objects = env.grid_objects()
    agent_ids = [id for id, obj in objects.items() if obj["type_name"] == "agent"]
    player_id = next(id for id in agent_ids if objects[id]["group_name"] == "player")
    enemy_id = next(id for id in agent_ids if objects[id]["group_name"] == "enemy")

    # Check initial orientations
    assert objects[player_id]["orientation"] == 0  # Up
    assert objects[enemy_id]["orientation"] == 0  # Up

    # Cardinal movement doesn't change orientation,
    # so orientation still affects directional actions like attack


def test_orientation_not_changed_by_cardinal_movement():
    """Test that agent orientation is NOT changed by cardinal movement."""
    config = make_test_config(
        num_agents=1,
        map=[
            ["wall", "wall", "wall"],
            ["wall", "agent.player", "wall"],
            ["wall", "wall", "wall"],
        ],
        actions={
            "move": {"enabled": False},
            "rotate": {"enabled": False},
            "move_cardinal": {"enabled": True},
        },
    )

    env = MettaGrid(config, config["map"], seed=42)
    env.reset()

    objects = env.grid_objects()
    agent_id = next(id for id, obj in objects.items() if obj["type_name"] == "agent")

    # Check initial orientation
    assert objects[agent_id]["orientation"] == 0  # Up

    action_names = env.action_names()
    move_cardinal_idx = action_names.index("move_cardinal")

    # Add rotate action to set initial orientation
    if "rotate" in action_names:
        rotate_idx = action_names.index("rotate")
        actions = np.zeros((1, 2), dtype=dtype_actions)
        actions[0] = [rotate_idx, 2]  # Face Left
        env.step(actions)
        
        objects = env.grid_objects()
        assert objects[agent_id]["orientation"] == 2  # Left

    # Try to move East into wall - should fail and NOT change orientation
    actions = np.zeros((1, 2), dtype=dtype_actions)
    actions[0] = [move_cardinal_idx, 3]  # East
    env.step(actions)

    objects = env.grid_objects()
    assert not env.action_success()[0]  # Movement should fail
    assert objects[agent_id]["orientation"] == 2  # Orientation should remain Left

    # Try to move North into wall
    actions[0] = [move_cardinal_idx, 0]  # North
    env.step(actions)

    objects = env.grid_objects()
    assert not env.action_success()[0]  # Movement should fail
    assert objects[agent_id]["orientation"] == 2  # Orientation should still be Left


def test_hybrid_movement_mode():
    """Test that both movement types can coexist in the same environment."""
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
            "move": {"enabled": True},
            "rotate": {"enabled": True},
            "move_cardinal": {"enabled": True},
        },
    )

    env = MettaGrid(config, config["map"], seed=42)
    env.reset()

    action_names = env.action_names()

    # All three movement-related actions should be present
    assert "move" in action_names
    assert "rotate" in action_names
    assert "move_cardinal" in action_names

    move_idx = action_names.index("move")
    rotate_idx = action_names.index("rotate")
    move_cardinal_idx = action_names.index("move_cardinal")

    objects = env.grid_objects()
    agent_id = next(id for id, obj in objects.items() if obj["type_name"] == "agent")

    # Test using cardinal movement
    actions = np.zeros((1, 2), dtype=dtype_actions)
    actions[0] = [move_cardinal_idx, 0]  # Move North
    env.step(actions)

    objects = env.grid_objects()
    assert (objects[agent_id]["r"], objects[agent_id]["c"]) == (1, 2)
    assert objects[agent_id]["orientation"] == 0  # Still facing initial direction (Up)

    # Test using tank-style movement
    actions[0] = [rotate_idx, 3]  # Rotate to face Right
    env.step(actions)

    objects = env.grid_objects()
    assert objects[agent_id]["orientation"] == 3  # Now facing Right

    actions[0] = [move_idx, 0]  # Move forward (East)
    env.step(actions)

    objects = env.grid_objects()
    assert (objects[agent_id]["r"], objects[agent_id]["c"]) == (1, 3)
