"""Tests for cardinal movement system."""

import numpy as np

from metta.mettagrid.mettagrid_c import MettaGrid
from metta.mettagrid.mettagrid_env import dtype_actions
from metta.mettagrid.util.action_remapper import TankToCardinalRemapper
from mettagrid.tests.conftest import make_test_config


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
        movement_mode="cardinal",
    )

    env = MettaGrid(config, config["map"], seed=42)
    env.reset()

    # Get initial agent position
    objects = env.grid_objects()
    agent_id = next(id for id, obj in objects.items() if obj["type_name"] == "agent")
    initial_pos = (objects[agent_id]["r"], objects[agent_id]["c"])
    assert initial_pos == (2, 2)

    # Test movement in each cardinal direction
    action_names = env.action_names()
    move_idx = action_names.index("move")

    # Note: In cardinal mode, rotate action should not be present
    assert "rotate" not in action_names

    # Move North (Up)
    actions = np.zeros((1, 2), dtype=dtype_actions)
    actions[0] = [move_idx, 0]  # 0 = North
    env.step(actions)

    objects = env.grid_objects()
    assert (objects[agent_id]["r"], objects[agent_id]["c"]) == (1, 2)

    # Move South (Down)
    actions[0] = [move_idx, 1]  # 1 = South
    env.step(actions)

    objects = env.grid_objects()
    assert (objects[agent_id]["r"], objects[agent_id]["c"]) == (2, 2)

    # Move West (Left)
    actions[0] = [move_idx, 2]  # 2 = West
    env.step(actions)

    objects = env.grid_objects()
    assert (objects[agent_id]["r"], objects[agent_id]["c"]) == (2, 1)

    # Move East (Right)
    actions[0] = [move_idx, 3]  # 3 = East
    env.step(actions)

    objects = env.grid_objects()
    assert (objects[agent_id]["r"], objects[agent_id]["c"]) == (2, 2)


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
        movement_mode="cardinal",
    )

    env = MettaGrid(config, config["map"], seed=42)
    env.reset()

    objects = env.grid_objects()
    agent_id = next(id for id, obj in objects.items() if obj["type_name"] == "agent")

    action_names = env.action_names()
    move_idx = action_names.index("move")

    # Try to move into walls - should fail
    actions = np.zeros((1, 2), dtype=dtype_actions)

    # North into wall
    actions[0] = [move_idx, 0]
    env.step(actions)
    success = env.action_success()
    assert not success[0]

    # Move to corner
    actions[0] = [move_idx, 2]  # West
    env.step(actions)
    actions[0] = [move_idx, 0]  # North
    env.step(actions)

    objects = env.grid_objects()
    assert (objects[agent_id]["r"], objects[agent_id]["c"]) == (1, 1)

    # Now both North and West should fail (walls)
    actions[0] = [move_idx, 0]  # North
    env.step(actions)
    assert not env.action_success()[0]

    actions[0] = [move_idx, 2]  # West
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
        movement_mode="cardinal",
        actions={"attack": {"enabled": True, "consumed_resources": {}}},
    )

    env = MettaGrid(config, config["map"], seed=42)
    env.reset()

    objects = env.grid_objects()
    agent_ids = [id for id, obj in objects.items() if obj["type_name"] == "agent"]
    player_id = next(id for id in agent_ids if objects[id]["group_name"] == "player")
    enemy_id = next(id for id in agent_ids if objects[id]["group_name"] == "enemy")

    # Check initial orientations (should still be tracked)
    assert objects[player_id]["orientation"] == 0  # Up
    assert objects[enemy_id]["orientation"] == 0  # Up

    # In cardinal mode, orientation doesn't affect movement
    # but it should still affect directional actions like attack


def test_orientation_updates_even_when_blocked():
    """Test that agent orientation updates even when movement is blocked."""
    config = make_test_config(
        num_agents=1,
        map=[
            ["wall", "wall", "wall"],
            ["wall", "agent.player", "wall"],
            ["wall", "wall", "wall"],
        ],
        movement_mode="cardinal",
    )

    env = MettaGrid(config, config["map"], seed=42)
    env.reset()

    objects = env.grid_objects()
    agent_id = next(id for id, obj in objects.items() if obj["type_name"] == "agent")

    # Check initial orientation
    assert objects[agent_id]["orientation"] == 0  # Up

    action_names = env.action_names()
    move_idx = action_names.index("move")

    # Try to move East into wall - should fail but still rotate
    actions = np.zeros((2, 2), dtype=dtype_actions)
    actions[0] = [move_idx, 3]  # East
    env.step(actions)

    objects = env.grid_objects()
    assert not env.action_success()[0]  # Movement should fail
    assert objects[agent_id]["orientation"] == 3  # But orientation should update to East

    # Try to move North into wall
    actions[0] = [move_idx, 0]  # North
    env.step(actions)

    objects = env.grid_objects()
    assert not env.action_success()[0]  # Movement should fail
    assert objects[agent_id]["orientation"] == 0  # But orientation should update to North


def test_action_remapper_tank_to_cardinal():
    """Test the tank to cardinal action remapper."""
    action_names = ["noop", "move", "rotate", "attack"]
    remapper = TankToCardinalRemapper(action_names)

    # Test forward movement conversion
    actions = np.array([[1, 0], [1, 0]], dtype=np.int32)  # Two agents moving forward
    orientations = {0: 0, 1: 2}  # Agent 0 facing Up, Agent 1 facing Left

    remapped = remapper.remap_actions(actions, orientations)

    # Agent 0 moving forward while facing Up -> Move North
    assert remapped[0, 0] == 1  # move action
    assert remapped[0, 1] == 0  # North

    # Agent 1 moving forward while facing Left -> Move West
    assert remapped[1, 0] == 1  # move action
    assert remapped[1, 1] == 2  # West

    # Test backward movement conversion
    actions = np.array([[1, 1], [1, 1]], dtype=np.int32)  # Moving backward
    remapped = remapper.remap_actions(actions, orientations)

    # Agent 0 moving backward while facing Up -> Move South
    assert remapped[0, 1] == 1  # South

    # Agent 1 moving backward while facing Left -> Move East
    assert remapped[1, 1] == 3  # East

    # Test rotation conversion
    actions = np.array([[2, 1], [2, 3]], dtype=np.int32)  # Rotate to Down, Right
    remapped = remapper.remap_actions(actions, orientations)

    # Rotations should be converted to moves in the target direction
    assert remapped[0, 0] == 1  # move action
    assert remapped[0, 1] == 1  # South (Down)

    assert remapped[1, 0] == 1  # move action
    assert remapped[1, 1] == 3  # East (Right)
