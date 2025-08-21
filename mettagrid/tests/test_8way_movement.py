"""Tests for 8-way movement system."""

import numpy as np

from metta.mettagrid.core import MettaGridCore
from metta.mettagrid.map_builder.ascii import AsciiMapBuilder
from metta.mettagrid.mettagrid_c import dtype_actions
from metta.mettagrid.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    EnvConfig,
    GameConfig,
    WallConfig,
)


def test_8way_movement_all_directions():
    """Test 8-way movement in all eight directions."""
    env_cfg = EnvConfig(
        game=GameConfig(
            num_agents=1,
            actions=ActionsConfig(
                move_8way=ActionConfig(),
                rotate=ActionConfig(),
                noop=ActionConfig(),
            ),
            map_builder=AsciiMapBuilder.Config(
                map_data=[
                    [".", ".", ".", ".", "."],
                    [".", ".", ".", ".", "."],
                    [".", ".", "@", ".", "."],
                    [".", ".", ".", ".", "."],
                    [".", ".", ".", ".", "."],
                ],
            ),
        )
    )
    env = MettaGridCore(env_cfg)
    env.reset()

    objects = env.grid_objects
    agent_id = next(id for id, obj in objects.items() if obj["type_id"] == 0)  # type_id 0 is agent
    initial_pos = (objects[agent_id]["r"], objects[agent_id]["c"])
    _ = objects[agent_id]["orientation"]
    assert initial_pos == (2, 2)

    action_names = env.action_names
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

        objects = env.grid_objects
        actual_pos = (objects[agent_id]["r"], objects[agent_id]["c"])
        assert actual_pos == expected_pos, f"Direction {direction}: expected {expected_pos}, got {actual_pos}"

        # Verify orientation has changed to match movement direction
        # For 8-way movement, diagonal directions map to nearest cardinal
        expected_orientations = {
            0: 0,  # North -> Up
            1: 0,  # Northeast -> Up
            2: 3,  # East -> Right
            3: 1,  # Southeast -> Down
            4: 1,  # South -> Down
            5: 1,  # Southwest -> Down
            6: 2,  # West -> Left
            7: 0,  # Northwest -> Up
        }
        if direction in expected_orientations:
            assert objects[agent_id]["orientation"] == expected_orientations[direction], (
                f"Direction {direction}: orientation should be {expected_orientations[direction]}"
            )


def test_8way_movement_obstacles():
    """Test that 8-way movement respects obstacles."""
    env_cfg = EnvConfig(
        game=GameConfig(
            num_agents=1,
            actions=ActionsConfig(
                move_8way=ActionConfig(),
                rotate=ActionConfig(),
                noop=ActionConfig(),
            ),
            objects={"wall": WallConfig(type_id=1)},
            map_builder=AsciiMapBuilder.Config(
                map_data=[
                    ["#", "#", "#", "#", "#"],
                    ["#", ".", ".", ".", "#"],
                    ["#", ".", "@", ".", "#"],
                    ["#", ".", ".", ".", "#"],
                    ["#", "#", "#", "#", "#"],
                ],
            ),
        )
    )
    env = MettaGridCore(env_cfg)
    env.reset()

    objects = env.grid_objects
    agent_id = next(id for id, obj in objects.items() if obj["type_id"] == 0)  # type_id 0 is agent

    action_names = env.action_names
    move_8dir_idx = action_names.index("move_8way")

    # Test diagonal movements near corners
    actions = np.zeros((1, 2), dtype=dtype_actions)

    # Move to top-left corner area
    actions[0] = [move_8dir_idx, 0]  # North
    env.step(actions)
    actions[0] = [move_8dir_idx, 6]  # West
    env.step(actions)

    objects = env.grid_objects
    assert (objects[agent_id]["r"], objects[agent_id]["c"]) == (1, 1)

    # Try to move Northwest into wall - should fail
    actions[0] = [move_8dir_idx, 7]  # Northwest
    env.step(actions)
    assert not env.action_success[0]

    # Position should remain unchanged
    objects = env.grid_objects
    assert (objects[agent_id]["r"], objects[agent_id]["c"]) == (1, 1)


def test_orientation_changes_with_8way():
    """Test that orientation changes to match movement direction with 8-way movement."""
    env_cfg = EnvConfig(
        game=GameConfig(
            num_agents=1,
            actions=ActionsConfig(
                move_8way=ActionConfig(),
                rotate=ActionConfig(),
                noop=ActionConfig(),
            ),
            map_builder=AsciiMapBuilder.Config(
                map_data=[
                    [".", ".", ".", ".", "."],
                    [".", "@", ".", ".", "."],
                    [".", ".", ".", ".", "."],
                ],
            ),
        ),
    )
    env = MettaGridCore(env_cfg)
    env.reset()

    objects = env.grid_objects
    agent_id = next(id for id, obj in objects.items() if obj["type_id"] == 0)  # type_id 0 is agent

    action_names = env.action_names
    move_8dir_idx = action_names.index("move_8way")
    rotate_idx = action_names.index("rotate")

    # First rotate to face right
    actions = np.zeros((1, 2), dtype=dtype_actions)
    actions[0] = [rotate_idx, 3]  # Face right
    env.step(actions)

    objects = env.grid_objects
    assert objects[agent_id]["orientation"] == 3  # Right

    # Now move in various directions and verify orientation changes appropriately
    expected_orientations = {
        0: 0,  # North -> Up
        1: 0,  # Northeast -> Up
        2: 3,  # East -> Right
        3: 1,  # Southeast -> Down
        4: 1,  # South -> Down
        5: 1,  # Southwest -> Down
        6: 2,  # West -> Left
        7: 0,  # Northwest -> Up
    }

    for direction in range(8):
        # Skip if movement would go out of bounds
        if direction in [0, 1, 7]:  # Skip north-facing movements from top row
            continue

        actions[0] = [move_8dir_idx, direction]
        env.step(actions)

        objects = env.grid_objects
        expected_orient = expected_orientations[direction]
        assert objects[agent_id]["orientation"] == expected_orient, (
            f"Direction {direction}: expected orientation {expected_orient}, got {objects[agent_id]['orientation']}"
        )


def test_cardinal_movement_changes_orientation():
    """Test that cardinal movement changes orientation to match movement direction."""
    env_cfg = EnvConfig(
        game=GameConfig(
            num_agents=1,
            actions=ActionsConfig(
                move_cardinal=ActionConfig(),
                rotate=ActionConfig(),
                noop=ActionConfig(),
            ),
            map_builder=AsciiMapBuilder.Config(
                map_data=[
                    [".", ".", ".", ".", "."],
                    [".", "@", ".", ".", "."],
                    [".", ".", ".", ".", "."],
                ],
            ),
        )
    )
    env = MettaGridCore(env_cfg)
    env.reset()

    objects = env.grid_objects
    agent_id = next(id for id, obj in objects.items() if obj["type_id"] == 0)  # type_id 0 is agent

    action_names = env.action_names
    move_cardinal_idx = action_names.index("move_cardinal")
    rotate_idx = action_names.index("rotate")

    # Rotate to face right
    actions = np.zeros((1, 2), dtype=dtype_actions)
    actions[0] = [rotate_idx, 3]  # Face right
    env.step(actions)

    objects = env.grid_objects
    assert objects[agent_id]["orientation"] == 3

    # Move north with cardinal movement
    actions[0] = [move_cardinal_idx, 0]  # North
    env.step(actions)

    objects = env.grid_objects
    assert (objects[agent_id]["r"], objects[agent_id]["c"]) == (0, 1)
    # Orientation should change to match movement direction (north = 0)
    assert objects[agent_id]["orientation"] == 0


def test_8way_movement_with_simple_environment():
    """Test 8-way movement using the simple environment builder."""
    # Create a larger environment to test diagonal movements
    env_cfg = EnvConfig(
        game=GameConfig(
            num_agents=1,
            actions=ActionsConfig(
                move_8way=ActionConfig(),
                rotate=ActionConfig(),
                noop=ActionConfig(),
            ),
            map_builder=AsciiMapBuilder.Config(
                map_data=[
                    [".", ".", ".", ".", ".", ".", ".", "."],
                    [".", ".", ".", ".", ".", ".", ".", "."],
                    [".", ".", ".", ".", ".", ".", ".", "."],
                    [".", ".", ".", ".", ".", ".", ".", "."],
                    [".", ".", ".", ".", "@", ".", ".", "."],
                    [".", ".", ".", ".", ".", ".", ".", "."],
                    [".", ".", ".", ".", ".", ".", ".", "."],
                    [".", ".", ".", ".", ".", ".", ".", "."],
                ],
            ),
        ),
    )
    env = MettaGridCore(env_cfg)
    env.reset()

    objects = env.grid_objects
    agent_id = next(id for id, obj in objects.items() if obj["type_id"] == 0)

    # Verify agent is at expected position
    assert (objects[agent_id]["r"], objects[agent_id]["c"]) == (4, 4)

    action_names = env.action_names
    move_8dir_idx = action_names.index("move_8way")

    # Test diagonal movement pattern (diamond shape)
    actions = np.zeros((1, 2), dtype=dtype_actions)

    # Move Northeast
    actions[0] = [move_8dir_idx, 1]
    env.step(actions)
    objects = env.grid_objects
    assert (objects[agent_id]["r"], objects[agent_id]["c"]) == (3, 5)

    # Move Southeast
    actions[0] = [move_8dir_idx, 3]
    env.step(actions)
    objects = env.grid_objects
    assert (objects[agent_id]["r"], objects[agent_id]["c"]) == (4, 6)

    # Move Southwest
    actions[0] = [move_8dir_idx, 5]
    env.step(actions)
    objects = env.grid_objects
    assert (objects[agent_id]["r"], objects[agent_id]["c"]) == (5, 5)

    # Move Northwest - back to start
    actions[0] = [move_8dir_idx, 7]
    env.step(actions)
    objects = env.grid_objects
    assert (objects[agent_id]["r"], objects[agent_id]["c"]) == (4, 4)


def test_8way_movement_boundary_check():
    """Test 8-way movement respects environment boundaries."""
    # Small environment to easily test boundaries
    env_cfg = EnvConfig(
        game=GameConfig(
            num_agents=1,
            actions=ActionsConfig(
                move_8way=ActionConfig(),
                rotate=ActionConfig(),
                noop=ActionConfig(),
            ),
            map_builder=AsciiMapBuilder.Config(
                map_data=[
                    [".", ".", "."],
                    [".", "@", "."],
                    [".", ".", "."],
                ],
            ),
        )
    )
    env = MettaGridCore(env_cfg)
    env.reset()

    objects = env.grid_objects
    agent_id = next(id for id, obj in objects.items() if obj["type_id"] == 0)

    action_names = env.action_names
    move_8dir_idx = action_names.index("move_8way")

    # Move to top-left corner
    actions = np.zeros((1, 2), dtype=dtype_actions)
    actions[0] = [move_8dir_idx, 7]  # Northwest
    env.step(actions)

    objects = env.grid_objects
    assert (objects[agent_id]["r"], objects[agent_id]["c"]) == (0, 0)

    # Try to move further northwest - should fail
    actions[0] = [move_8dir_idx, 7]  # Northwest
    env.step(actions)
    assert not env.action_success[0]

    # Try to move north - should fail
    actions[0] = [move_8dir_idx, 0]  # North
    env.step(actions)
    assert not env.action_success[0]

    # Try to move west - should fail
    actions[0] = [move_8dir_idx, 6]  # West
    env.step(actions)
    assert not env.action_success[0]

    # Move southeast - should succeed
    actions[0] = [move_8dir_idx, 3]  # Southeast
    env.step(actions)
    assert env.action_success[0]

    objects = env.grid_objects
    assert (objects[agent_id]["r"], objects[agent_id]["c"]) == (1, 1)


def test_orientation_changes_on_failed_8way_movement():
    """Test that orientation changes even when 8-way movement fails due to obstacles."""
    env_cfg = EnvConfig(
        game=GameConfig(
            num_agents=1,
            actions=ActionsConfig(
                move=ActionConfig(enabled=False),
                rotate=ActionConfig(enabled=True),  # Enable rotate to test orientation changes
                move_8way=ActionConfig(enabled=True),
            ),
            objects={"wall": WallConfig(type_id=1)},
            map_builder=AsciiMapBuilder.Config(
                map_data=[
                    ["#", "#", "#"],
                    ["#", "@", "#"],
                    ["#", "#", "#"],
                ],
            ),
        )
    )
    env = MettaGridCore(env_cfg)
    env.reset()

    objects = env.grid_objects
    agent_id = next(id for id, obj in objects.items() if obj["type_id"] == 0)

    # Check initial orientation
    assert objects[agent_id]["orientation"] == 0  # Up

    action_names = env.action_names
    move_8way_idx = action_names.index("move_8way")

    # Set initial orientation to Left
    if "rotate" in action_names:
        rotate_idx = action_names.index("rotate")
        actions = np.zeros((1, 2), dtype=dtype_actions)
        actions[0] = [rotate_idx, 2]  # Face Left
        env.step(actions)

        objects = env.grid_objects
        assert objects[agent_id]["orientation"] == 2  # Left

    # Try to move East into wall - should fail BUT change orientation
    actions = np.zeros((1, 2), dtype=dtype_actions)
    actions[0] = [move_8way_idx, 2]  # East
    env.step(actions)

    objects = env.grid_objects
    assert not env.action_success[0]  # Movement should fail
    assert objects[agent_id]["orientation"] == 3  # Orientation SHOULD change to Right (East)

    # Try to move Northeast into wall - should fail BUT change orientation
    # Northeast tries multiple fallbacks when blocked, final orientation depends on last attempt
    actions[0] = [move_8way_idx, 1]  # Northeast
    env.step(actions)

    objects = env.grid_objects
    assert not env.action_success[0]  # Movement should fail
    # For Northeast when all attempts fail, final orientation is Right (from last fallback attempt)
    assert objects[agent_id]["orientation"] == 3  # Orientation ends at Right after fallback attempts

    # Try to move Southwest into wall - should fail BUT change orientation
    actions[0] = [move_8way_idx, 5]  # Southwest
    env.step(actions)

    objects = env.grid_objects
    assert not env.action_success[0]  # Movement should fail
    # For Southwest when all attempts fail, final orientation is Left (from last fallback attempt)
    assert objects[agent_id]["orientation"] == 2  # Orientation ends at Left after fallback attempts
