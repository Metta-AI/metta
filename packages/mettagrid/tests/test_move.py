"""Tests for 8-way movement system."""

import numpy as np
import pytest

from mettagrid.config.mettagrid_c_config import from_mettagrid_config
from mettagrid.config.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    GameConfig,
    MettaGridConfig,
    WallConfig,
)
from mettagrid.core import MettaGridCore
from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.mapgen.utils.ascii_grid import DEFAULT_CHAR_TO_NAME
from mettagrid.mettagrid_c import (
    MettaGrid,
    dtype_actions,
    dtype_observations,
    dtype_rewards,
    dtype_terminals,
    dtype_truncations,
)
from mettagrid.test_support.actions import action_index, get_agent_position, move
from mettagrid.test_support.orientation import Orientation


# Test fixtures for MettaGrid environments
@pytest.fixture
def base_config() -> GameConfig:
    """Base configuration for MettaGrid tests."""
    return GameConfig(
        max_steps=50,
        num_agents=1,
        obs_width=3,
        obs_height=3,
        num_observation_tokens=100,
        resource_names=["laser", "armor"],
        actions=ActionsConfig(
            noop=ActionConfig(enabled=True),
            move=ActionConfig(enabled=True),
            rotate=ActionConfig(enabled=True),
        ),
        objects={
            "wall": WallConfig(type_id=1),
        },
        allow_diagonals=True,
    )


@pytest.fixture
def movement_game_map():
    """Game map with agent in center and room to move."""
    return [
        ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
        ["wall", "empty", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "empty", "empty", "agent.red", "empty", "empty", "wall"],  # Agent in center
        ["wall", "empty", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
    ]


@pytest.fixture
def small_movement_game_map():
    """Smaller game map for focused movement tests."""
    return [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "empty", "agent.red", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]


@pytest.fixture
def blocked_game_map():
    """Game map where agent is completely surrounded by walls."""
    return [
        ["wall", "wall", "wall"],
        ["wall", "agent.red", "wall"],
        ["wall", "wall", "wall"],
    ]


@pytest.fixture
def corridor_game_map():
    """Game map with a corridor for walking tests."""
    return [
        ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
    ]


@pytest.fixture
def configured_env(base_config):
    """Factory fixture that creates a configured MettaGrid environment."""

    def _create_env(game_map, config_overrides=None):
        if config_overrides:
            # Create a new config with overrides
            game_config = base_config.model_copy(update=config_overrides)
        else:
            game_config = base_config

        env = MettaGrid(from_mettagrid_config(game_config), game_map, 42)

        # Set up buffers
        observations = np.zeros((1, 100, 3), dtype=dtype_observations)
        terminals = np.zeros(1, dtype=dtype_terminals)
        truncations = np.zeros(1, dtype=dtype_truncations)
        rewards = np.zeros(1, dtype=dtype_rewards)
        env.set_buffers(observations, terminals, truncations, rewards)

        env.reset()
        return env

    return _create_env


# Tests for MettaGridCore (low-level API)
def test_8way_movement_all_directions():
    """Test 8-way movement in all eight directions using MettaGridCore."""
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            actions=ActionsConfig(
                move=ActionConfig(),
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
                char_to_name_map=DEFAULT_CHAR_TO_NAME,
            ),
            allow_diagonals=True,
        )
    )
    env = MettaGridCore(cfg)
    env.reset()
    objects = env.grid_objects()
    agent_id = next(id for id, obj in objects.items() if obj["type_id"] == 0)  # type_id 0 is agent
    initial_pos = (objects[agent_id]["r"], objects[agent_id]["c"])
    assert initial_pos == (2, 2)

    # Test moves with correct enum values and expected positions
    # Starting from (2,2) in center
    moves = [
        (Orientation.NORTH, (1, 2)),  # North (0) - up
        (Orientation.EAST, (1, 3)),  # East (3) - right from (1,2)
        (Orientation.SOUTH, (2, 3)),  # South (1) - down
        (Orientation.WEST, (2, 2)),  # West (2) - left back to center
        (Orientation.NORTHEAST, (1, 3)),  # Northeast (5) - up-right
        (Orientation.SOUTHEAST, (2, 4)),  # Southeast (7) - down-right
        (Orientation.SOUTHWEST, (3, 3)),  # Southwest (6) - down-left
        (Orientation.NORTHWEST, (2, 2)),  # Northwest (4) - up-left back to center
    ]

    for orientation, expected_pos in moves:
        actions = np.zeros((1,), dtype=dtype_actions)
        actions[0] = action_index(env, "move", orientation)
        env.step(actions)

        objects = env.grid_objects()
        actual_pos = (objects[agent_id]["r"], objects[agent_id]["c"])
        assert actual_pos == expected_pos, f"Direction {orientation.name}: expected {expected_pos}, got {actual_pos}"

        # Verify orientation changes to match the exact movement direction
        # Based on move.hpp: actor->orientation = move_direction;
        actual_facing = objects[agent_id]["orientation"]
        expected_facing = orientation.value
        assert actual_facing == expected_facing, (
            f"After moving {orientation.name}, expected facing {expected_facing}, got {actual_facing}"
        )


def test_8way_movement_obstacles():
    """Test that 8-way movement respects obstacles."""
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            actions=ActionsConfig(
                move=ActionConfig(),
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
                char_to_name_map=DEFAULT_CHAR_TO_NAME,
            ),
        )
    )
    cfg.game.allow_diagonals = True
    env = MettaGridCore(cfg)
    env.reset()

    objects = env.grid_objects()
    agent_id = next(id for id, obj in objects.items() if obj["type_id"] == 0)

    # Test diagonal movements near corners
    actions = np.zeros((1,), dtype=dtype_actions)

    # Move to top-left corner area
    actions[0] = action_index(env, "move", Orientation.NORTH)
    env.step(actions)
    actions[0] = action_index(env, "move", Orientation.WEST)
    env.step(actions)

    objects = env.grid_objects()
    assert (objects[agent_id]["r"], objects[agent_id]["c"]) == (1, 1)

    # Try to move Northwest into wall - should fail
    actions[0] = action_index(env, "move", Orientation.NORTHWEST)
    env.step(actions)
    assert not env.action_success[0]

    # Position should remain unchanged
    objects = env.grid_objects()
    assert (objects[agent_id]["r"], objects[agent_id]["c"]) == (1, 1)


def test_orientation_changes_with_8way():
    """Test that orientation changes to match movement direction with 8-way movement."""
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            actions=ActionsConfig(
                move=ActionConfig(),
                rotate=ActionConfig(),
                noop=ActionConfig(),
            ),
            map_builder=AsciiMapBuilder.Config(
                map_data=[
                    [".", ".", ".", ".", "."],
                    [".", "@", ".", ".", "."],
                    [".", ".", ".", ".", "."],
                ],
                char_to_name_map=DEFAULT_CHAR_TO_NAME,
            ),
        ),
    )
    env = MettaGridCore(cfg)
    env.reset()

    objects = env.grid_objects()
    agent_id = next(id for id, obj in objects.items() if obj["type_id"] == 0)

    # First rotate to face right
    actions = np.zeros((1,), dtype=dtype_actions)
    actions[0] = action_index(env, "rotate", Orientation.EAST)
    env.step(actions)

    objects = env.grid_objects()
    assert objects[agent_id]["orientation"] == 3  # Right

    # Test movements and verify orientation changes
    # Based on move.hpp: actor->orientation = move_direction
    test_moves = [
        (Orientation.EAST, Orientation.EAST.value),  # East -> faces East (3)
        (Orientation.SOUTH, Orientation.SOUTH.value),  # South -> faces South (1)
        (Orientation.WEST, Orientation.WEST.value),  # West -> faces West (2)
        (Orientation.SOUTHEAST, Orientation.SOUTHEAST.value),  # Southeast -> faces Southeast (7)
        (Orientation.SOUTHWEST, Orientation.SOUTHWEST.value),  # Southwest -> faces Southwest (6)
    ]

    for orientation, expected_facing in test_moves:
        # Skip movements that would go out of bounds
        objects = env.grid_objects()
        r, c = objects[agent_id]["r"], objects[agent_id]["c"]

        # Check bounds based on movement delta
        dr, dc = 0, 0
        if orientation == Orientation.NORTH:
            dr = -1
        elif orientation == Orientation.SOUTH:
            dr = 1
        elif orientation == Orientation.WEST:
            dc = -1
        elif orientation == Orientation.EAST:
            dc = 1
        elif orientation == Orientation.NORTHWEST:
            dr, dc = -1, -1
        elif orientation == Orientation.NORTHEAST:
            dr, dc = -1, 1
        elif orientation == Orientation.SOUTHWEST:
            dr, dc = 1, -1
        elif orientation == Orientation.SOUTHEAST:
            dr, dc = 1, 1

        new_r, new_c = r + dr, c + dc
        if new_r < 0 or new_r >= 3 or new_c < 0 or new_c >= 5:
            continue

        actions[0] = action_index(env, "move", orientation)
        env.step(actions)

        objects = env.grid_objects()
        actual_facing = objects[agent_id]["orientation"]
        assert actual_facing == expected_facing, (
            f"Direction {orientation.name}: expected facing {expected_facing}, got {actual_facing}"
        )


def test_8way_movement_with_simple_environment():
    """Test 8-way movement using the simple environment builder."""
    # Create a larger environment to test diagonal movements
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            actions=ActionsConfig(
                move=ActionConfig(),
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
                char_to_name_map=DEFAULT_CHAR_TO_NAME,
            ),
            allow_diagonals=True,
        ),
    )
    env = MettaGridCore(cfg)
    env.reset()

    objects = env.grid_objects()
    agent_id = next(id for id, obj in objects.items() if obj["type_id"] == 0)

    # Verify agent is at expected position
    assert (objects[agent_id]["r"], objects[agent_id]["c"]) == (4, 4)

    # Test diagonal movement pattern (diamond shape)
    actions = np.zeros((1,), dtype=dtype_actions)

    # Move Northeast
    actions[0] = action_index(env, "move", Orientation.NORTHEAST)
    env.step(actions)
    objects = env.grid_objects()
    assert (objects[agent_id]["r"], objects[agent_id]["c"]) == (3, 5)

    # Move Southeast
    actions[0] = action_index(env, "move", Orientation.SOUTHEAST)
    env.step(actions)
    objects = env.grid_objects()
    assert (objects[agent_id]["r"], objects[agent_id]["c"]) == (4, 6)

    # Move Southwest
    actions[0] = action_index(env, "move", Orientation.SOUTHWEST)
    env.step(actions)
    objects = env.grid_objects()
    assert (objects[agent_id]["r"], objects[agent_id]["c"]) == (5, 5)

    # Move Northwest - back to start
    actions[0] = action_index(env, "move", Orientation.NORTHWEST)
    env.step(actions)
    objects = env.grid_objects()
    assert (objects[agent_id]["r"], objects[agent_id]["c"]) == (4, 4)


def test_8way_movement_boundary_check():
    """Test 8-way movement respects environment boundaries."""
    # Small environment to easily test boundaries
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            actions=ActionsConfig(
                move=ActionConfig(),
                rotate=ActionConfig(),
                noop=ActionConfig(),
            ),
            map_builder=AsciiMapBuilder.Config(
                map_data=[
                    [".", ".", "."],
                    [".", "@", "."],
                    [".", ".", "."],
                ],
                char_to_name_map=DEFAULT_CHAR_TO_NAME,
            ),
            allow_diagonals=True,
        )
    )
    env = MettaGridCore(cfg)
    env.reset()

    objects = env.grid_objects()
    agent_id = next(id for id, obj in objects.items() if obj["type_id"] == 0)

    # Move to top-left corner
    actions = np.zeros((1,), dtype=dtype_actions)
    actions[0] = action_index(env, "move", Orientation.NORTHWEST)
    env.step(actions)

    objects = env.grid_objects()
    assert (objects[agent_id]["r"], objects[agent_id]["c"]) == (0, 0)

    # Try to move further northwest - should fail
    actions[0] = action_index(env, "move", Orientation.NORTHWEST)
    env.step(actions)
    assert not env.action_success[0]

    # Try to move north - should fail
    actions[0] = action_index(env, "move", Orientation.NORTH)
    env.step(actions)
    assert not env.action_success[0]

    # Try to move west - should fail
    actions[0] = action_index(env, "move", Orientation.WEST)
    env.step(actions)
    assert not env.action_success[0]

    # Move southeast - should succeed
    actions[0] = action_index(env, "move", Orientation.SOUTHEAST)
    env.step(actions)
    assert env.action_success[0]

    objects = env.grid_objects()
    assert (objects[agent_id]["r"], objects[agent_id]["c"]) == (1, 1)


def test_orientation_changes_on_failed_8way_movement():
    """Test that orientation DOES change when 8-way movement fails due to obstacles (new behavior)."""
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            allow_diagonals=True,  # Enable diagonal movements for this test
            actions=ActionsConfig(
                rotate=ActionConfig(),
                move=ActionConfig(),
            ),
            objects={"wall": WallConfig(type_id=1)},
            map_builder=AsciiMapBuilder.Config(
                map_data=[
                    ["#", "#", "#"],
                    ["#", "@", "#"],
                    ["#", "#", "#"],
                ],
                char_to_name_map=DEFAULT_CHAR_TO_NAME,
            ),
        )
    )
    env = MettaGridCore(cfg)
    env.reset()

    objects = env.grid_objects()
    agent_id = next(id for id, obj in objects.items() if obj["type_id"] == 0)

    # Check initial orientation
    assert objects[agent_id]["orientation"] == 0  # Up

    # Set initial orientation to Left
    action_names = env.action_names
    if "rotate_west" in action_names:
        actions = np.zeros((1,), dtype=dtype_actions)
        actions[0] = action_index(env, "rotate", Orientation.WEST)
        env.step(actions)

        objects = env.grid_objects()
        assert objects[agent_id]["orientation"] == 2  # Left

    # Try to move East into wall - should fail but SHOULD change orientation to East
    actions = np.zeros((1,), dtype=dtype_actions)
    actions[0] = action_index(env, "move", Orientation.EAST)
    env.step(actions)

    objects = env.grid_objects()
    assert not env.action_success[0]  # Movement should fail
    assert objects[agent_id]["orientation"] == Orientation.EAST.value  # Orientation should change to East

    # Try to move Northeast into wall - should fail but SHOULD change orientation to Northeast
    actions[0] = action_index(env, "move", Orientation.NORTHEAST)
    env.step(actions)

    objects = env.grid_objects()
    assert not env.action_success[0]  # Movement should fail
    assert objects[agent_id]["orientation"] == Orientation.NORTHEAST.value  # Orientation should change to Northeast

    # Try to move Southwest into wall - should fail but SHOULD change orientation to Southwest
    actions[0] = action_index(env, "move", Orientation.SOUTHWEST)
    env.step(actions)

    objects = env.grid_objects()
    assert not env.action_success[0]  # Movement should fail
    assert objects[agent_id]["orientation"] == Orientation.SOUTHWEST.value  # Orientation should change to Southwest


# Tests for MettaGrid (high-level API) using helper functions
def test_move_all_directions(configured_env, movement_game_map):
    """Test the move function in all eight compass directions."""
    env = configured_env(movement_game_map)

    initial_pos = get_agent_position(env)
    assert initial_pos is not None, "Agent should have a valid initial position"

    # Test cardinal directions first
    cardinal_tests = [
        (Orientation.NORTH, (-1, 0)),
        (Orientation.EAST, (0, 1)),
        (Orientation.SOUTH, (1, 0)),
        (Orientation.WEST, (0, -1)),
    ]

    for orientation, (expected_dr, expected_dc) in cardinal_tests:
        direction_name = str(orientation)

        print(f"Testing move {direction_name} (value {orientation.value})")

        position_before = get_agent_position(env, 0)
        result = move(env, orientation)
        position_after = get_agent_position(env, 0)

        # Assert movement was successful
        assert result["success"], f"Move {direction_name} should succeed. Error: {result.get('error', 'Unknown')}"

        # Assert position changed
        assert position_before != position_after, f"Agent should have moved {direction_name}"

        # Assert movement was in correct direction
        dr = position_after[0] - position_before[0]
        dc = position_after[1] - position_before[1]
        assert (dr, dc) == (expected_dr, expected_dc), f"Agent should have moved correctly {direction_name}"

        print(f"✅ Move {direction_name}: {position_before} → {position_after}")


def test_move_diagonal_directions(configured_env, movement_game_map):
    """Test the move function in all four diagonal directions."""
    env = configured_env(movement_game_map)

    # Test diagonal directions
    diagonal_tests = [
        (Orientation.NORTHEAST, (-1, 1)),
        (Orientation.SOUTHEAST, (1, 1)),
        (Orientation.SOUTHWEST, (1, -1)),
        (Orientation.NORTHWEST, (-1, -1)),
    ]

    for orientation, (expected_dr, expected_dc) in diagonal_tests:
        # Reset to center for each test
        env = configured_env(movement_game_map)

        direction_name = str(orientation)
        print(f"Testing diagonal move {direction_name} (value {orientation.value})")

        position_before = get_agent_position(env, 0)
        result = move(env, orientation)
        position_after = get_agent_position(env, 0)

        # Assert movement was successful
        assert result["success"], f"Move {direction_name} should succeed. Error: {result.get('error', 'Unknown')}"

        # Assert position changed
        assert position_before != position_after, f"Agent should have moved {direction_name}"

        # Assert movement was in correct direction
        dr = position_after[0] - position_before[0]
        dc = position_after[1] - position_before[1]
        assert (dr, dc) == (expected_dr, expected_dc), f"Agent should have moved correctly {direction_name}"

        print(f"✅ Move {direction_name}: {position_before} → {position_after}")


def test_move_up(configured_env, small_movement_game_map):
    """Test moving north specifically."""
    env = configured_env(small_movement_game_map, {"max_steps": 10})

    # Get position before move
    position_before = get_agent_position(env, 0)

    result = move(env, Orientation.NORTH)  # Use Orientation.NORTH for moving up

    assert result["success"], f"Move north should succeed. Error: {result.get('error')}"

    # Get position after move and verify
    position_after = get_agent_position(env, 0)
    assert position_before[0] - position_after[0] == 1, "Should move up by 1 row"


def test_move_blocked_by_wall(configured_env, blocked_game_map):
    """Test that movement is properly blocked by walls."""
    env = configured_env(blocked_game_map, {"max_steps": 10})

    directions = [
        Orientation.NORTH,
        Orientation.EAST,
        Orientation.SOUTH,
        Orientation.WEST,
    ]

    for orientation in directions:
        position_before = get_agent_position(env, 0)
        result = move(env, orientation)
        position_after = get_agent_position(env, 0)
        direction_name = str(orientation)

        # Movement should fail or position should remain unchanged
        if result["success"]:
            # This shouldn't happen for blocked movement
            raise AssertionError(f"Move {direction_name} should fail when blocked by wall")
        else:
            # Action failed, which is expected for blocked movement
            assert result["error"] is not None, f"Failed move {direction_name} should have an error message"
            assert position_before == position_after, "Position should not change when blocked"


def test_move_returns_to_center(configured_env, movement_game_map):
    """Test that we can move in a square and return to center."""
    env = configured_env(movement_game_map)

    initial_pos = get_agent_position(env)

    # Move in a square: north, east, south, west
    moves = [
        Orientation.NORTH,
        Orientation.EAST,
        Orientation.SOUTH,
        Orientation.WEST,
    ]

    for orientation in moves:
        result = move(env, orientation)
        direction_name = str(orientation)

        assert result["success"], f"Move {direction_name} should succeed"

    # Should be back at original position
    final_pos = get_agent_position(env)
    assert final_pos == initial_pos, f"Agent should return to original position {initial_pos}, but is at {final_pos}"


def test_agent_walks_across_room(configured_env, corridor_game_map):
    """
    Test where a single agent walks across a room.
    Creates a simple corridor and attempts to walk the agent from one end to the other.
    """
    print("Testing agent walking across room...")

    # Create environment with walking-specific config
    env = configured_env(
        corridor_game_map,
        {
            "max_steps": 20,
            "obs_width": 3,
            "obs_height": 3,
        },
    )

    print(f"Environment created: {env.map_width}x{env.map_height}")
    print(f"Initial timestep: {env.current_step}")

    # Find a working direction using Orientation enum
    successful_moves = []
    total_moves = 0

    print("\n=== Testing which direction allows movement ===")
    working_direction = None

    # Test all cardinal directions
    directions = [
        Orientation.NORTH,
        Orientation.EAST,
        Orientation.SOUTH,
        Orientation.WEST,
    ]

    for orientation in directions:
        direction_name = str(orientation)

        print(f"\nTesting movement {direction_name}...")

        result = move(env, orientation, agent_idx=0)

        if result["success"]:
            print(f"✓ Found working direction: {direction_name}")
            working_direction = orientation
            working_direction_str = direction_name
            break
        else:
            print(f"✗ Direction {direction_name} failed: {result.get('error', 'Unknown error')}")

    # Assert we found at least one working direction
    assert working_direction is not None, "Should find at least one direction that allows movement"

    print(f"\n=== Walking across room in direction: {working_direction_str} ===")

    # Reset for clean walk
    env = configured_env(
        corridor_game_map,
        {
            "max_steps": 20,
            "obs_width": 3,
            "obs_height": 3,
        },
    )

    # Walk multiple steps
    max_steps = 5

    for step in range(1, max_steps + 1):
        print(f"\n--- Step {step}: Moving {working_direction_str} ---")

        position_before = get_agent_position(env, 0)
        result = move(env, working_direction, agent_idx=0)
        position_after = get_agent_position(env, 0)
        total_moves += 1

        if result["success"]:
            successful_moves.append(step)
            print(f"✓ Successful move #{len(successful_moves)}")
            print(f"  Position: {position_before} → {position_after}")
        else:
            print(f"✗ Move failed: {result.get('error', 'Unknown error')}")
            print("  Agent likely hit an obstacle or boundary")
            break

        if env.current_step >= 18:
            print("  Approaching max steps limit")
            break

    print("\n=== Walking Test Summary ===")
    print(f"Working direction: {working_direction_str}")
    print(f"Total move attempts: {total_moves}")
    print(f"Successful moves: {len(successful_moves)}")
    print(f"Success rate: {len(successful_moves) / total_moves:.1%}" if total_moves > 0 else "N/A")

    # Validation
    assert len(successful_moves) >= 1, (
        f"Agent should have moved at least once. Got {len(successful_moves)} successful moves."
    )

    assert total_moves > 0, "Should have attempted at least one move"

    print("✅ Agent walking test passed!")


def test_agent_walks_in_all_cardinal_directions(configured_env, corridor_game_map):
    """Test that agent can move in all non-blocked cardinal directions."""
    print("Testing agent movement in all possible cardinal directions...")

    env = configured_env(corridor_game_map, {"max_steps": 20})

    successful_directions = []
    failed_directions = []

    # Test all cardinal directions using Orientation
    directions = [
        Orientation.NORTH,
        Orientation.EAST,
        Orientation.SOUTH,
        Orientation.WEST,
    ]

    for orientation in directions:
        direction_name = str(orientation)

        print(f"\nTesting {direction_name} (value {orientation.value})...")

        # Reset environment for each direction test
        env = configured_env(corridor_game_map, {"max_steps": 20})

        position_before = get_agent_position(env, 0)
        result = move(env, orientation, agent_idx=0)
        position_after = get_agent_position(env, 0)

        if result["success"] and position_before != position_after:
            successful_directions.append(direction_name)
            print(f"✓ {direction_name}: {position_before} → {position_after}")
        else:
            failed_directions.append(direction_name)
            print(f"✗ {direction_name}: {result.get('error', 'Movement failed')}")

    print("\nResults:")
    print(f"Successful directions: {successful_directions}")
    print(f"Failed directions: {failed_directions}")

    # Based on the corridor map, "east" should work (agent starts at (1,1), can move to (1,2))
    # west should be blocked by wall, north/south might be blocked depending on exact positioning
    assert len(successful_directions) >= 1, (
        f"Should be able to move in at least one direction. "
        f"Successful: {successful_directions}, Failed: {failed_directions}"
    )

    # In a corridor, we expect east to work since agent starts next to empty spaces
    if "east" not in successful_directions:
        print("Warning: East movement failed, which is unexpected in this corridor layout")


def test_orientation_enum_functionality():
    """Test that the Orientation enum works as expected."""
    assert Orientation.NORTH.value == 0
    assert Orientation.SOUTH.value == 1
    assert Orientation.WEST.value == 2
    assert Orientation.EAST.value == 3
    assert Orientation.NORTHWEST.value == 4
    assert Orientation.NORTHEAST.value == 5
    assert Orientation.SOUTHWEST.value == 6
    assert Orientation.SOUTHEAST.value == 7

    assert str(Orientation.NORTH) == "north"
    assert str(Orientation.EAST) == "east"
    assert str(Orientation.SOUTH) == "south"
    assert str(Orientation.WEST) == "west"
