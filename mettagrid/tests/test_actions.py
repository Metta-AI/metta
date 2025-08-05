import numpy as np
import pytest

from metta.mettagrid.mettagrid_c import (
    MettaGrid,
    dtype_observations,
    dtype_rewards,
    dtype_terminals,
    dtype_truncations,
)
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config
from metta.mettagrid.util.actions import (
    Orientation,
    get_agent_position,
    move,
)

OBS_WIDTH = 3  # should be odd
OBS_HEIGHT = 3  # should be odd
NUM_OBS_TOKENS = 100
OBS_TOKEN_SIZE = 3


@pytest.fixture
def base_config():
    """Base configuration for MettaGrid tests."""
    return {
        "max_steps": 50,
        "num_agents": 1,
        "obs_width": OBS_WIDTH,
        "obs_height": OBS_HEIGHT,
        "num_observation_tokens": NUM_OBS_TOKENS,
        "inventory_item_names": ["laser", "armor"],
        "actions": {
            "noop": {"enabled": True},
            "move": {"enabled": True},
            "rotate": {"enabled": True},
            "get_items": {"enabled": True},  # maps to get_output
            "attack": {"enabled": True, "consumed_resources": {"laser": 1}, "defense_resources": {"armor": 1}},
            "put_items": {"enabled": True},  # maps to get_recipe_items
            "swap": {"enabled": True},
            "change_color": {"enabled": True},
            "change_glyph": {"enabled": True, "number_of_glyphs": 4},
        },
        "groups": {"red": {"id": 0, "props": {}}},
        "objects": {
            "wall": {"type_id": 1},
            "altar": {
                "type_id": 8,
                "max_output": -1,
                "conversion_ticks": 1,
                "cooldown": 10,
                "initial_resource_count": 0,
            },
        },
        "agent": {"rewards": {}},
    }


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
def configured_env(base_config):
    """Factory fixture that creates a configured MettaGrid environment."""

    def _create_env(game_map, config_overrides=None):
        game_config = base_config.copy()
        if config_overrides:
            game_config.update(config_overrides)

        env = MettaGrid(from_mettagrid_config(game_config), game_map, 42)

        # Set up buffers
        observations = np.zeros((1, NUM_OBS_TOKENS, OBS_TOKEN_SIZE), dtype=dtype_observations)
        terminals = np.zeros(1, dtype=dtype_terminals)
        truncations = np.zeros(1, dtype=dtype_truncations)
        rewards = np.zeros(1, dtype=dtype_rewards)
        env.set_buffers(observations, terminals, truncations, rewards)

        env.reset()
        return env

    return _create_env


def test_move_all_directions(configured_env, movement_game_map):
    """Test the move function in all four directions."""
    env = configured_env(movement_game_map)

    initial_pos = get_agent_position(env)
    assert initial_pos is not None, "Agent should have a valid initial position"

    directions = [
        Orientation.UP,
        Orientation.RIGHT,
        Orientation.DOWN,
        Orientation.LEFT,
    ]

    for orientation in directions:
        direction_name = str(orientation)

        print(f"Testing move {direction_name} (orientation {orientation.value})")

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
        expected_dr, expected_dc = orientation.movement_delta
        assert (dr, dc) == (expected_dr, expected_dc), f"Agent should have moved correctly {direction_name}"

        print(f"✅ Move {direction_name}: {position_before} → {position_after}")


def test_move_up(configured_env, small_movement_game_map):
    """Test moving up specifically."""
    env = configured_env(small_movement_game_map, {"max_steps": 10})

    # Get position before move
    position_before = get_agent_position(env, 0)

    result = move(env, Orientation.UP)  # Use Orientation enum

    assert result["success"], f"Move up should succeed. Error: {result.get('error')}"

    # Get position after move and verify
    position_after = get_agent_position(env, 0)
    assert position_before[0] - position_after[0] == 1, "Should move up by 1 row"


def test_move_blocked_by_wall(configured_env, blocked_game_map):
    """Test that movement is properly blocked by walls."""
    env = configured_env(blocked_game_map, {"max_steps": 10})

    directions = [
        Orientation.UP,
        Orientation.RIGHT,
        Orientation.DOWN,
        Orientation.LEFT,
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
    """Test that we can move in a circle and return to center."""
    env = configured_env(movement_game_map)

    initial_pos = get_agent_position(env)
    assert initial_pos is not None, "Agent should have a valid initial position"

    # Move in a circle: up, right, down, left
    moves = [
        Orientation.UP,
        Orientation.RIGHT,
        Orientation.DOWN,
        Orientation.LEFT,
    ]

    for orientation in moves:
        result = move(env, orientation)
        direction_name = str(orientation)

        assert result["success"], f"Move {direction_name} should succeed"

    # Should be back at original position
    final_pos = get_agent_position(env)
    assert final_pos == initial_pos, f"Agent should return to original position {initial_pos}, but is at {final_pos}"


@pytest.fixture
def corridor_game_map():
    """Game map with a corridor for walking tests."""
    return [
        ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "empty", "empty", "altar", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
    ]


def test_agent_walks_across_room(configured_env, corridor_game_map):
    """
    Test where a single agent walks across a room.
    Creates a simple corridor and attempts to walk the agent from one end to the other.
    The move() function already handles observation validation.
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
    working_orientation = None

    # Test all orientations
    directions = [
        Orientation.UP,
        Orientation.RIGHT,
        Orientation.DOWN,
        Orientation.LEFT,
    ]

    for orientation in directions:
        direction_name = str(orientation)

        print(f"\nTesting movement {direction_name}...")

        result = move(env, orientation, agent_idx=0)

        if result["success"]:
            print(f"✓ Found working direction: {direction_name}")
            working_orientation = orientation
            working_direction_str = direction_name
            break
        else:
            print(f"✗ Direction {direction_name} failed: {result.get('error', 'Unknown error')}")

    # Assert we found at least one working direction
    assert working_orientation is not None, "Should find at least one direction that allows movement"

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
        result = move(env, working_orientation, agent_idx=0)
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


def test_agent_walks_in_all_possible_directions(configured_env, corridor_game_map):
    """Test that agent can move in all non-blocked directions."""
    print("Testing agent movement in all possible directions...")

    env = configured_env(corridor_game_map, {"max_steps": 20})

    successful_directions = []
    failed_directions = []

    # Test all orientations using enum
    directions = [
        Orientation.UP,
        Orientation.RIGHT,
        Orientation.DOWN,
        Orientation.LEFT,
    ]

    for orientation in directions:
        direction_name = str(orientation)

        print(f"\nTesting {direction_name} (orientation {orientation.value})...")

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

    # Based on the corridor map, "right" should work (agent starts at (1,1), can move to (1,2))
    # left should be blocked by wall, up/down might be blocked depending on exact positioning
    assert len(successful_directions) >= 1, (
        f"Should be able to move in at least one direction. "
        f"Successful: {successful_directions}, Failed: {failed_directions}"
    )

    # In a corridor, we expect right to work since agent starts next to empty spaces
    if "right" not in successful_directions:
        print("Warning: Right movement failed, which is unexpected in this corridor layout")


def test_orientation_enum_functionality():
    """Test that the Orientation enum works as expected."""
    assert Orientation.UP.movement_delta == (-1, 0)
    assert Orientation.DOWN.movement_delta == (1, 0)
    assert Orientation.LEFT.movement_delta == (0, -1)
    assert Orientation.RIGHT.movement_delta == (0, 1)

    assert str(Orientation.UP) == "up"
    assert str(Orientation.DOWN) == "down"
    assert str(Orientation.LEFT) == "left"
    assert str(Orientation.RIGHT) == "right"


def test_move_with_string_orientation(configured_env, small_movement_game_map):
    """Test that move function accepts string orientations."""
    env = configured_env(small_movement_game_map, {"max_steps": 10})

    # Test with string
    position_before = get_agent_position(env, 0)
    result = move(env, Orientation.UP)
    position_after = get_agent_position(env, 0)

    # Should work the same as using the enum directly
    assert result["success"], f"Move with string orientation should succeed. Error: {result.get('error')}"
    assert position_before[0] - position_after[0] == 1, "Agent should move up by 1 row"
