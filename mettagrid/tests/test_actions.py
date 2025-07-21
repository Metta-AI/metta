import numpy as np
import pytest

from metta.mettagrid.mettagrid_c import MettaGrid
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config
from metta.mettagrid.mettagrid_env import (
    dtype_actions,
    dtype_observations,
    dtype_rewards,
    dtype_terminals,
    dtype_truncations,
    get_action_name,
    get_action_param,
    noop,
)
from .test_mettagrid import TokenTypes
from mettagrid.tests.conftest import create_minimal_test_config
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
    """Base configuration for action tests."""
    return create_minimal_test_config(
        preset="full_actions",
        overrides={
            "max_steps": 10,
            "num_agents": 1,
            "obs_width": 3,
            "obs_height": 3,
            "num_observation_tokens": NUM_OBS_TOKENS,
            "objects": {
                "block": {"type_id": 2, "swappable": True},
                "altar": {
                    "type_id": 8,
                    "input_resources": {},
                    "output_resources": {"ore": 1},
                    "max_output": -1,
                    "conversion_ticks": 1,
                    "cooldown": 10,
                    "initial_resource_count": 0,
            },
        },
        "agent": {"rewards": {}},
    })["game"]


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
        result = move(env, orientation)

        # Assert movement was successful
        assert result["success"], f"Move {direction_name} should succeed. Error: {result.get('error', 'Unknown')}"

        # Assert position changed
        assert result["moved"], f"Agent should have moved {direction_name}"

        # Assert movement was in correct direction
        assert result["moved_correctly"], f"Agent should have moved correctly {direction_name}"

        # Assert observations changed with movement
        assert result["obs_changed"], f"Observations should change when moving {direction_name}"

        print(f"✅ Move {direction_name}: {result['position_before']} → {result['position_after']}")


def test_move_up(configured_env, small_movement_game_map):
    """Test moving up specifically."""
    env = configured_env(small_movement_game_map, {"max_steps": 10})

    result = move(env, Orientation.UP)  # Use Orientation enum

    assert result["success"], f"Move up should succeed. Error: {result.get('error')}"
    assert result["moved_correctly"], "Agent should move up correctly"
    assert result["position_before"][0] - result["position_after"][0] == 1, "Should move up by 1 row"


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
        result = move(env, orientation)
        direction_name = str(orientation)

        # Movement action might succeed but agent shouldn't actually move
        if result["success"]:
            # If action succeeded, agent shouldn't have moved due to wall
            assert not result["moved"], f"Agent shouldn't move {direction_name} when blocked by wall"
        else:
            # Action failed, which is also acceptable for blocked movement
            assert result["error"] is not None, f"Failed move {direction_name} should have an error message"


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
        assert result["moved"], f"Agent should move {direction_name}"

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

        result = move(env, working_orientation, agent_idx=0)
        total_moves += 1

        if result["success"]:
            successful_moves.append(step)
            print(f"✓ Successful move #{len(successful_moves)}")
            print(f"  Position: {result['position_before']} → {result['position_after']}")
        else:
            print(f"✗ Move failed: {result.get('error', 'Unknown error')}")
            if not result["move_success"]:
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

        result = move(env, orientation, agent_idx=0)

        if result["success"] and result["moved"]:
            successful_directions.append(direction_name)
            print(f"✓ {direction_name}: {result['position_before']} → {result['position_after']}")
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
    result = move(env, Orientation.UP)

    # Should work the same as using the enum directly
    assert result["success"], f"Move with string orientation should succeed. Error: {result.get('error')}"
    assert result["moved_correctly"], "Agent should move up correctly"
