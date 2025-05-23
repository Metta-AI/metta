import numpy as np
import pytest

from mettagrid.mettagrid_c import MettaGrid
from mettagrid.tests.actions import (
    get_agent_position,
    move,
    np_observations_type,
    np_rewards_type,
    np_terminals_type,
    np_truncations_type,
)


@pytest.fixture
def base_config():
    """Base configuration for MettaGrid tests."""
    return {
        "max_steps": 50,
        "num_agents": 1,
        "obs_width": 3,
        "obs_height": 3,
        "actions": {
            "noop": {"enabled": True},
            "move": {"enabled": True},
            "rotate": {"enabled": True},
            "get_items": {"enabled": True},  # maps to get_output
            "attack": {"enabled": True},
            "put_items": {"enabled": True},  # maps to get_recipe_items
            "swap": {"enabled": True},
            "change_color": {"enabled": True},
        },
        "groups": {"red": {"id": 0, "props": {}}},
        "objects": {"wall": {"type_id": 1, "hp": 100}},
        "agent": {"inventory_size": 10, "hp": 100},
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
        config = base_config.copy()
        if config_overrides:
            config.update(config_overrides)

        env_config = {"game": config}
        env = MettaGrid(env_config, game_map)

        # Set up buffers
        num_features = len(env.grid_features())
        observations = np.zeros((1, 3, 3, num_features), dtype=np_observations_type)
        terminals = np.zeros(1, dtype=np_terminals_type)
        truncations = np.zeros(1, dtype=np_truncations_type)
        rewards = np.zeros(1, dtype=np_rewards_type)
        env.set_buffers(observations, terminals, truncations, rewards)

        env.reset()
        return env

    return _create_env


def test_move_all_directions(configured_env, movement_game_map):
    """Test the move function in all four directions."""
    env = configured_env(movement_game_map)

    initial_pos = get_agent_position(env)
    assert initial_pos is not None, "Agent should have a valid initial position"

    # Test all 4 directions
    directions = [(0, "up"), (3, "right"), (1, "down"), (2, "left")]

    for orientation, direction_name in directions:
        print(f"Testing move {direction_name} (orientation {orientation})")
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

    result = move(env, 0)  # Move up

    assert result["success"], f"Move up should succeed. Error: {result.get('error')}"
    assert result["moved_correctly"], "Agent should move up correctly"
    assert result["position_before"][0] - result["position_after"][0] == 1, "Should move up by 1 row"


def test_move_blocked_by_wall(configured_env, blocked_game_map):
    """Test that movement is properly blocked by walls."""
    env = configured_env(blocked_game_map, {"max_steps": 10})

    # Try to move in any direction - should be blocked
    for orientation, direction_name in [(0, "up"), (1, "down"), (2, "left"), (3, "right")]:
        result = move(env, orientation)

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
    moves = [(0, "up"), (3, "right"), (1, "down"), (2, "left")]

    for orientation, direction_name in moves:
        result = move(env, orientation)
        assert result["success"], f"Move {direction_name} should succeed"
        assert result["moved"], f"Agent should move {direction_name}"

    # Should be back at original position
    final_pos = get_agent_position(env)
    assert final_pos == initial_pos, f"Agent should return to original position {initial_pos}, but is at {final_pos}"
