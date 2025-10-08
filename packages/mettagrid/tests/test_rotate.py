"""Tests for rotation functionality."""

import numpy as np
import pytest

from mettagrid.config.mettagrid_c_config import from_mettagrid_config
from mettagrid.config.mettagrid_config import ActionConfig, ActionsConfig, AgentConfig, GameConfig, WallConfig
from mettagrid.mettagrid_c import (
    MettaGrid,
    dtype_observations,
    dtype_rewards,
    dtype_terminals,
    dtype_truncations,
)
from mettagrid.test_support import Orientation
from mettagrid.test_support.actions import get_agent_orientation, rotate


@pytest.fixture
def base_config() -> GameConfig:
    """Base configuration for rotation tests."""
    return GameConfig(
        max_steps=50,
        num_agents=1,
        obs_width=3,
        obs_height=3,
        num_observation_tokens=100,
        resource_names=[],
        actions=ActionsConfig(
            noop=ActionConfig(enabled=True),
            rotate=ActionConfig(enabled=True),
        ),
        objects={
            "wall": WallConfig(type_id=1),
        },
        allow_diagonals=True,
    )


@pytest.fixture
def simple_game_map():
    """Simple game map for rotation tests."""
    return [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "empty", "agent.red", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]


@pytest.fixture
def configured_env(base_config):
    """Factory fixture that creates a configured MettaGrid environment."""

    def _create_env(game_map, game_config: GameConfig | None = None):
        if game_config is None:
            game_config = base_config

        env = MettaGrid(from_mettagrid_config(game_config), game_map, 42)

        # Set up buffers based on number of agents
        num_agents = game_config.num_agents
        observations = np.zeros((num_agents, 100, 3), dtype=dtype_observations)
        terminals = np.zeros(num_agents, dtype=dtype_terminals)
        truncations = np.zeros(num_agents, dtype=dtype_truncations)
        rewards = np.zeros(num_agents, dtype=dtype_rewards)
        env.set_buffers(observations, terminals, truncations, rewards)

        env.reset()
        return env

    return _create_env


def test_rotation_functionality(configured_env, simple_game_map):
    """Test that rotation works correctly for all 8 orientations."""
    env = configured_env(simple_game_map)

    # Test all 8 orientations
    orientations = [
        Orientation.NORTH,
        Orientation.SOUTH,
        Orientation.WEST,
        Orientation.EAST,
        Orientation.NORTHWEST,
        Orientation.NORTHEAST,
        Orientation.SOUTHWEST,
        Orientation.SOUTHEAST,
    ]

    for orientation in orientations:
        result = rotate(env, orientation)
        assert result["success"], f"Rotation to {orientation} should succeed"
        assert result["orientation_after"] == orientation.value, (
            f"Agent should be facing {orientation} (value {orientation.value})"
        )

        # Verify with get_agent_orientation
        current = get_agent_orientation(env, 0)
        assert current == orientation.value, (
            f"get_agent_orientation should return {orientation.value} for {orientation}, got {current}"
        )


def test_rotation_cycle(configured_env, simple_game_map):
    """Test rotating through all orientations in a cycle."""
    env = configured_env(simple_game_map)

    # Start by setting a known orientation
    initial_result = rotate(env, Orientation.NORTH)
    assert initial_result["success"], "Initial rotation should succeed"

    # Rotate through all orientations
    orientations = [
        Orientation.EAST,
        Orientation.SOUTH,
        Orientation.WEST,
        Orientation.NORTH,
    ]

    for orientation in orientations:
        result = rotate(env, orientation)
        assert result["success"], f"Rotation to {orientation} should succeed"
        assert result["orientation_after"] == orientation.value

        # Double-check with get_agent_orientation
        current_orientation = get_agent_orientation(env, 0)
        assert current_orientation == orientation.value, (
            f"get_agent_orientation should return {orientation.value}, got {current_orientation}"
        )


def test_rotation_preserves_position(configured_env, simple_game_map):
    """Test that rotation doesn't change agent position."""
    env = configured_env(simple_game_map)

    # Get initial position
    from mettagrid.test_support.actions import get_agent_position

    initial_pos = get_agent_position(env, 0)

    # Rotate through all orientations
    orientations = [
        Orientation.NORTH,
        Orientation.EAST,
        Orientation.SOUTH,
        Orientation.WEST,
    ]

    for orientation in orientations:
        rotate(env, orientation)
        current_pos = get_agent_position(env, 0)
        assert current_pos == initial_pos, (
            f"Position should not change during rotation. Expected {initial_pos}, got {current_pos}"
        )


def test_multiple_agents_rotation(configured_env, base_config):
    """Test rotation with multiple agents."""
    # Create a map with multiple agents
    multi_agent_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "empty", "agent.blue", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    game_config = base_config.model_copy(
        update={
            "num_agents": 2,
            "agents": [
                AgentConfig(team_id=0),  # red
                AgentConfig(team_id=1),  # blue
            ],
        }
    )

    env = configured_env(multi_agent_map, game_config)

    # Rotate each agent to different orientations
    result0 = rotate(env, Orientation.EAST, agent_idx=0)
    result1 = rotate(env, Orientation.SOUTH, agent_idx=1)

    assert result0["success"], "Agent 0 rotation should succeed"
    assert result1["success"], "Agent 1 rotation should succeed"

    # Verify each agent has the correct orientation
    orientation0 = get_agent_orientation(env, 0)
    orientation1 = get_agent_orientation(env, 1)

    assert orientation0 == Orientation.EAST.value, f"Agent 0 should face RIGHT, got {orientation0}"
    assert orientation1 == Orientation.SOUTH.value, f"Agent 1 should face DOWN, got {orientation1}"


def test_rotation_helper_return_values(configured_env, simple_game_map):
    """Test that the rotate helper returns expected values."""
    env = configured_env(simple_game_map)

    # Get initial orientation
    initial_orientation = get_agent_orientation(env, 0)

    # Rotate to a different orientation
    target_orientation = Orientation.EAST if initial_orientation != 3 else Orientation.WEST
    result = rotate(env, target_orientation)

    # Check all return values
    assert "success" in result
    assert "action_success" in result
    assert "orientation_before" in result
    assert "orientation_after" in result
    assert "rotated_correctly" in result
    assert "error" in result
    assert "direction" in result
    assert "target_orientation" in result

    # Verify values are correct
    assert result["success"] is True
    assert result["action_success"] is True
    assert result["orientation_before"] == initial_orientation
    assert result["orientation_after"] == target_orientation.value
    assert result["rotated_correctly"] is True
    assert result["error"] is None
    assert result["direction"] == str(target_orientation)
    assert result["target_orientation"] == target_orientation.value


def test_rotation_without_rotate_action(configured_env, simple_game_map, base_config):
    """Test that rotation fails gracefully when rotate action is not available."""
    game_config = base_config.model_copy(
        update={
            "actions": ActionsConfig(
                noop=ActionConfig(enabled=True),
                # rotate is not enabled
            )
        }
    )

    env = configured_env(simple_game_map, game_config)

    result = rotate(env, Orientation.EAST)

    assert result["success"] is False
    assert result["error"] == "rotate_east action not available"
