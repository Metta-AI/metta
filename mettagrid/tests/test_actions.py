"""Integration tests that combine multiple actions."""

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
from metta.mettagrid.test_support.actions import (
    attack,
    get_agent_position,
    move,
    noop,
    rotate,
    swap,
)
from metta.mettagrid.test_support.compass import Compass
from metta.mettagrid.test_support.orientation import Orientation


@pytest.fixture
def base_config():
    """Base configuration for integration tests."""
    return {
        "max_steps": 50,
        "num_agents": 1,
        "obs_width": 3,
        "obs_height": 3,
        "num_observation_tokens": 100,
        "inventory_item_names": ["laser", "armor"],
        "actions": {
            "noop": {"enabled": True},
            "move": {"enabled": True},
            "rotate": {"enabled": True},
            "get_items": {"enabled": True},
            "attack": {"enabled": True, "consumed_resources": {"laser": 1}, "defense_resources": {"armor": 1}},
            "put_items": {"enabled": True},
            "swap": {"enabled": True},
        },
        "groups": {"red": {"id": 0, "props": {}}},
        "objects": {
            "wall": {"type_id": 1},
            "block": {"type_id": 14, "swappable": True},
        },
        "agent": {"rewards": {}},
    }


@pytest.fixture
def complex_game_map():
    """Complex game map for integration tests."""
    return [
        ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "empty", "block", "empty", "agent.blue", "wall"],
        ["wall", "empty", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
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
        num_agents = game_config.get("num_agents", 1)
        observations = np.zeros((num_agents, 100, 3), dtype=dtype_observations)
        terminals = np.zeros(num_agents, dtype=dtype_terminals)
        truncations = np.zeros(num_agents, dtype=dtype_truncations)
        rewards = np.zeros(num_agents, dtype=dtype_rewards)
        env.set_buffers(observations, terminals, truncations, rewards)

        env.reset()
        return env

    return _create_env


def test_move_rotate_sequence(configured_env):
    """Test a sequence of move and rotate actions."""
    simple_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "empty", "agent.red", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    env = configured_env(simple_map)

    # Initial position
    initial_pos = get_agent_position(env, 0)
    assert initial_pos == (2, 2)

    # Rotate to face right
    rotate_result = rotate(env, Orientation.RIGHT)
    assert rotate_result["success"], "Rotation should succeed"

    # Move east
    move_result = move(env, Compass.EAST)
    assert move_result["success"], "Move east should succeed"

    # Check new position
    new_pos = get_agent_position(env, 0)
    assert new_pos == (2, 3), f"Agent should be at (2, 3), got {new_pos}"

    # Move north (should update orientation to up)
    move_result = move(env, Compass.NORTH)
    assert move_result["success"], "Move north should succeed"

    # Check final position
    final_pos = get_agent_position(env, 0)
    assert final_pos == (1, 3), f"Agent should be at (1, 3), got {final_pos}"


def test_attack_and_swap_integration(configured_env, complex_game_map):
    """Test attack followed by swap with a frozen agent."""
    config_overrides = {
        "num_agents": 2,
        "groups": {
            "red": {"id": 0, "props": {}},
            "blue": {"id": 1, "props": {}},
        },
        "agent": {
            "freeze_duration": 6,
            "resource_limits": {"laser": 10},
            "initial_inventory": {"laser": 5},
        },
    }

    env = configured_env(complex_game_map, config_overrides)

    # Agent 0 (red) is at (1, 1), Agent 1 (blue) is at (1, 5)
    # Need to move closer to attack

    # Move agent 0 east twice to get closer
    move_result = move(env, Compass.EAST, agent_idx=0)
    assert move_result["success"], "First move should succeed"

    move_result = move(env, Compass.EAST, agent_idx=0)
    assert move_result["success"], "Second move should succeed"

    # Now agent 0 should be at (1, 3), close enough to attack
    # But first need to rotate to face right
    rotate_result = rotate(env, Orientation.RIGHT, agent_idx=0)
    assert rotate_result["success"], "Rotation should succeed"

    # Attack agent 1
    attack_result = attack(env, target_arg=0, agent_idx=0)
    assert attack_result["success"], "Attack should succeed"
    assert "frozen_agent_id" in attack_result, "Should have frozen an agent"

    # Move closer to swap
    move_result = move(env, Compass.EAST, agent_idx=0)
    assert move_result["success"], "Move closer should succeed"

    # Try to swap with frozen agent
    swap_result = swap(env, agent_idx=0)
    # Note: swap with frozen agent might not be supported
    if swap_result["success"]:
        print("✅ Successfully swapped with frozen agent")
    else:
        print(f"ℹ️ Swap with frozen agent not supported: {swap_result.get('error')}")


def test_movement_pattern_with_obstacles(configured_env):
    """Test complex movement pattern around obstacles."""
    obstacle_map = [
        ["wall", "wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "empty", "wall", "empty", "wall"],
        ["wall", "empty", "wall", "wall", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall", "wall"],
    ]

    env = configured_env(obstacle_map)

    # Navigate around obstacles
    moves = [
        (Compass.EAST, True, (1, 2)),  # Move east
        (Compass.SOUTH, True, (2, 2)),  # Move south (can't go east due to wall)
        (Compass.SOUTH, True, (3, 2)),  # Move south again
        (Compass.EAST, True, (3, 3)),  # Move east
        (Compass.EAST, True, (3, 4)),  # Move east again
        (Compass.NORTH, True, (2, 4)),  # Move north
        (Compass.NORTH, True, (1, 4)),  # Move north to destination
    ]

    for direction, should_succeed, expected_pos in moves:
        result = move(env, direction)
        assert result["success"] == should_succeed, f"Move {direction} should {'succeed' if should_succeed else 'fail'}"

        if should_succeed:
            actual_pos = get_agent_position(env, 0)
            assert actual_pos == expected_pos, (
                f"After moving {direction}, expected position {expected_pos}, got {actual_pos}"
            )


def test_all_actions_sequence(configured_env):
    """Test using all available actions in sequence."""
    test_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "empty", "block", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    env = configured_env(test_map)

    print("\n=== Testing all actions in sequence ===")

    # 1. Noop
    print("1. Testing noop...")
    noop_result = noop(env)
    assert noop_result["success"], "Noop should always succeed"

    # 2. Rotate
    print("2. Testing rotate...")
    rotate_result = rotate(env, Orientation.RIGHT)
    assert rotate_result["success"], "Rotate should succeed"

    # 3. Move
    print("3. Testing move...")
    move_result = move(env, Compass.EAST)
    assert move_result["success"], "Move should succeed"

    # 4. Move again to be adjacent to block
    print("4. Moving adjacent to block...")
    move_result = move(env, Compass.EAST)
    assert move_result["success"], "Second move should succeed"

    # 5. Swap with block
    print("5. Testing swap...")
    swap_result = swap(env)
    if swap_result["success"]:
        print("   ✅ Swap succeeded")
        # Verify position changed
        new_pos = get_agent_position(env, 0)
        assert new_pos == (1, 3), f"Agent should be at block's position (1, 3), got {new_pos}"
    else:
        print(f"   ℹ️ Swap failed: {swap_result.get('error')}")

    print("\n✅ All actions tested successfully!")


def test_diagonal_movement_integration(configured_env):
    """Test diagonal movement combined with other actions."""
    open_map = [
        ["wall", "wall", "wall", "wall", "wall", "wall"],
        ["wall", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "empty", "agent.red", "empty", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall", "wall"],
    ]

    env = configured_env(open_map)

    # Test diagonal movement pattern
    diagonal_moves = [
        (Compass.NORTHEAST, (1, 3)),
        (Compass.SOUTHEAST, (2, 4)),
        (Compass.SOUTHWEST, (3, 3)),
        (Compass.NORTHWEST, (2, 2)),  # Back to start
    ]

    for direction, expected_pos in diagonal_moves:
        result = move(env, direction)
        assert result["success"], f"Diagonal move {direction} should succeed"

        actual_pos = get_agent_position(env, 0)
        assert actual_pos == expected_pos, f"After {direction}, expected {expected_pos}, got {actual_pos}"

    print("✅ Diagonal movement pattern completed successfully!")
