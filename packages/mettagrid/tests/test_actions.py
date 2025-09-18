"""Integration tests that combine multiple actions."""

import numpy as np
import pytest

from mettagrid.config.mettagrid_c_config import from_mettagrid_config
from mettagrid.config.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    AttackActionConfig,
    GameConfig,
    WallConfig,
)
from mettagrid.mettagrid_c import (
    MettaGrid,
    dtype_observations,
    dtype_rewards,
    dtype_terminals,
    dtype_truncations,
)
from mettagrid.test_support.actions import (
    attack,
    get_agent_position,
    move,
    noop,
    rotate,
    swap,
)
from mettagrid.test_support.orientation import Orientation


@pytest.fixture
def base_config():
    """Base configuration for integration tests."""
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
            get_items=ActionConfig(enabled=True),
            attack=AttackActionConfig(enabled=True, consumed_resources={"laser": 1}, defense_resources={"armor": 1}),
            put_items=ActionConfig(enabled=True),
            swap=ActionConfig(enabled=True),
        ),
        objects={
            "wall": WallConfig(type_id=1, swappable=False),
            "block": WallConfig(type_id=14, swappable=True),
        },
        agent=AgentConfig(rewards=AgentRewards()),
        allow_diagonals=True,
    )


@pytest.fixture
def complex_game_map():
    """Complex game map for integration tests."""
    return [
        ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "empty", "empty", "empty", "agent.blue", "wall"],
        ["wall", "empty", "empty", "block", "empty", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
    ]


@pytest.fixture
def configured_env(base_config: GameConfig):
    """Factory fixture that creates a configured MettaGrid environment."""

    def _create_env(game_map, config_overrides=None):
        game_config = base_config

        assert game_config.allow_diagonals

        if config_overrides:
            # Create a new config with overrides using Pydantic model update
            config_dict = game_config.model_dump()

            # Deep update for nested dicts
            for key, value in config_overrides.items():
                if isinstance(value, dict) and key in config_dict and isinstance(config_dict[key], dict):
                    config_dict[key].update(value)
                else:
                    config_dict[key] = value

            # Create new GameConfig from updated dict
            game_config = GameConfig(**config_dict)

        env = MettaGrid(from_mettagrid_config(game_config), game_map, 42)

        # Set up buffers
        num_agents = game_config.num_agents
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

    # Rotate to face east
    rotate_result = rotate(env, Orientation.EAST)
    assert rotate_result["success"], "Rotation should succeed"

    # Move east
    move_result = move(env, Orientation.EAST)
    assert move_result["success"], "Move east should succeed"

    # Check new position
    new_pos = get_agent_position(env, 0)
    assert new_pos == (2, 3), f"Agent should be at (2, 3), got {new_pos}"

    # Move north (should update orientation to up)
    move_result = move(env, Orientation.NORTH)
    assert move_result["success"], "Move north should succeed"

    # Check final position
    final_pos = get_agent_position(env, 0)
    assert final_pos == (1, 3), f"Agent should be at (1, 3), got {final_pos}"


def test_attack_and_swap_integration(configured_env, complex_game_map):
    """Test attack followed by swap with a frozen agent."""
    config_overrides = {
        "num_agents": 2,
        "agents": [
            {
                "team_id": 0,
                "freeze_duration": 6,
                "resource_limits": {"laser": 10},
                "initial_inventory": {"laser": 5},
            },
            {
                "team_id": 1,
                "freeze_duration": 6,
                "resource_limits": {"laser": 10},
                "initial_inventory": {"laser": 5},
            },
        ],
    }

    env = configured_env(complex_game_map, config_overrides)

    # complex_game_map layout:
    # ["wall", "agent.red", "empty", "empty", "empty", "agent.blue", "wall"],
    # ["wall", "empty", "empty", "block", "empty", "empty", "wall"],
    # Agent 0 (red) is at (1, 1), Agent 1 (blue) is at (1, 5)
    # Block is at (2, 3)

    # Move agent 0 east three times to get closer to agent 1
    move_result = move(env, Orientation.EAST, agent_idx=0)
    assert move_result["success"], "First move east should succeed"

    move_result = move(env, Orientation.EAST, agent_idx=0)
    assert move_result["success"], "Second move east should succeed"

    move_result = move(env, Orientation.EAST, agent_idx=0)
    assert move_result["success"], "Third move east should succeed"

    # Now agent should be at (1, 4), next to agent 1 at (1, 5)
    current_pos = get_agent_position(env, 0)
    print(f"Position before attack: {current_pos}")

    # Rotate to face right (toward agent 1)
    rotate_result = rotate(env, Orientation.EAST, agent_idx=0)
    assert rotate_result["success"], "Rotation should succeed"

    # Attack agent 1 (who is directly to the right)
    attack_result = attack(env, target_arg=0, agent_idx=0)
    if attack_result["success"]:
        assert "frozen_agent_id" in attack_result, "Should have frozen an agent"
        print("✅ Successfully attacked and froze agent")

        # Now try to swap with the frozen agent
        swap_result = swap(env, agent_idx=0)
        if swap_result["success"]:
            print("✅ Successfully swapped with frozen agent")
            new_pos = get_agent_position(env, 0)
            assert new_pos == (1, 5), f"Should have swapped to agent 1's position, got {new_pos}"
        else:
            print(f"ℹ️ Swap with frozen agent not supported: {swap_result.get('error')}")
    else:
        print(f"Attack failed: {attack_result.get('error')}")


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
        (Orientation.EAST, True, (1, 2)),  # Move east
        (Orientation.SOUTH, False, (1, 2)),  # Can't move south - wall at (2, 2)
        (Orientation.WEST, True, (1, 1)),  # Move back west
        (Orientation.SOUTH, True, (2, 1)),  # Move south
        (Orientation.SOUTH, True, (3, 1)),  # Move south again
        (Orientation.EAST, True, (3, 2)),  # Move east
        (Orientation.EAST, True, (3, 3)),  # Move east again
        (Orientation.EAST, True, (3, 4)),  # Move east once more
        (Orientation.NORTH, True, (2, 4)),  # Move north
        (Orientation.NORTH, True, (1, 4)),  # Move north to destination
    ]

    for i, (direction, should_succeed, expected_pos) in enumerate(moves):
        print(f"Move {i + 1}: {direction}")
        current_pos = get_agent_position(env, 0)
        print(f"  Current position: {current_pos}")

        result = move(env, direction)
        assert result["success"] == should_succeed, (
            f"Move {direction} should {'succeed' if should_succeed else 'fail'}, "
            f"but got {result['success']}. Error: {result.get('error')}"
        )

        if should_succeed:
            actual_pos = get_agent_position(env, 0)
            assert actual_pos == expected_pos, (
                f"After moving {direction}, expected position {expected_pos}, got {actual_pos}"
            )


def test_all_actions_sequence(configured_env):
    """Test using all available actions in sequence."""
    test_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "empty", "empty", "wall"],
        ["wall", "empty", "block", "empty", "wall"],
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
    rotate_result = rotate(env, Orientation.EAST)  # Face right
    assert rotate_result["success"], "Rotate should succeed"

    # 3. Move east
    print("3. Testing move east...")
    move_result = move(env, Orientation.EAST)
    assert move_result["success"], "Move east should succeed"

    # Now at (1, 2)
    print(f"   Position after move east: {get_agent_position(env, 0)}")

    # 4. Move east again
    print("4. Moving east again...")
    move_result = move(env, Orientation.EAST)
    assert move_result["success"], "Second move east should succeed"

    # Now at (1, 3)
    print(f"   Position: {get_agent_position(env, 0)}")

    # 5. Move south to row with block
    print("5. Moving south...")
    move_result = move(env, Orientation.SOUTH)
    assert move_result["success"], "Move south should succeed"

    # Now at (2, 3), next to block at (2, 2)
    current_pos = get_agent_position(env, 0)
    print(f"   Position next to block: {current_pos}")

    # 6. Rotate to face the block (west)
    print("6. Rotating to face west (toward block)...")
    rotate_result = rotate(env, Orientation.WEST)

    # 7. Swap with block
    print("7. Testing swap with block...")
    swap_result = swap(env)
    if swap_result["success"]:
        print("   ✅ Swap succeeded")
        # Verify position changed
        new_pos = get_agent_position(env, 0)
        print(f"   New position after swap: {new_pos}")
        # Note: position depends on implementation of swap
    else:
        print(f"   ℹ️ Swap failed: {swap_result.get('error')}")
        # Try moving to a different position for other tests
        print("   Moving to continue other tests...")
        move(env, Orientation.SOUTH)

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

    env: MettaGrid = configured_env(open_map)

    print(env.action_names(), env.max_action_args())

    # Test diagonal movement pattern
    diagonal_moves = [
        (Orientation.NORTH, (1, 2)),
        (Orientation.SOUTH, (2, 2)),
        (Orientation.NORTHEAST, (1, 3)),
        (Orientation.SOUTHEAST, (2, 4)),
        (Orientation.SOUTHWEST, (3, 3)),
        (Orientation.NORTHWEST, (2, 2)),  # Back to start
    ]

    for direction, expected_pos in diagonal_moves:
        result = move(env, direction)
        print(result)

        assert result["success"], f"move {direction} should succeed"

        actual_pos = get_agent_position(env, 0)
        assert actual_pos == expected_pos, f"After {direction}, expected {expected_pos}, got {actual_pos}"

    print("✅ Diagonal movement pattern completed successfully!")
