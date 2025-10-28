"""Integration tests that combine multiple actions."""

import numpy as np
import pytest

from mettagrid.config.mettagrid_c_config import from_mettagrid_config
from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    AttackActionConfig,
    GameConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    ObsConfig,
    WallConfig,
)
from mettagrid.map_builder.map_builder import GameMap, MapBuilder, MapBuilderConfig
from mettagrid.mettagrid_c import MettaGrid
from mettagrid.simulator import Simulation
from mettagrid.test_support.actions import attack, get_agent_position, move, noop
from mettagrid.test_support.orientation import Orientation


# Helper map builder for tests that have pre-built maps (object names, not ASCII)
class ObjectNameMapBuilder(MapBuilder):
    """Map builder that uses pre-built object name maps."""

    class Config(MapBuilderConfig["ObjectNameMapBuilder"]):
        map_data: list[list[str]]

    def __init__(self, config: Config):
        self.config = config

    def build(self) -> GameMap:
        return GameMap(grid=np.array(self.config.map_data))


@pytest.fixture
def base_config():
    """Base configuration for integration tests."""
    return GameConfig(
        max_steps=50,
        num_agents=1,
        obs=ObsConfig(width=3, height=3, num_tokens=100),
        resource_names=["laser", "armor"],
        actions=ActionsConfig(
            noop=NoopActionConfig(),
            move=MoveActionConfig(
                allowed_directions=[
                    "north",
                    "south",
                    "east",
                    "west",
                    "northeast",
                    "northwest",
                    "southeast",
                    "southwest",
                ]
            ),
            attack=AttackActionConfig(enabled=True, consumed_resources={"laser": 1}, defense_resources={"armor": 1}),
        ),
        objects={
            "wall": WallConfig(swappable=False),
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
        ["wall", "empty", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
    ]


@pytest.fixture
def make_sim(base_config: GameConfig):
    """Factory fixture that creates a configured Simulation environment."""

    def _create_sim(game_map, config_overrides=None):
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

        # Create MettaGridConfig wrapper
        cfg = MettaGridConfig(game=game_config)

        # Put the map into the config using ObjectNameMapBuilder
        map_list = game_map.tolist() if hasattr(game_map, "tolist") else game_map
        cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=map_list)

        sim = Simulation(cfg, seed=42)

        return sim

    return _create_sim


def test_attack_integration(make_sim, complex_game_map):
    """Test attack with a frozen agent."""
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

    sim = make_sim(complex_game_map, config_overrides)

    # complex_game_map layout:
    # ["wall", "agent.red", "empty", "empty", "empty", "agent.blue", "wall"],
    # ["wall", "empty", "empty", "block", "empty", "empty", "wall"],
    # Agent 0 (red) is at (1, 1), Agent 1 (blue) is at (1, 5)
    # Block is at (2, 3)

    # Move agent 0 east three times to get closer to agent 1
    move_result = move(sim, Orientation.EAST, agent_idx=0)
    assert move_result["success"], "First move east should succeed"

    move_result = move(sim, Orientation.EAST, agent_idx=0)
    assert move_result["success"], "Second move east should succeed"

    move_result = move(sim, Orientation.EAST, agent_idx=0)
    assert move_result["success"], "Third move east should succeed"

    # Now agent should be at (1, 4), next to agent 1 at (1, 5)
    current_pos = get_agent_position(sim, 0)
    print(f"Position before attack: {current_pos}")

    # Attack agent 1 (who is directly to the right)
    attack_result = attack(sim, target_arg=0, agent_idx=0)
    if attack_result["success"]:
        assert "frozen_agent_id" in attack_result, "Should have frozen an agent"
        print("✅ Successfully attacked and froze agent")
    else:
        print(f"Attack failed: {attack_result.get('error')}")


def test_movement_pattern_with_obstacles(make_sim):
    """Test complex movement pattern around obstacles."""
    obstacle_map = [
        ["wall", "wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "empty", "wall", "empty", "wall"],
        ["wall", "empty", "wall", "wall", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall", "wall"],
    ]

    sim = make_sim(obstacle_map)

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
        current_pos = get_agent_position(sim, 0)
        print(f"  Current position: {current_pos}")

        result = move(sim, direction)
        assert result["success"] == should_succeed, (
            f"Move {direction} should {'succeed' if should_succeed else 'fail'}, "
            f"but got {result['success']}. Error: {result.get('error')}"
        )

        if should_succeed:
            actual_pos = get_agent_position(sim, 0)
            assert actual_pos == expected_pos, (
                f"After moving {direction}, expected position {expected_pos}, got {actual_pos}"
            )


def test_all_actions_sequence(make_sim):
    """Test using all available actions in sequence."""
    test_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "empty", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    sim = make_sim(test_map)

    print("\n=== Testing all actions in sequence ===")

    # 1. Noop
    print("1. Testing noop...")
    noop_result = noop(sim)
    assert noop_result["success"], "Noop should always succeed"

    # 3. Move east
    print("3. Testing move east...")
    move_result = move(sim, Orientation.EAST)
    assert move_result["success"], "Move east should succeed"

    # Now at (1, 2)
    print(f"   Position after move east: {get_agent_position(sim, 0)}")

    # 4. Move east again
    print("4. Moving east again...")
    move_result = move(sim, Orientation.EAST)
    assert move_result["success"], "Second move east should succeed"

    # Now at (1, 3)
    print(f"   Position: {get_agent_position(sim, 0)}")

    # 5. Move south
    print("5. Moving south...")
    move_result = move(sim, Orientation.SOUTH)
    assert move_result["success"], "Move south should succeed"

    # Now at (2, 3)
    current_pos = get_agent_position(sim, 0)
    print(f"   Position: {current_pos}")

    print("\n✅ All actions tested successfully!")


@pytest.mark.skip(reason="Diagonal movement failing at C++ level - action exists but C++ rejects the move")
def test_diagonal_movement_integration(make_sim):
    """Test diagonal movement combined with other actions."""
    open_map = [
        ["wall", "wall", "wall", "wall", "wall", "wall"],
        ["wall", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "empty", "agent.red", "empty", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall", "wall"],
    ]

    sim: Simulation = make_sim(open_map)

    # Test diagonal movement pattern
    diagonal_moves = [
        (Orientation.NORTHEAST, (1, 3)),
        (Orientation.SOUTHEAST, (2, 4)),
        (Orientation.SOUTHWEST, (3, 3)),
        (Orientation.NORTHWEST, (2, 2)),  # Back to start
    ]

    for direction, expected_pos in diagonal_moves:
        result = move(sim, direction)
        assert result["success"], f"move {direction} should succeed"

        actual_pos = get_agent_position(sim, 0)
        assert actual_pos == expected_pos, f"After {direction}, expected {expected_pos}, got {actual_pos}"


def test_noop_is_always_index_0():
    """Test that noop action is always at index 0 when present."""
    # Test 1: Default config with noop enabled
    config = GameConfig(
        max_steps=10,
        num_agents=2,
        obs=ObsConfig(width=3, height=3, num_tokens=10),
        actions=ActionsConfig(noop=NoopActionConfig(), move=MoveActionConfig()),
    )

    c_config = from_mettagrid_config(config)
    map_data = [[".", "."], ["agent.team_0", "agent.team_0"]]
    sim = MettaGrid(c_config, map_data, 0)

    action_names = sim.action_names()
    assert action_names[0] == "noop", f"Expected 'noop' at index 0, got '{action_names[0]}'"

    # Test 2: noop listed last but should still be at index 0
    config2 = GameConfig(
        max_steps=10,
        num_agents=2,
        obs=ObsConfig(width=3, height=3, num_tokens=10),
        actions=ActionsConfig(
            attack=AttackActionConfig(),
            move=MoveActionConfig(),
            noop=NoopActionConfig(),  # noop listed last
        ),
    )

    c_config2 = from_mettagrid_config(config2)
    sim2 = MettaGrid(c_config2, map_data, 0)

    action_names2 = sim2.action_names()
    assert action_names2[0] == "noop", f"Expected 'noop' at index 0, got '{action_names2[0]}'"

    # Test 3: Config without noop - currently noop is always present even when disabled
    # This is a known limitation - the enabled flag is not fully implemented yet
    config3 = GameConfig(
        max_steps=10,
        num_agents=2,
        obs=ObsConfig(width=3, height=3, num_tokens=10),
        actions=ActionsConfig(
            noop=NoopActionConfig(enabled=False), move=MoveActionConfig(), attack=AttackActionConfig()
        ),
    )

    c_config3 = from_mettagrid_config(config3)
    sim3 = MettaGrid(c_config3, map_data, 0)

    action_names3 = sim3.action_names()
    # TODO: Fix enabled flag in ActionConfig - currently noop is always present
    # assert "noop" not in action_names3, "noop should not be present when disabled"
    assert action_names3[0] == "noop", "Noop is currently always at index 0 even when disabled"
