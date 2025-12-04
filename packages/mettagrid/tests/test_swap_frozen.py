"""Tests for swapping positions with frozen agents."""

import pytest

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
from mettagrid.simulator import Simulation
from mettagrid.test_support.actions import attack, get_agent_position, move
from mettagrid.test_support.map_builders import ObjectNameMapBuilder
from mettagrid.test_support.orientation import Orientation


@pytest.fixture
def base_config():
    """Base configuration for swap tests."""
    return GameConfig(
        max_steps=50,
        num_agents=2,
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
                ]
            ),
            attack=AttackActionConfig(enabled=True, consumed_resources={"laser": 1}, defense_resources={"armor": 1}),
        ),
        objects={
            "wall": WallConfig(),
        },
        agent=AgentConfig(rewards=AgentRewards()),
    )


@pytest.fixture
def adjacent_agents_map():
    """Map with two agents adjacent to each other."""
    return [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "agent.blue", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]


@pytest.fixture
def make_sim(base_config: GameConfig):
    """Factory fixture that creates a configured Simulation environment."""

    def _create_sim(game_map, config_overrides=None):
        game_config = base_config

        if config_overrides:
            config_dict = game_config.model_dump()

            if "obs" in config_dict and "features" in config_dict["obs"]:
                config_dict["obs"] = config_dict["obs"].copy()
                config_dict["obs"].pop("features", None)

            for key, value in config_overrides.items():
                if isinstance(value, dict) and key in config_dict and isinstance(config_dict[key], dict):
                    config_dict[key].update(value)
                else:
                    config_dict[key] = value

            game_config = GameConfig(**config_dict)

        cfg = MettaGridConfig(game=game_config)
        map_list = game_map.tolist() if hasattr(game_map, "tolist") else game_map
        cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=map_list)

        sim = Simulation(cfg, seed=42)
        return sim

    return _create_sim


def test_swap_with_frozen_agent(make_sim, adjacent_agents_map):
    """Test that an agent can swap positions with a frozen agent."""
    config_overrides = {
        "num_agents": 2,
        "agents": [
            {
                "team_id": 0,
                "freeze_duration": 10,
                "resource_limits": {"laser": {"limit": 10, "resources": ["laser"]}},
                "initial_inventory": {"laser": 5},
            },
            {
                "team_id": 1,
                "freeze_duration": 10,
                "resource_limits": {"laser": {"limit": 10, "resources": ["laser"]}},
                "initial_inventory": {"laser": 5},
            },
        ],
    }

    sim = make_sim(adjacent_agents_map, config_overrides)

    # Map layout:
    # ["wall", "agent.red", "agent.blue", "empty", "wall"],
    # Agent 0 (red) is at (1, 1), Agent 1 (blue) is at (1, 2)

    pos_agent0_before = get_agent_position(sim, 0)
    pos_agent1_before = get_agent_position(sim, 1)
    print(f"Initial positions - Agent 0: {pos_agent0_before}, Agent 1: {pos_agent1_before}")

    # Verify initial positions
    assert pos_agent0_before == (1, 1), f"Agent 0 should be at (1, 1), got {pos_agent0_before}"
    assert pos_agent1_before == (1, 2), f"Agent 1 should be at (1, 2), got {pos_agent1_before}"

    # Agent 0 attacks Agent 1 to freeze them
    attack_result = attack(sim, target_arg=0, agent_idx=0)
    print(f"Attack result: {attack_result}")
    assert attack_result["success"], f"Attack should succeed: {attack_result}"

    # Verify agent 1 is frozen
    grid_objects = sim.grid_objects()
    agent1_obj = None
    for _obj_id, obj in grid_objects.items():
        if obj["type_name"] == "agent":
            pos = (obj["r"], obj["c"])
            if pos == pos_agent1_before:
                agent1_obj = obj
                break

    assert agent1_obj is not None, "Should find agent 1"
    assert agent1_obj.get("is_frozen", False), f"Agent 1 should be frozen: {agent1_obj}"
    print(f"Agent 1 frozen status: {agent1_obj.get('is_frozen')}")

    # Now agent 0 tries to move east onto frozen agent 1
    move_result = move(sim, Orientation.EAST, agent_idx=0)
    print(f"Move result: {move_result}")

    pos_agent0_after = get_agent_position(sim, 0)
    pos_agent1_after = get_agent_position(sim, 1)
    print(f"After swap - Agent 0: {pos_agent0_after}, Agent 1: {pos_agent1_after}")

    # Verify the swap happened
    assert move_result["success"], f"Move onto frozen agent should succeed (swap): {move_result}"
    assert pos_agent0_after == pos_agent1_before, (
        f"Agent 0 should now be at Agent 1's old position {pos_agent1_before}, got {pos_agent0_after}"
    )
    assert pos_agent1_after == pos_agent0_before, (
        f"Agent 1 should now be at Agent 0's old position {pos_agent0_before}, got {pos_agent1_after}"
    )

    print("✅ Swap with frozen agent succeeded!")


def test_cannot_move_onto_non_frozen_agent(make_sim, adjacent_agents_map):
    """Test that an agent cannot move onto a non-frozen agent (no swap)."""
    config_overrides = {
        "num_agents": 2,
        "agents": [
            {
                "team_id": 0,
                "freeze_duration": 10,
                "resource_limits": {"laser": {"limit": 10, "resources": ["laser"]}},
                "initial_inventory": {"laser": 5},
            },
            {
                "team_id": 1,
                "freeze_duration": 10,
                "resource_limits": {"laser": {"limit": 10, "resources": ["laser"]}},
                "initial_inventory": {"laser": 5},
            },
        ],
    }

    sim = make_sim(adjacent_agents_map, config_overrides)

    pos_agent0_before = get_agent_position(sim, 0)
    pos_agent1_before = get_agent_position(sim, 1)
    print(f"Initial positions - Agent 0: {pos_agent0_before}, Agent 1: {pos_agent1_before}")

    # Without attacking first, try to move onto agent 1
    move_result = move(sim, Orientation.EAST, agent_idx=0)
    print(f"Move result: {move_result}")

    pos_agent0_after = get_agent_position(sim, 0)
    pos_agent1_after = get_agent_position(sim, 1)
    print(f"After move attempt - Agent 0: {pos_agent0_after}, Agent 1: {pos_agent1_after}")

    # Move should fail - can't move onto non-frozen agent
    assert not move_result["success"], "Move onto non-frozen agent should fail"
    assert pos_agent0_after == pos_agent0_before, "Agent 0 should not have moved"
    assert pos_agent1_after == pos_agent1_before, "Agent 1 should not have moved"

    print("✅ Cannot move onto non-frozen agent (as expected)")
