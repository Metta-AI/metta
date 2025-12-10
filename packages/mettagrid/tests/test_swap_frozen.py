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

    # Verify initial positions
    assert pos_agent0_before == (1, 1), f"Agent 0 should be at (1, 1), got {pos_agent0_before}"
    assert pos_agent1_before == (1, 2), f"Agent 1 should be at (1, 2), got {pos_agent1_before}"

    # Agent 0 attacks Agent 1 to freeze them
    attack_result = attack(sim, target_arg=0, agent_idx=0)
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

    # Now agent 0 tries to move east onto frozen agent 1
    move_result = move(sim, Orientation.EAST, agent_idx=0)

    pos_agent0_after = get_agent_position(sim, 0)
    pos_agent1_after = get_agent_position(sim, 1)

    # Verify the swap happened
    assert move_result["success"], f"Move onto frozen agent should succeed (swap): {move_result}"
    assert pos_agent0_after == pos_agent1_before, (
        f"Agent 0 should now be at Agent 1's old position {pos_agent1_before}, got {pos_agent0_after}"
    )
    assert pos_agent1_after == pos_agent0_before, (
        f"Agent 1 should now be at Agent 0's old position {pos_agent0_before}, got {pos_agent1_after}"
    )


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

    # Without attacking first, try to move onto agent 1
    move_result = move(sim, Orientation.EAST, agent_idx=0)

    pos_agent0_after = get_agent_position(sim, 0)
    pos_agent1_after = get_agent_position(sim, 1)

    # Move should fail - can't move onto non-frozen agent
    assert not move_result["success"], "Move onto non-frozen agent should fail"
    assert pos_agent0_after == pos_agent0_before, "Agent 0 should not have moved"
    assert pos_agent1_after == pos_agent1_before, "Agent 1 should not have moved"


def test_swap_increments_stat(make_sim, adjacent_agents_map):
    """Test that swapping with a frozen agent increments the actions.swap stat."""
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

    # Freeze agent 1
    attack_result = attack(sim, target_arg=0, agent_idx=0)
    assert attack_result["success"], "Attack should succeed"

    # Get stats before swap
    swap_count_before = sim.episode_stats["agent"][0].get("actions.swap", 0)

    # Move onto frozen agent (swap)
    move_result = move(sim, Orientation.EAST, agent_idx=0)
    assert move_result["success"], "Swap should succeed"

    # Check that actions.swap was incremented
    swap_count_after = sim.episode_stats["agent"][0].get("actions.swap", 0)
    assert swap_count_after == swap_count_before + 1, (
        f"actions.swap should be incremented. Before: {swap_count_before}, After: {swap_count_after}"
    )


def test_multiple_swaps(make_sim):
    """Test multiple consecutive swaps with frozen agents."""
    # Map with three agents in a row
    three_agents_map = [
        ["wall", "wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "agent.blue", "agent.green", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall", "wall"],
    ]

    config_overrides = {
        "num_agents": 3,
        "agents": [
            {
                "team_id": 0,
                "freeze_duration": 10,
                "resource_limits": {"laser": {"limit": 10, "resources": ["laser"]}},
                "initial_inventory": {"laser": 10},
            },
            {
                "team_id": 1,
                "freeze_duration": 10,
                "resource_limits": {"laser": {"limit": 10, "resources": ["laser"]}},
                "initial_inventory": {"laser": 5},
            },
            {
                "team_id": 2,
                "freeze_duration": 10,
                "resource_limits": {"laser": {"limit": 10, "resources": ["laser"]}},
                "initial_inventory": {"laser": 5},
            },
        ],
    }

    sim = make_sim(three_agents_map, config_overrides)

    # Verify initial positions
    pos_agent0 = get_agent_position(sim, 0)
    pos_agent1 = get_agent_position(sim, 1)
    pos_agent2 = get_agent_position(sim, 2)

    assert pos_agent0 == (1, 1)
    assert pos_agent1 == (1, 2)
    assert pos_agent2 == (1, 3)

    # Agent 0 attacks agent 1 to freeze them
    attack_result = attack(sim, target_arg=0, agent_idx=0)
    assert attack_result["success"], "First attack should succeed"

    # Swap agent 0 with frozen agent 1
    move_result = move(sim, Orientation.EAST, agent_idx=0)
    assert move_result["success"], "First swap should succeed"

    # Check positions after first swap
    pos_agent0 = get_agent_position(sim, 0)
    pos_agent1 = get_agent_position(sim, 1)
    assert pos_agent0 == (1, 2), f"Agent 0 should be at (1, 2), got {pos_agent0}"
    assert pos_agent1 == (1, 1), f"Agent 1 should be at (1, 1), got {pos_agent1}"

    # Agent 0 attacks agent 2 to freeze them
    attack_result = attack(sim, target_arg=0, agent_idx=0)
    assert attack_result["success"], "Second attack should succeed"

    # Swap agent 0 with frozen agent 2
    move_result = move(sim, Orientation.EAST, agent_idx=0)
    assert move_result["success"], "Second swap should succeed"

    # Check final positions
    pos_agent0 = get_agent_position(sim, 0)
    pos_agent2 = get_agent_position(sim, 2)
    assert pos_agent0 == (1, 3), f"Agent 0 should be at (1, 3), got {pos_agent0}"
    assert pos_agent2 == (1, 2), f"Agent 2 should be at (1, 2), got {pos_agent2}"
