"""Tests for swapping positions with frozen agents."""

import pytest

from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    AttackActionConfig,
    AttackOutcome,
    ChangeVibeActionConfig,
    GameConfig,
    InventoryConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    ObsConfig,
    ResourceLimitsConfig,
    WallConfig,
)
from mettagrid.simulator import Simulation
from mettagrid.test_support.actions import get_agent_position, move
from mettagrid.test_support.map_builders import ObjectNameMapBuilder
from mettagrid.test_support.orientation import Orientation


def get_agent_frozen_status(sim: Simulation, agent_id: int) -> bool:
    """Check if an agent is frozen."""
    grid_objects = sim.grid_objects()
    for obj in grid_objects.values():
        if obj.get("agent_id") == agent_id:
            return obj.get("is_frozen", False)
    return False


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
            change_vibe=ChangeVibeActionConfig(),
            attack=AttackActionConfig(
                enabled=False,  # No standalone attack actions
                vibes=["charger"],  # Attack triggers on move when agent has charger vibe
                consumed_resources={"laser": 1},
                defense_resources={"armor": 1},
                success=AttackOutcome(freeze=10),
            ),
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
        game_config = base_config.model_copy(deep=True)

        if config_overrides:
            # Handle agents override - convert to new InventoryConfig format
            if "agents" in config_overrides:
                agents = []
                for agent_override in config_overrides["agents"]:
                    inventory = InventoryConfig(
                        initial=agent_override.get("initial_inventory", {}),
                        limits={
                            k: ResourceLimitsConfig(**v) if isinstance(v, dict) else v
                            for k, v in agent_override.get("resource_limits", {}).items()
                        },
                    )
                    agents.append(
                        AgentConfig(
                            team_id=agent_override.get("team_id", 0),
                            freeze_duration=agent_override.get("freeze_duration", 10),
                            inventory=inventory,
                        )
                    )
                game_config.agents = agents

            if "num_agents" in config_overrides:
                game_config.num_agents = config_overrides["num_agents"]

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
                "resource_limits": {"laser": ResourceLimitsConfig(limit=10, resources=["laser"])},
                "initial_inventory": {"laser": 5},
            },
            {
                "team_id": 1,
                "freeze_duration": 10,
                "resource_limits": {"laser": ResourceLimitsConfig(limit=10, resources=["laser"])},
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

    # Verify neither agent is frozen initially
    assert not get_agent_frozen_status(sim, 0), "Agent 0 should not start frozen"
    assert not get_agent_frozen_status(sim, 1), "Agent 1 should not start frozen"

    # Agent 0 changes vibe to "charger" to enable attack on move
    sim.agent(0).set_action("change_vibe_charger")
    sim.agent(1).set_action("noop")
    sim.step()

    # Agent 0 moves east into Agent 1 - attack triggers due to attacker vibe
    sim.agent(0).set_action("move_east")
    sim.agent(1).set_action("noop")
    sim.step()

    # Verify agent 1 is frozen (attack succeeded)
    assert get_agent_frozen_status(sim, 1), "Agent 1 should be frozen after attack"
    print(f"Agent 1 frozen status: {get_agent_frozen_status(sim, 1)}")

    # Positions should not have changed during attack (attacker stays in place)
    pos_agent0_after_attack = get_agent_position(sim, 0)
    pos_agent1_after_attack = get_agent_position(sim, 1)
    assert pos_agent0_after_attack == pos_agent0_before, "Agent 0 should still be at original position"
    assert pos_agent1_after_attack == pos_agent1_before, "Agent 1 should still be at original position"

    # Now agent 0 tries to move east onto frozen agent 1 - attack returns false on frozen target, enabling swap
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
                "resource_limits": {"laser": ResourceLimitsConfig(limit=10, resources=["laser"])},
                "initial_inventory": {"laser": 5},
            },
            {
                "team_id": 1,
                "freeze_duration": 10,
                "resource_limits": {"laser": ResourceLimitsConfig(limit=10, resources=["laser"])},
                "initial_inventory": {"laser": 5},
            },
        ],
    }

    sim = make_sim(adjacent_agents_map, config_overrides)

    pos_agent0_before = get_agent_position(sim, 0)
    pos_agent1_before = get_agent_position(sim, 1)
    print(f"Initial positions - Agent 0: {pos_agent0_before}, Agent 1: {pos_agent1_before}")

    # Make sure neither agent is frozen
    assert not get_agent_frozen_status(sim, 1), "Agent 1 should not be frozen initially"

    # Without attacking first, try to move onto agent 1 (no attacker vibe set)
    move_result = move(sim, Orientation.EAST, agent_idx=0)
    print(f"Move result: {move_result}")

    pos_agent0_after = get_agent_position(sim, 0)
    pos_agent1_after = get_agent_position(sim, 1)
    print(f"After move attempt - Agent 0: {pos_agent0_after}, Agent 1: {pos_agent1_after}")

    # Move should fail - can't move onto non-frozen agent (without attack triggering)
    assert not move_result["success"], "Move onto non-frozen agent should fail"
    assert pos_agent0_after == pos_agent0_before, "Agent 0 should not have moved"
    assert pos_agent1_after == pos_agent1_before, "Agent 1 should not have moved"

    print("✅ Cannot move onto non-frozen agent (as expected)")
