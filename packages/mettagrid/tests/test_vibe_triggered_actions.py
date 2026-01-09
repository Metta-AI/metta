"""Tests for vibe-triggered attack on move."""

from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AgentConfig,
    AttackActionConfig,
    AttackOutcome,
    ChangeVibeActionConfig,
    GameConfig,
    InventoryConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    ObsConfig,
    WallConfig,
)
from mettagrid.simulator import Simulation
from mettagrid.test_support.map_builders import ObjectNameMapBuilder


def get_agent_position(sim: Simulation, agent_id: int) -> tuple | None:
    """Get an agent's position as (row, col)."""
    grid_objects = sim.grid_objects()
    for obj in grid_objects.values():
        if obj.get("agent_id") == agent_id:
            return (obj["r"], obj["c"])
    return None


def get_agent_frozen_status(sim: Simulation, agent_id: int) -> bool:
    """Check if an agent is frozen."""
    grid_objects = sim.grid_objects()
    for obj in grid_objects.values():
        if obj.get("agent_id") == agent_id:
            return obj.get("is_frozen", False)
    return False


class TestVibeTriggeredAttack:
    """Tests for attack-on-move triggered by agent vibe."""

    def test_attack_triggers_on_move_with_matching_vibe(self):
        """When an agent with a weapon vibe moves into another agent, attack is triggered."""
        map_data = [
            ["wall", "wall", "wall", "wall"],
            ["wall", "agent.red", "agent.blue", "wall"],
            ["wall", "wall", "wall", "wall"],
        ]

        config = GameConfig(
            max_steps=100,
            num_agents=2,
            obs=ObsConfig(width=5, height=5, num_tokens=50),
            resource_names=["energy", "heart"],
            actions=ActionsConfig(
                noop=NoopActionConfig(),
                move=MoveActionConfig(),
                change_vibe=ChangeVibeActionConfig(),
                attack=AttackActionConfig(
                    enabled=False,  # Disable direct attack actions
                    vibes=["charger"],  # Attack triggers on move when agent has charger vibe
                    success=AttackOutcome(freeze=5),
                ),
            ),
            objects={"wall": WallConfig()},
            agents=[
                AgentConfig(
                    team_id=0,
                    inventory=InventoryConfig(initial={"energy": 10, "heart": 5}),
                ),
                AgentConfig(
                    team_id=1,
                    inventory=InventoryConfig(initial={"energy": 10, "heart": 5}),
                ),
            ],
        )

        mg_config = MettaGridConfig(game=config)
        mg_config.game.map_builder = ObjectNameMapBuilder.Config(map_data=map_data)
        sim = Simulation(mg_config, seed=42)

        # Verify initial state - neither agent is frozen
        assert not get_agent_frozen_status(sim, 0), "Agent 0 should not start frozen"
        assert not get_agent_frozen_status(sim, 1), "Agent 1 should not start frozen"

        # Agent 0 changes vibe to charger (vibe index 1 = "charger")
        sim.agent(0).set_action("change_vibe_charger")
        sim.agent(1).set_action("noop")
        sim.step()

        # Agent 0 moves into Agent 1 (east) - should trigger attack due to weapon vibe
        sim.agent(0).set_action("move_east")
        sim.agent(1).set_action("noop")
        sim.step()

        # Agent 0 should not have moved (target occupied), but attack should have happened
        pos0 = get_agent_position(sim, 0)
        pos1 = get_agent_position(sim, 1)
        assert pos0 == (1, 1), f"Agent 0 should still be at (1,1), got {pos0}"
        assert pos1 == (1, 2), f"Agent 1 should still be at (1,2), got {pos1}"

        # Agent 1 should be frozen from the attack
        assert get_agent_frozen_status(sim, 1), "Agent 1 should be frozen after attack"

    def test_no_attack_without_matching_vibe(self):
        """When an agent without weapon vibe moves into another agent, no attack happens."""
        map_data = [
            ["wall", "wall", "wall", "wall"],
            ["wall", "agent.red", "agent.blue", "wall"],
            ["wall", "wall", "wall", "wall"],
        ]

        config = GameConfig(
            max_steps=100,
            num_agents=2,
            obs=ObsConfig(width=5, height=5, num_tokens=50),
            resource_names=["energy", "heart"],
            actions=ActionsConfig(
                noop=NoopActionConfig(),
                move=MoveActionConfig(),
                change_vibe=ChangeVibeActionConfig(),
                attack=AttackActionConfig(
                    enabled=False,
                    vibes=["charger"],  # Attack only triggers with charger vibe
                ),
            ),
            objects={"wall": WallConfig()},
            agents=[
                AgentConfig(
                    team_id=0,
                    inventory=InventoryConfig(initial={"energy": 10, "heart": 5}),
                ),
                AgentConfig(
                    team_id=1,
                    inventory=InventoryConfig(initial={"energy": 10, "heart": 5}),
                ),
            ],
        )

        mg_config = MettaGridConfig(game=config)
        mg_config.game.map_builder = ObjectNameMapBuilder.Config(map_data=map_data)
        sim = Simulation(mg_config, seed=42)

        # Agent 0 tries to move east into Agent 1 without weapon vibe
        # (default vibe should not trigger attack)
        sim.agent(0).set_action("move_east")
        sim.agent(1).set_action("noop")
        sim.step()

        # Agent 1 should NOT be frozen (no attack happened)
        assert not get_agent_frozen_status(sim, 1), "Agent 1 should not be frozen without attack"


class TestVibeActionWithEmptyTarget:
    """Tests that vibe actions don't interfere with normal movement."""

    def test_movement_works_normally_into_empty_space(self):
        """Agent can still move normally into empty space even with vibe actions configured."""
        map_data = [
            ["wall", "wall", "wall", "wall"],
            ["wall", "agent.red", "empty", "wall"],
            ["wall", "wall", "wall", "wall"],
        ]

        config = GameConfig(
            max_steps=100,
            num_agents=1,
            obs=ObsConfig(width=5, height=5, num_tokens=50),
            resource_names=["energy"],
            actions=ActionsConfig(
                noop=NoopActionConfig(),
                move=MoveActionConfig(),
                change_vibe=ChangeVibeActionConfig(),
                attack=AttackActionConfig(
                    enabled=False,
                    vibes=["charger"],
                ),
            ),
            objects={"wall": WallConfig()},
            agent=AgentConfig(
                inventory=InventoryConfig(initial={"energy": 10}),
            ),
        )

        mg_config = MettaGridConfig(game=config)
        mg_config.game.map_builder = ObjectNameMapBuilder.Config(map_data=map_data)
        sim = Simulation(mg_config, seed=42)

        # Agent 0 changes vibe to charger
        sim.agent(0).set_action("change_vibe_charger")
        sim.step()

        initial_pos = get_agent_position(sim, 0)
        assert initial_pos == (1, 1), f"Agent should start at (1,1), got {initial_pos}"

        # Agent 0 moves east into empty space - should work normally
        sim.agent(0).set_action("move_east")
        sim.step()

        new_pos = get_agent_position(sim, 0)
        assert new_pos == (1, 2), f"Agent should move to (1,2), got {new_pos}"
