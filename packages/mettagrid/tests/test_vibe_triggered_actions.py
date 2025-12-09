"""Tests for vibe-triggered actions on move (attack and transfer)."""

import pytest

from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AgentConfig,
    AttackActionConfig,
    ChangeVibeActionConfig,
    GameConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    ObsConfig,
    TransferActionConfig,
    VibeTransfer,
    WallConfig,
)
from mettagrid.simulator import Simulation
from mettagrid.test_support.map_builders import ObjectNameMapBuilder


def get_resource_index(sim: Simulation, resource_name: str) -> int:
    """Get the index of a resource by name."""
    return sim.resource_names.index(resource_name)


def get_agent_inventory(sim: Simulation, agent_id: int) -> dict:
    """Get an agent's inventory as a dict of resource_name -> amount."""
    grid_objects = sim.grid_objects()
    for obj in grid_objects.values():
        if obj.get("agent_id") == agent_id:
            inventory = obj["inventory"]
            return {
                name: inventory.get(idx, 0) for idx, name in enumerate(sim.resource_names) if idx in inventory
            }
    return {}


def get_agent_position(sim: Simulation, agent_id: int) -> tuple:
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
                change_vibe=ChangeVibeActionConfig(number_of_vibes=10),
                attack=AttackActionConfig(
                    enabled=False,  # Disable direct attack actions
                    vibes=["weapon"],  # Attack triggers on move when agent has weapon vibe
                ),
            ),
            objects={"wall": WallConfig()},
            agents=[
                AgentConfig(
                    team_id=0,
                    freeze_duration=5,
                    initial_inventory={"energy": 10, "heart": 5},
                ),
                AgentConfig(
                    team_id=1,
                    freeze_duration=5,
                    initial_inventory={"energy": 10, "heart": 5},
                ),
            ],
        )

        mg_config = MettaGridConfig(game=config)
        mg_config.game.map_builder = ObjectNameMapBuilder.Config(map_data=map_data)
        sim = Simulation(mg_config, seed=42)

        # Verify initial state - neither agent is frozen
        assert not get_agent_frozen_status(sim, 0), "Agent 0 should not start frozen"
        assert not get_agent_frozen_status(sim, 1), "Agent 1 should not start frozen"

        # Agent 0 changes vibe to weapon (vibe index 1 = "weapon")
        sim.agent(0).set_action("change_vibe_weapon")
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
                change_vibe=ChangeVibeActionConfig(number_of_vibes=10),
                attack=AttackActionConfig(
                    enabled=False,
                    vibes=["weapon"],  # Attack only triggers with weapon vibe
                ),
            ),
            objects={"wall": WallConfig()},
            agents=[
                AgentConfig(
                    team_id=0,
                    freeze_duration=5,
                    initial_inventory={"energy": 10, "heart": 5},
                ),
                AgentConfig(
                    team_id=1,
                    freeze_duration=5,
                    initial_inventory={"energy": 10, "heart": 5},
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


class TestVibeTriggeredTransfer:
    """Tests for transfer-on-move triggered by agent vibe."""

    def test_transfer_triggers_on_move_with_matching_vibe(self):
        """When an agent with a transfer vibe moves into another agent, resources transfer."""
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
                change_vibe=ChangeVibeActionConfig(number_of_vibes=10),
                transfer=TransferActionConfig(
                    enabled=True,
                    vibes=["plug"],  # Transfer triggers on move when agent has plug vibe
                    vibe_transfers=[
                        VibeTransfer(
                            vibe="plug",
                            target={"energy": 5},  # Target gains 5 energy
                            actor={"energy": -5, "heart": -1},  # Actor loses 5 energy and 1 heart
                        ),
                    ],
                ),
            ),
            objects={"wall": WallConfig()},
            agents=[
                AgentConfig(
                    team_id=0,
                    initial_inventory={"energy": 10, "heart": 5},
                ),
                AgentConfig(
                    team_id=1,
                    initial_inventory={"energy": 10, "heart": 5},
                ),
            ],
        )

        mg_config = MettaGridConfig(game=config)
        mg_config.game.map_builder = ObjectNameMapBuilder.Config(map_data=map_data)
        sim = Simulation(mg_config, seed=42)

        # Verify initial inventories
        inv0_before = get_agent_inventory(sim, 0)
        inv1_before = get_agent_inventory(sim, 1)
        assert inv0_before.get("energy", 0) == 10, "Agent 0 should start with 10 energy"
        assert inv0_before.get("heart", 0) == 5, "Agent 0 should start with 5 heart"
        assert inv1_before.get("energy", 0) == 10, "Agent 1 should start with 10 energy"

        # Agent 0 changes vibe to plug (vibe index 3 = "plug")
        sim.agent(0).set_action("change_vibe_plug")
        sim.agent(1).set_action("noop")
        sim.step()

        # Agent 0 moves into Agent 1 (east) - should trigger transfer due to plug vibe
        sim.agent(0).set_action("move_east")
        sim.agent(1).set_action("noop")
        sim.step()

        # Check inventories after transfer
        inv0_after = get_agent_inventory(sim, 0)
        inv1_after = get_agent_inventory(sim, 1)

        # Agent 0 should have lost 5 energy and 1 heart
        assert inv0_after.get("energy", 0) == 5, f"Agent 0 should have 5 energy, got {inv0_after.get('energy', 0)}"
        assert inv0_after.get("heart", 0) == 4, f"Agent 0 should have 4 heart, got {inv0_after.get('heart', 0)}"

        # Agent 1 should have gained 5 energy
        assert inv1_after.get("energy", 0) == 15, f"Agent 1 should have 15 energy, got {inv1_after.get('energy', 0)}"

    def test_transfer_fails_without_enough_resources(self):
        """Transfer doesn't happen if actor doesn't have enough resources."""
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
                change_vibe=ChangeVibeActionConfig(number_of_vibes=10),
                transfer=TransferActionConfig(
                    enabled=True,
                    vibes=["plug"],
                    vibe_transfers=[
                        VibeTransfer(
                            vibe="plug",
                            target={"energy": 50},  # Target gains 50 energy
                            actor={"energy": -50},  # Actor needs to give 50 energy (more than they have)
                        ),
                    ],
                ),
            ),
            objects={"wall": WallConfig()},
            agents=[
                AgentConfig(
                    team_id=0,
                    initial_inventory={"energy": 10},  # Only 10 energy, not enough
                ),
                AgentConfig(
                    team_id=1,
                    initial_inventory={"energy": 10},
                ),
            ],
        )

        mg_config = MettaGridConfig(game=config)
        mg_config.game.map_builder = ObjectNameMapBuilder.Config(map_data=map_data)
        sim = Simulation(mg_config, seed=42)

        # Agent 0 changes vibe to plug
        sim.agent(0).set_action("change_vibe_plug")
        sim.agent(1).set_action("noop")
        sim.step()

        # Agent 0 moves into Agent 1 (east) - transfer should fail due to insufficient resources
        sim.agent(0).set_action("move_east")
        sim.agent(1).set_action("noop")
        sim.step()

        # Check inventories - should be unchanged
        inv0_after = get_agent_inventory(sim, 0)
        inv1_after = get_agent_inventory(sim, 1)

        assert inv0_after.get("energy", 0) == 10, f"Agent 0 should still have 10 energy, got {inv0_after.get('energy', 0)}"
        assert inv1_after.get("energy", 0) == 10, f"Agent 1 should still have 10 energy, got {inv1_after.get('energy', 0)}"


class TestVibeActionPriority:
    """Tests for priority between attack and transfer on move."""

    def test_attack_takes_priority_over_transfer(self):
        """When a vibe is configured for both attack and transfer, attack should happen first."""
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
                change_vibe=ChangeVibeActionConfig(number_of_vibes=10),
                attack=AttackActionConfig(
                    enabled=False,
                    vibes=["weapon"],  # Attack triggers on weapon vibe
                ),
                transfer=TransferActionConfig(
                    enabled=True,
                    vibes=["weapon"],  # Transfer also triggers on weapon vibe
                    vibe_transfers=[
                        VibeTransfer(
                            vibe="weapon",
                            target={"energy": 5},
                            actor={"energy": -5},
                        ),
                    ],
                ),
            ),
            objects={"wall": WallConfig()},
            agents=[
                AgentConfig(
                    team_id=0,
                    freeze_duration=5,
                    initial_inventory={"energy": 10, "heart": 5},
                ),
                AgentConfig(
                    team_id=1,
                    freeze_duration=5,
                    initial_inventory={"energy": 10, "heart": 5},
                ),
            ],
        )

        mg_config = MettaGridConfig(game=config)
        mg_config.game.map_builder = ObjectNameMapBuilder.Config(map_data=map_data)
        sim = Simulation(mg_config, seed=42)

        # Agent 0 changes vibe to weapon
        sim.agent(0).set_action("change_vibe_weapon")
        sim.agent(1).set_action("noop")
        sim.step()

        # Agent 0 moves into Agent 1 (east)
        sim.agent(0).set_action("move_east")
        sim.agent(1).set_action("noop")
        sim.step()

        # Attack should have happened (Agent 1 frozen), not transfer
        # (Attack is checked first in the Move handler)
        assert get_agent_frozen_status(sim, 1), "Agent 1 should be frozen from attack"


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
                change_vibe=ChangeVibeActionConfig(number_of_vibes=10),
                attack=AttackActionConfig(
                    enabled=False,
                    vibes=["weapon"],
                ),
            ),
            objects={"wall": WallConfig()},
            agent=AgentConfig(
                initial_inventory={"energy": 10},
            ),
        )

        mg_config = MettaGridConfig(game=config)
        mg_config.game.map_builder = ObjectNameMapBuilder.Config(map_data=map_data)
        sim = Simulation(mg_config, seed=42)

        # Agent 0 changes vibe to weapon
        sim.agent(0).set_action("change_vibe_weapon")
        sim.step()

        initial_pos = get_agent_position(sim, 0)
        assert initial_pos == (1, 1), f"Agent should start at (1,1), got {initial_pos}"

        # Agent 0 moves east into empty space - should work normally
        sim.agent(0).set_action("move_east")
        sim.step()

        new_pos = get_agent_position(sim, 0)
        assert new_pos == (1, 2), f"Agent should move to (1,2), got {new_pos}"
