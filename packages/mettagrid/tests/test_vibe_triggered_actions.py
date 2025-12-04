"""Tests for vibe-triggered actions on move.

This tests the new functionality where:
- Attack handler can specify `vibes` list to trigger attack when agent with that vibe moves into another agent
- Transfer handler can specify `vibes` list to trigger transfer when agent with that vibe moves into another agent
"""

import pytest

from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AgentConfig,
    AgentRewards,
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
from mettagrid.config.vibes import Vibe
from mettagrid.simulator import Simulation
from mettagrid.test_support.map_builders import ObjectNameMapBuilder

# Define test vibes - use a subset from global VIBES to ensure consistency
# VIBES[0] = default, VIBES[1] = charger, etc.
# We'll create custom vibes and make sure vibe_names is properly set
TEST_VIBES = [
    Vibe("😐", "default", category="emotion"),
    Vibe("⚔️", "weapon", category="gear"),
    Vibe("❤️", "heart", category="resource"),
    Vibe("🔋", "battery", category="resource"),
]


def create_two_agent_sim(
    attack_vibes: list[str] | None = None,
    transfer_vibes: list[str] | None = None,
    vibe_transfers: list[VibeTransfer] | None = None,
    initial_inventory: dict[str, int] | None = None,
) -> Simulation:
    """Create a simulation with 2 adjacent agents for testing vibe-triggered actions."""
    game_map = [
        ["wall", "wall", "wall", "wall"],
        ["wall", "agent.agent", "agent.agent", "wall"],
        ["wall", "wall", "wall", "wall"],
    ]

    # vibe_names must be set to match the vibes we're using
    vibe_names = [v.name for v in TEST_VIBES]

    game_config = GameConfig(
        max_steps=50,
        num_agents=2,
        obs=ObsConfig(width=3, height=3, num_tokens=100),
        resource_names=["energy", "heart", "weapon"],
        vibe_names=vibe_names,
        actions=ActionsConfig(
            noop=NoopActionConfig(),
            move=MoveActionConfig(enabled=True),
            # Must set both vibes AND number_of_vibes - vibes for action names, number_of_vibes for C++
            change_vibe=ChangeVibeActionConfig(enabled=True, vibes=TEST_VIBES, number_of_vibes=len(TEST_VIBES)),
            attack=AttackActionConfig(
                enabled=False,  # Not enabled as standalone action
                defense_resources={},  # No defense so attack always succeeds
                loot=["heart"],
                vibes=attack_vibes or [],
            ),
            transfer=TransferActionConfig(
                enabled=True,
                vibe_transfers=vibe_transfers or [],
                vibes=transfer_vibes or [],
            ),
        ),
        agent=AgentConfig(
            rewards=AgentRewards(),
            initial_inventory=initial_inventory or {"energy": 100, "heart": 10, "weapon": 5},
        ),
        objects={
            "wall": WallConfig(),
        },
    )

    cfg = MettaGridConfig(game=game_config)
    cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)

    return Simulation(cfg, seed=42)


class TestVibeTriggeredAttack:
    """Test attack triggered by vibes on move."""

    def test_attack_triggered_by_weapon_vibe(self):
        """Test that moving into another agent triggers attack when agent has weapon vibe."""
        sim = create_two_agent_sim(
            attack_vibes=["weapon"],
            initial_inventory={"energy": 100, "heart": 10, "weapon": 5},
        )

        # Agent 0 changes vibe to "weapon"
        sim.agent(0).set_action("change_vibe_weapon")
        sim.agent(1).set_action("noop")
        sim.step()

        # Verify vibe is set
        objects = sim.grid_objects()
        agents = sorted([obj for obj in objects.values() if "agent_id" in obj], key=lambda x: x["agent_id"])
        assert agents[0]["vibe"] == 1, "Agent 0 should have weapon vibe (id=1)"

        # Get inventories before attack
        heart_idx = sim.resource_names.index("heart")
        agent0_hearts_before = agents[0]["inventory"][heart_idx]

        # Agent 0 moves east into Agent 1 - should trigger attack
        sim.agent(0).set_action("move_east")
        sim.agent(1).set_action("noop")
        sim.step()

        # Check that attack occurred (hearts were looted)
        # After successful attack, attacker should have gained hearts
        objects_after = sim.grid_objects()
        agents_after = sorted([obj for obj in objects_after.values() if "agent_id" in obj], key=lambda x: x["agent_id"])

        agent0_hearts_after = agents_after[0]["inventory"][heart_idx]

        # Agent 0 should have gained hearts (attack succeeded, looted hearts from frozen agent)
        assert agent0_hearts_after > agent0_hearts_before, (
            f"Agent 0 should have gained hearts from attack. Had {agent0_hearts_before}, now has {agent0_hearts_after}"
        )

    def test_no_attack_without_weapon_vibe(self):
        """Test that moving into another agent does NOT trigger attack without weapon vibe."""
        sim = create_two_agent_sim(
            attack_vibes=["weapon"],
            initial_inventory={"energy": 100, "heart": 10, "weapon": 5},
        )

        # Agent 0 keeps default vibe (not weapon)
        # Get inventories before move
        objects = sim.grid_objects()
        agents = sorted([obj for obj in objects.values() if "agent_id" in obj], key=lambda x: x["agent_id"])
        heart_idx = sim.resource_names.index("heart")
        agent0_hearts_before = agents[0]["inventory"][heart_idx]
        agent1_hearts_before = agents[1]["inventory"][heart_idx]

        # Agent 0 moves east into Agent 1 - should NOT trigger attack (wrong vibe)
        sim.agent(0).set_action("move_east")
        sim.agent(1).set_action("noop")
        sim.step()

        # Check that no attack occurred
        objects_after = sim.grid_objects()
        agents_after = sorted([obj for obj in objects_after.values() if "agent_id" in obj], key=lambda x: x["agent_id"])

        agent0_hearts_after = agents_after[0]["inventory"][heart_idx]
        agent1_hearts_after = agents_after[1]["inventory"][heart_idx]

        # Hearts should be unchanged
        assert agent0_hearts_after == agent0_hearts_before, (
            f"Agent 0 hearts should be unchanged. Was {agent0_hearts_before}, now {agent0_hearts_after}"
        )
        assert agent1_hearts_after == agent1_hearts_before, (
            f"Agent 1 hearts should be unchanged. Was {agent1_hearts_before}, now {agent1_hearts_after}"
        )


class TestVibeTriggeredTransfer:
    """Test transfer triggered by vibes on move."""

    def test_transfer_triggered_by_heart_vibe(self):
        """Test that moving into another agent triggers transfer when agent has heart vibe."""
        sim = create_two_agent_sim(
            transfer_vibes=["heart"],
            vibe_transfers=[
                VibeTransfer(vibe="heart", target={"heart": 5}, actor={"heart": -5}),
            ],
            initial_inventory={"energy": 100, "heart": 10, "weapon": 5},
        )

        # Agent 0 changes vibe to "heart"
        sim.agent(0).set_action("change_vibe_heart")
        sim.agent(1).set_action("noop")
        sim.step()

        # Verify vibe is set (heart should be vibe id 2)
        objects = sim.grid_objects()
        agents = sorted([obj for obj in objects.values() if "agent_id" in obj], key=lambda x: x["agent_id"])
        assert agents[0]["vibe"] == 2, f"Agent 0 should have heart vibe (id=2), got {agents[0]['vibe']}"

        # Get inventories before transfer
        heart_idx = sim.resource_names.index("heart")
        agent0_hearts_before = agents[0]["inventory"][heart_idx]
        agent1_hearts_before = agents[1]["inventory"][heart_idx]

        # Agent 0 moves east into Agent 1 - should trigger transfer
        sim.agent(0).set_action("move_east")
        sim.agent(1).set_action("noop")
        sim.step()

        # Check that transfer occurred
        objects_after = sim.grid_objects()
        agents_after = sorted([obj for obj in objects_after.values() if "agent_id" in obj], key=lambda x: x["agent_id"])

        agent0_hearts_after = agents_after[0]["inventory"][heart_idx]
        agent1_hearts_after = agents_after[1]["inventory"][heart_idx]

        # Agent 0 should have lost 5 hearts, Agent 1 should have gained 5 hearts
        assert agent0_hearts_after == agent0_hearts_before - 5, (
            f"Agent 0 should have lost 5 hearts. Had {agent0_hearts_before}, now has {agent0_hearts_after}"
        )
        assert agent1_hearts_after == agent1_hearts_before + 5, (
            f"Agent 1 should have gained 5 hearts. Had {agent1_hearts_before}, now has {agent1_hearts_after}"
        )

    def test_no_transfer_without_heart_vibe(self):
        """Test that moving into another agent does NOT trigger transfer without heart vibe."""
        sim = create_two_agent_sim(
            transfer_vibes=["heart"],
            vibe_transfers=[
                VibeTransfer(vibe="heart", target={"heart": 5}, actor={"heart": -5}),
            ],
            initial_inventory={"energy": 100, "heart": 10, "weapon": 5},
        )

        # Agent 0 keeps default vibe (not heart)
        # Get inventories before move
        objects = sim.grid_objects()
        agents = sorted([obj for obj in objects.values() if "agent_id" in obj], key=lambda x: x["agent_id"])
        heart_idx = sim.resource_names.index("heart")
        agent0_hearts_before = agents[0]["inventory"][heart_idx]
        agent1_hearts_before = agents[1]["inventory"][heart_idx]

        # Agent 0 moves east into Agent 1 - should NOT trigger transfer
        sim.agent(0).set_action("move_east")
        sim.agent(1).set_action("noop")
        sim.step()

        # Check that no transfer occurred
        objects_after = sim.grid_objects()
        agents_after = sorted([obj for obj in objects_after.values() if "agent_id" in obj], key=lambda x: x["agent_id"])

        agent0_hearts_after = agents_after[0]["inventory"][heart_idx]
        agent1_hearts_after = agents_after[1]["inventory"][heart_idx]

        # Hearts should be unchanged
        assert agent0_hearts_after == agent0_hearts_before, (
            f"Agent 0 hearts should be unchanged. Was {agent0_hearts_before}, now {agent0_hearts_after}"
        )
        assert agent1_hearts_after == agent1_hearts_before, (
            f"Agent 1 hearts should be unchanged. Was {agent1_hearts_before}, now {agent1_hearts_after}"
        )


class TestMultipleVibeHandlers:
    """Test multiple vibes triggering different handlers."""

    def test_attack_and_transfer_different_vibes(self):
        """Test that attack and transfer can be triggered by different vibes."""
        sim = create_two_agent_sim(
            attack_vibes=["weapon"],
            transfer_vibes=["heart", "battery"],
            vibe_transfers=[
                VibeTransfer(vibe="heart", target={"heart": 5}, actor={"heart": -5}),
                VibeTransfer(vibe="battery", target={"energy": 20}, actor={"energy": -20}),
            ],
            initial_inventory={"energy": 100, "heart": 10, "weapon": 5},
        )

        # Test 1: battery vibe triggers energy transfer
        sim.agent(0).set_action("change_vibe_battery")
        sim.agent(1).set_action("noop")
        sim.step()

        # Verify vibe is set (battery should be vibe id 3)
        objects = sim.grid_objects()
        agents = sorted([obj for obj in objects.values() if "agent_id" in obj], key=lambda x: x["agent_id"])
        assert agents[0]["vibe"] == 3, f"Agent 0 should have battery vibe (id=3), got {agents[0]['vibe']}"

        energy_idx = sim.resource_names.index("energy")
        agent0_energy_before = agents[0]["inventory"][energy_idx]
        agent1_energy_before = agents[1]["inventory"][energy_idx]

        # Agent 0 moves east into Agent 1 - should trigger energy transfer
        sim.agent(0).set_action("move_east")
        sim.agent(1).set_action("noop")
        sim.step()

        objects_after = sim.grid_objects()
        agents_after = sorted([obj for obj in objects_after.values() if "agent_id" in obj], key=lambda x: x["agent_id"])

        agent0_energy_after = agents_after[0]["inventory"][energy_idx]
        agent1_energy_after = agents_after[1]["inventory"][energy_idx]

        # Energy transfer should have occurred
        assert agent0_energy_after == agent0_energy_before - 20, (
            f"Agent 0 should have lost 20 energy. Had {agent0_energy_before}, now has {agent0_energy_after}"
        )
        assert agent1_energy_after == agent1_energy_before + 20, (
            f"Agent 1 should have gained 20 energy. Had {agent1_energy_before}, now has {agent1_energy_after}"
        )


class TestVibeConfigValidation:
    """Test configuration validation for vibes."""

    def test_invalid_vibe_name_raises_error(self):
        """Test that using an invalid vibe name in attack.vibes raises an error."""
        with pytest.raises(ValueError, match="Unknown vibe name"):
            game_config = GameConfig(
                max_steps=50,
                num_agents=2,
                resource_names=["energy"],
                vibe_names=["default", "weapon"],
                actions=ActionsConfig(
                    noop=NoopActionConfig(),
                    attack=AttackActionConfig(
                        enabled=False,
                        vibes=["nonexistent_vibe"],  # Invalid vibe name
                    ),
                ),
            )
            cfg = MettaGridConfig(game=game_config)
            cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=[["agent.agent", "agent.agent"]])
            Simulation(cfg)

    def test_invalid_vibe_name_in_transfer_raises_error(self):
        """Test that using an invalid vibe name in transfer.vibes raises an error."""
        with pytest.raises(ValueError, match="Unknown vibe name"):
            game_config = GameConfig(
                max_steps=50,
                num_agents=2,
                resource_names=["energy"],
                vibe_names=["default", "weapon"],
                actions=ActionsConfig(
                    noop=NoopActionConfig(),
                    transfer=TransferActionConfig(
                        enabled=True,
                        vibes=["nonexistent_vibe"],  # Invalid vibe name
                    ),
                ),
            )
            cfg = MettaGridConfig(game=game_config)
            cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=[["agent.agent", "agent.agent"]])
            Simulation(cfg)
