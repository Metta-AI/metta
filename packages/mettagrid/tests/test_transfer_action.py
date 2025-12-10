"""Tests for vibe-triggered transfer actions on move.

This tests the Transfer action functionality where:
- Transfer handler can specify `vibes` list to trigger transfer when agent with that vibe moves into another agent
- VibeTransfer configs define what resource changes happen for both actor and target
"""

import pytest

from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AgentConfig,
    AgentRewards,
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

# Define test vibes
TEST_VIBES = [
    Vibe("😐", "default", category="emotion"),
    Vibe("🔋", "battery", category="resource"),
    Vibe("❤️", "heart", category="resource"),
]


def create_two_agent_sim(
    transfer_vibes: list[str] | None = None,
    vibe_transfers: list[VibeTransfer] | None = None,
    initial_inventory: dict[str, int] | None = None,
) -> Simulation:
    """Create a simulation with 2 adjacent agents for testing vibe-triggered transfers."""
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
        resource_names=["energy", "heart"],
        vibe_names=vibe_names,
        actions=ActionsConfig(
            noop=NoopActionConfig(),
            move=MoveActionConfig(enabled=True),
            change_vibe=ChangeVibeActionConfig(enabled=True, vibes=TEST_VIBES, number_of_vibes=len(TEST_VIBES)),
            transfer=TransferActionConfig(
                enabled=True,
                vibe_transfers=vibe_transfers or [],
                vibes=transfer_vibes or [],
            ),
        ),
        agent=AgentConfig(
            rewards=AgentRewards(),
            initial_inventory=initial_inventory or {"energy": 100, "heart": 10},
        ),
        objects={
            "wall": WallConfig(),
        },
    )

    cfg = MettaGridConfig(game=game_config)
    cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)

    return Simulation(cfg, seed=42)


class TestVibeTriggeredTransfer:
    """Test transfer triggered by vibes on move."""

    def test_transfer_triggered_by_battery_vibe(self):
        """Test that moving into another agent triggers transfer when agent has battery vibe."""
        sim = create_two_agent_sim(
            transfer_vibes=["battery"],
            vibe_transfers=[
                VibeTransfer(vibe="battery", target={"energy": 50}, actor={"energy": -50}),
            ],
            initial_inventory={"energy": 100, "heart": 10},
        )

        # Agent 0 changes vibe to "battery"
        sim.agent(0).set_action("change_vibe_battery")
        sim.agent(1).set_action("noop")
        sim.step()

        # Verify vibe is set (battery should be vibe id 1)
        objects = sim.grid_objects()
        agents = sorted([obj for obj in objects.values() if "agent_id" in obj], key=lambda x: x["agent_id"])
        assert agents[0]["vibe"] == 1, f"Agent 0 should have battery vibe (id=1), got {agents[0]['vibe']}"

        # Get inventories before transfer
        energy_idx = sim.resource_names.index("energy")
        agent0_energy_before = agents[0]["inventory"][energy_idx]
        agent1_energy_before = agents[1]["inventory"][energy_idx]

        # Agent 0 moves east into Agent 1 - should trigger transfer
        sim.agent(0).set_action("move_east")
        sim.agent(1).set_action("noop")
        sim.step()

        # Check that transfer occurred
        objects_after = sim.grid_objects()
        agents_after = sorted([obj for obj in objects_after.values() if "agent_id" in obj], key=lambda x: x["agent_id"])

        agent0_energy_after = agents_after[0]["inventory"][energy_idx]
        agent1_energy_after = agents_after[1]["inventory"][energy_idx]

        # Agent 0 should have lost 50 energy, Agent 1 should have gained 50 energy
        assert agent0_energy_after == agent0_energy_before - 50, (
            f"Agent 0 should have lost 50 energy. Had {agent0_energy_before}, now has {agent0_energy_after}"
        )
        assert agent1_energy_after == agent1_energy_before + 50, (
            f"Agent 1 should have gained 50 energy. Had {agent1_energy_before}, now has {agent1_energy_after}"
        )

    def test_no_transfer_without_configured_vibe(self):
        """Test that moving into another agent does NOT trigger transfer without the right vibe."""
        sim = create_two_agent_sim(
            transfer_vibes=["battery"],
            vibe_transfers=[
                VibeTransfer(vibe="battery", target={"energy": 50}, actor={"energy": -50}),
            ],
            initial_inventory={"energy": 100, "heart": 10},
        )

        # Agent 0 keeps default vibe (not battery)
        # Get inventories before move
        objects = sim.grid_objects()
        agents = sorted([obj for obj in objects.values() if "agent_id" in obj], key=lambda x: x["agent_id"])
        energy_idx = sim.resource_names.index("energy")
        agent0_energy_before = agents[0]["inventory"][energy_idx]
        agent1_energy_before = agents[1]["inventory"][energy_idx]

        # Agent 0 moves east into Agent 1 - should NOT trigger transfer (wrong vibe)
        sim.agent(0).set_action("move_east")
        sim.agent(1).set_action("noop")
        sim.step()

        # Check that no transfer occurred
        objects_after = sim.grid_objects()
        agents_after = sorted([obj for obj in objects_after.values() if "agent_id" in obj], key=lambda x: x["agent_id"])

        agent0_energy_after = agents_after[0]["inventory"][energy_idx]
        agent1_energy_after = agents_after[1]["inventory"][energy_idx]

        # Energy should be unchanged
        assert agent0_energy_after == agent0_energy_before, (
            f"Agent 0 energy should be unchanged. Was {agent0_energy_before}, now {agent0_energy_after}"
        )
        assert agent1_energy_after == agent1_energy_before, (
            f"Agent 1 energy should be unchanged. Was {agent1_energy_before}, now {agent1_energy_after}"
        )

    def test_transfer_with_multiple_resources(self):
        """Test that transfer can affect multiple resources."""
        sim = create_two_agent_sim(
            transfer_vibes=["heart"],
            vibe_transfers=[
                VibeTransfer(vibe="heart", target={"heart": 5, "energy": 10}, actor={"heart": -5, "energy": -10}),
            ],
            initial_inventory={"energy": 100, "heart": 10},
        )

        # Agent 0 changes vibe to "heart"
        sim.agent(0).set_action("change_vibe_heart")
        sim.agent(1).set_action("noop")
        sim.step()

        # Get inventories before transfer
        objects = sim.grid_objects()
        agents = sorted([obj for obj in objects.values() if "agent_id" in obj], key=lambda x: x["agent_id"])
        energy_idx = sim.resource_names.index("energy")
        heart_idx = sim.resource_names.index("heart")
        agent0_energy_before = agents[0]["inventory"][energy_idx]
        agent0_heart_before = agents[0]["inventory"][heart_idx]
        agent1_energy_before = agents[1]["inventory"][energy_idx]
        agent1_heart_before = agents[1]["inventory"][heart_idx]

        # Agent 0 moves east into Agent 1 - should trigger transfer
        sim.agent(0).set_action("move_east")
        sim.agent(1).set_action("noop")
        sim.step()

        # Check that transfer occurred for both resources
        objects_after = sim.grid_objects()
        agents_after = sorted([obj for obj in objects_after.values() if "agent_id" in obj], key=lambda x: x["agent_id"])

        # Agent 0 should have lost resources
        assert agents_after[0]["inventory"][energy_idx] == agent0_energy_before - 10
        assert agents_after[0]["inventory"][heart_idx] == agent0_heart_before - 5

        # Agent 1 should have gained resources
        assert agents_after[1]["inventory"][energy_idx] == agent1_energy_before + 10
        assert agents_after[1]["inventory"][heart_idx] == agent1_heart_before + 5

    def test_transfer_fails_without_enough_resources(self):
        """Test that transfer fails if actor doesn't have enough resources."""
        sim = create_two_agent_sim(
            transfer_vibes=["battery"],
            vibe_transfers=[
                VibeTransfer(vibe="battery", target={"energy": 200}, actor={"energy": -200}),
            ],
            initial_inventory={"energy": 100, "heart": 10},  # Only 100 energy, transfer needs 200
        )

        # Agent 0 changes vibe to "battery"
        sim.agent(0).set_action("change_vibe_battery")
        sim.agent(1).set_action("noop")
        sim.step()

        # Get inventories before attempted transfer
        objects = sim.grid_objects()
        agents = sorted([obj for obj in objects.values() if "agent_id" in obj], key=lambda x: x["agent_id"])
        energy_idx = sim.resource_names.index("energy")
        agent0_energy_before = agents[0]["inventory"][energy_idx]
        agent1_energy_before = agents[1]["inventory"][energy_idx]

        # Agent 0 moves east into Agent 1 - should NOT trigger transfer (not enough resources)
        sim.agent(0).set_action("move_east")
        sim.agent(1).set_action("noop")
        sim.step()

        # Check that no transfer occurred
        objects_after = sim.grid_objects()
        agents_after = sorted([obj for obj in objects_after.values() if "agent_id" in obj], key=lambda x: x["agent_id"])

        agent0_energy_after = agents_after[0]["inventory"][energy_idx]
        agent1_energy_after = agents_after[1]["inventory"][energy_idx]

        # Energy should be unchanged (transfer failed)
        assert agent0_energy_after == agent0_energy_before, (
            f"Agent 0 energy should be unchanged. Was {agent0_energy_before}, now {agent0_energy_after}"
        )
        assert agent1_energy_after == agent1_energy_before, (
            f"Agent 1 energy should be unchanged. Was {agent1_energy_before}, now {agent1_energy_after}"
        )


class TestVibeConfigValidation:
    """Test configuration validation for transfer vibes."""

    def test_invalid_vibe_name_in_transfer_raises_error(self):
        """Test that using an invalid vibe name in transfer.vibes raises an error."""
        with pytest.raises(ValueError, match="Unknown vibe name"):
            game_config = GameConfig(
                max_steps=50,
                num_agents=2,
                resource_names=["energy"],
                vibe_names=["default", "battery"],
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
