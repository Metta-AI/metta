"""Tests for vibe-triggered transfer actions on move.

This tests the Transfer action functionality where:
- Transfer is triggered when agent with a vibe in vibe_transfers moves into another agent
- VibeTransfer configs define what resource changes happen for both actor and target
"""

import pytest

from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    ChangeVibeActionConfig,
    GameConfig,
    InventoryConfig,
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

# Skip all tests in this module if C++ doesn't support transfer action
try:
    from mettagrid.mettagrid_c import TransferActionConfig as _  # noqa: F401

    HAS_TRANSFER = True
except ImportError:
    HAS_TRANSFER = False

pytestmark = pytest.mark.skipif(not HAS_TRANSFER, reason="Transfer action not available in C++ bindings")

# Use vibes from the global VIBES list
# Default (vibe id 0): "default"
# Charger (vibe id 1): "charger" - use for energy transfer tests
# Heart_a (vibe id 10): "heart_a" - use for heart transfer tests
CHARGER_VIBE_NAME = "charger"  # VIBES[1].name
HEART_VIBE_NAME = "heart_a"  # VIBES[10].name


def create_two_agent_sim(
    vibe_transfers: list[VibeTransfer] | None = None,
    initial_inventory: dict[str, int] | None = None,
) -> Simulation:
    """Create a simulation with 2 adjacent agents for testing vibe-triggered transfers."""
    game_map = [
        ["wall", "wall", "wall", "wall"],
        ["wall", "agent.agent", "agent.agent", "wall"],
        ["wall", "wall", "wall", "wall"],
    ]

    game_config = GameConfig(
        max_steps=50,
        num_agents=2,
        obs=ObsConfig(width=3, height=3, num_tokens=100),
        resource_names=["energy", "heart"],
        actions=ActionsConfig(
            noop=NoopActionConfig(),
            move=MoveActionConfig(enabled=True),
            change_vibe=ChangeVibeActionConfig(enabled=True),
            transfer=TransferActionConfig(
                enabled=True,
                vibe_transfers=vibe_transfers or [],
            ),
        ),
        agent=AgentConfig(
            rewards=AgentRewards(),
            inventory=InventoryConfig(initial=initial_inventory or {"energy": 100, "heart": 10}),
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

    def test_transfer_triggered_by_charger_vibe(self):
        """Test that moving into another agent triggers transfer when agent has charger vibe."""
        sim = create_two_agent_sim(
            vibe_transfers=[
                VibeTransfer(vibe=CHARGER_VIBE_NAME, target={"energy": 50}, actor={"energy": -50}),
            ],
            initial_inventory={"energy": 100, "heart": 10},
        )

        # Agent 0 changes vibe to "charger" (vibe id 1)
        sim.agent(0).set_action("change_vibe_charger")
        sim.agent(1).set_action("noop")
        sim.step()

        # Verify vibe is set (charger should be vibe id 1)
        objects = sim.grid_objects()
        agents = sorted([obj for obj in objects.values() if "agent_id" in obj], key=lambda x: x["agent_id"])
        assert agents[0]["vibe"] == 1, f"Agent 0 should have charger vibe (id=1), got {agents[0]['vibe']}"

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
            vibe_transfers=[
                VibeTransfer(vibe=CHARGER_VIBE_NAME, target={"energy": 50}, actor={"energy": -50}),
            ],
            initial_inventory={"energy": 100, "heart": 10},
        )

        # Agent 0 keeps default vibe (not charger)
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
            vibe_transfers=[
                VibeTransfer(
                    vibe=HEART_VIBE_NAME,
                    target={"heart": 5, "energy": 10},
                    actor={"heart": -5, "energy": -10},
                ),
            ],
            initial_inventory={"energy": 100, "heart": 10},
        )

        # Agent 0 changes vibe to "heart_a"
        sim.agent(0).set_action("change_vibe_heart_a")
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
            vibe_transfers=[
                VibeTransfer(vibe=CHARGER_VIBE_NAME, target={"energy": 200}, actor={"energy": -200}),
            ],
            initial_inventory={"energy": 100, "heart": 10},  # Only 100 energy, transfer needs 200
        )

        # Agent 0 changes vibe to "charger"
        sim.agent(0).set_action("change_vibe_charger")
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

    def test_transfer_fails_with_large_delta_exceeding_inventory(self):
        """Test that transfer correctly fails when delta exceeds uint16_t range.

        This tests the fix for integer overflow: if delta is cast to uint16_t before comparison,
        values exceeding 65535 would be truncated (e.g., 70000 becomes 4464).
        The fix casts inventory amounts to int instead, preserving the full delta value.
        """
        # Configure transfer that requires 70000 energy (exceeds uint16_t max of 65535)
        # Actor only has 5000 energy - transfer should fail
        sim = create_two_agent_sim(
            vibe_transfers=[
                VibeTransfer(vibe=CHARGER_VIBE_NAME, target={"energy": 70000}, actor={"energy": -70000}),
            ],
            initial_inventory={"energy": 5000, "heart": 10},
        )

        # Agent 0 changes vibe to "charger"
        sim.agent(0).set_action("change_vibe_charger")
        sim.agent(1).set_action("noop")
        sim.step()

        # Get inventories before attempted transfer
        objects = sim.grid_objects()
        agents = sorted([obj for obj in objects.values() if "agent_id" in obj], key=lambda x: x["agent_id"])
        energy_idx = sim.resource_names.index("energy")
        agent0_energy_before = agents[0]["inventory"][energy_idx]
        agent1_energy_before = agents[1]["inventory"][energy_idx]

        # Agent 0 moves east into Agent 1 - should NOT trigger transfer
        # (agent only has 5000 energy, needs 70000)
        sim.agent(0).set_action("move_east")
        sim.agent(1).set_action("noop")
        sim.step()

        # Check that no transfer occurred (validation correctly rejected it)
        objects_after = sim.grid_objects()
        agents_after = sorted([obj for obj in objects_after.values() if "agent_id" in obj], key=lambda x: x["agent_id"])

        agent0_energy_after = agents_after[0]["inventory"][energy_idx]
        agent1_energy_after = agents_after[1]["inventory"][energy_idx]

        # Energy should be unchanged (transfer failed due to insufficient resources)
        assert agent0_energy_after == agent0_energy_before, (
            f"Agent 0 energy should be unchanged. Was {agent0_energy_before}, now {agent0_energy_after}. "
            "Transfer should have been rejected because agent doesn't have 70000 energy."
        )
        assert agent1_energy_after == agent1_energy_before, (
            f"Agent 1 energy should be unchanged. Was {agent1_energy_before}, now {agent1_energy_after}"
        )


class TestVibeConfigValidation:
    """Test configuration validation for transfer vibes."""

    def test_invalid_vibe_name_in_vibe_transfers_raises_error(self):
        """Test that using an invalid vibe name in vibe_transfers raises an error."""
        with pytest.raises(ValueError, match="Unknown vibe name"):
            game_config = GameConfig(
                max_steps=50,
                num_agents=2,
                resource_names=["energy"],
                actions=ActionsConfig(
                    noop=NoopActionConfig(),
                    transfer=TransferActionConfig(
                        enabled=True,
                        vibe_transfers=[
                            VibeTransfer(vibe="nonexistent_vibe", target={"energy": 10}),
                        ],
                    ),
                ),
            )
            cfg = MettaGridConfig(game=game_config)
            cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=[["agent.agent", "agent.agent"]])
            Simulation(cfg)
