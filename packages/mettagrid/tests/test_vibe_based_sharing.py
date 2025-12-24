"""Test vibe-based agent resource sharing functionality.

This test verifies that agents share resources based on their current vibe:
- Agents transfer resources according to vibe_transfers configuration
- vibe_transfers maps vibe names to resource names and amounts
- If vibe doesn't have configured transfers, no sharing occurs
"""

from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.simulator import Simulation


class TestVibeBasedSharing:
    """Test vibe-based resource sharing when agents use each other."""

    def test_vibe_based_single_resource_sharing(self):
        """Test that agents share resources according to vibe_transfers config."""
        # Create a simple environment with 2 agents
        cfg = MettaGridConfig.EmptyRoom(num_agents=2, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#", "#"],
                ["#", "@", "@", "#"],
                ["#", "#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty"},
        )

        # Configure resources and sharing
        cfg.game.resource_names = ["charger", "water", "food"]
        cfg.game.agent.inventory.initial = {"charger": 10, "water": 8, "food": 6}

        # Configure vibe_transfers: when agent has "charger" vibe, transfer 5 charger
        # Only resources listed here participate in transfers
        cfg.game.agent.vibe_transfers = {
            "charger": {"charger": 5}  # vibe "charger" -> transfer 5 charger
        }

        # Enable move and change_vibe actions
        cfg.game.actions.move.enabled = True
        cfg.game.actions.noop.enabled = True
        cfg.game.actions.change_vibe.enabled = True
        cfg.game.actions.change_vibe.number_of_vibes = 100  # Ensure we have enough vibes

        sim = Simulation(cfg)

        # Get initial state
        charger_idx = sim.resource_names.index("charger")
        water_idx = sim.resource_names.index("water")
        food_idx = sim.resource_names.index("food")

        # Set agent 0's vibe to "charger" (vibe_id = 1, as defined in vibes.py)
        # Vibe 0 is "default", Vibe 1 is "charger"
        sim.agent(0).set_action("change_vibe_charger")
        sim.agent(1).set_action("noop")
        sim.step()

        # Get agent states before sharing
        grid_objects = sim.grid_objects()
        agents = sorted([obj for obj in grid_objects.values() if "agent_id" in obj], key=lambda x: x["agent_id"])
        agent0_before = agents[0]
        agent1_before = agents[1]

        assert agent0_before["vibe"] == 1, "Agent 0 should have vibe set to 1 (charger)"

        # Check initial inventory
        assert agent0_before["inventory"][charger_idx] == 10
        assert agent0_before["inventory"][water_idx] == 8
        assert agent0_before["inventory"][food_idx] == 6

        assert agent1_before["inventory"][charger_idx] == 10
        assert agent1_before["inventory"][water_idx] == 8
        assert agent1_before["inventory"][food_idx] == 6

        # Have agent 0 move onto agent 1 to trigger onUse with vibe=charger
        # This should share only charger, not water
        sim.agent(0).set_action("move_east")
        sim.agent(1).set_action("noop")
        sim.step()

        # Check inventory after vibe-based sharing
        grid_objects_after = sim.grid_objects()
        agents_after = sorted(
            [obj for obj in grid_objects_after.values() if "agent_id" in obj], key=lambda x: x["agent_id"]
        )
        agent0_after = agents_after[0]
        agent1_after = agents_after[1]

        # With vibe_transfers configured to transfer 5 charger:
        # - Only charger should be shared (5 units as configured)
        # - Water should NOT be shared (no config for water in this vibe)
        # - Food is not configured for transfer
        assert agent0_after["inventory"][charger_idx] == 5, (
            f"Agent 0 should have 5 charger after transferring 5. Has {agent0_after['inventory'][charger_idx]}"
        )
        assert agent0_after["inventory"][water_idx] == 8, (
            f"Agent 0 should still have 8 water (vibe-based sharing only affects charger). "
            f"Has {agent0_after['inventory'][water_idx]}"
        )
        assert agent0_after["inventory"][food_idx] == 6, (
            f"Agent 0 should still have 6 food (not configured). Has {agent0_after['inventory'][food_idx]}"
        )

        assert agent1_after["inventory"][charger_idx] == 15, (
            f"Agent 1 should have 15 charger after receiving half. Has {agent1_after['inventory'][charger_idx]}"
        )
        assert agent1_after["inventory"][water_idx] == 8, (
            f"Agent 1 should still have 8 water (not shared via vibe). Has {agent1_after['inventory'][water_idx]}"
        )
        assert agent1_after["inventory"][food_idx] == 6, (
            f"Agent 1 should still have 6 food (not configured). Has {agent1_after['inventory'][food_idx]}"
        )

    def test_vibe_with_no_configured_transfers(self):
        """Test that vibes without configured transfers result in no sharing.

        This tests the case where a vibe has no entry in vibe_transfers config.
        In this case, no resources should be shared even if they're shareable.
        """
        # Create environment
        cfg = MettaGridConfig.EmptyRoom(num_agents=2, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#", "#"],
                ["#", "@", "@", "#"],
                ["#", "#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty"},
        )

        cfg.game.resource_names = ["charger", "water", "food"]
        cfg.game.agent.inventory.initial = {"charger": 10, "water": 8, "food": 6}

        # No vibe_transfers configured - no resources will be shared
        cfg.game.agent.vibe_transfers = {}

        cfg.game.actions.move.enabled = True
        cfg.game.actions.noop.enabled = True
        cfg.game.actions.change_vibe.enabled = True
        cfg.game.actions.change_vibe.number_of_vibes = 100

        sim = Simulation(cfg)

        # Get resource indices
        charger_idx = sim.resource_names.index("charger")
        water_idx = sim.resource_names.index("water")
        food_idx = sim.resource_names.index("food")

        # Set agent 0's vibe to "charger"
        sim.agent(0).set_action("change_vibe_charger")
        sim.agent(1).set_action("noop")
        sim.step()

        # Verify vibe is set
        grid_objects = sim.grid_objects()
        agents = sorted([obj for obj in grid_objects.values() if "agent_id" in obj], key=lambda x: x["agent_id"])
        agent0 = agents[0]

        assert agent0["vibe"] == 1, "Agent 0 should have vibe set to 1 (charger)"

        # Have agent 0 move onto agent 1
        # Since no vibe_transfers configured, no sharing should occur
        sim.agent(0).set_action("move_east")
        sim.agent(1).set_action("noop")
        sim.step()

        # Check inventory - nothing should change
        grid_objects_after = sim.grid_objects()
        agents_after = sorted(
            [obj for obj in grid_objects_after.values() if "agent_id" in obj], key=lambda x: x["agent_id"]
        )
        agent0_after = agents_after[0]
        agent1_after = agents_after[1]

        # No sharing should occur
        assert agent0_after["inventory"][charger_idx] == 10, (
            f"Agent 0 should still have 10 charger (no sharing). Has {agent0_after['inventory'][charger_idx]}"
        )
        assert agent0_after["inventory"][water_idx] == 8, (
            f"Agent 0 should still have 8 water (no sharing). Has {agent0_after['inventory'][water_idx]}"
        )
        assert agent0_after["inventory"][food_idx] == 6, (
            f"Agent 0 should still have 6 food (no sharing). Has {agent0_after['inventory'][food_idx]}"
        )
        assert agent1_after["inventory"][charger_idx] == 10
        assert agent1_after["inventory"][water_idx] == 8
        assert agent1_after["inventory"][food_idx] == 6
