"""Test vibe-based agent resource sharing functionality.

This test verifies that agents share resources based on their current vibe:
- When vibe=0 (neutral): shares half of ALL shareable resources (original behavior)
- When vibe!=0: shares half of ONLY the resource matching the vibe name
"""

from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.simulator import Simulation


class TestVibeBasedSharing:
    """Test vibe-based resource sharing when agents use each other."""

    def test_vibe_based_single_resource_sharing(self):
        """Test that agents share only the vibed resource when vibe is set."""
        # Create a simple environment with 2 agents
        cfg = MettaGridConfig.EmptyRoom(num_agents=2, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#", "#"],
                ["#", "@", "@", "#"],
                ["#", "#", "#", "#"],
            ]
        )

        # Configure resources and sharing
        # Resource names match some vibe names (e.g., "charger" vibe -> "charger" resource)
        cfg.game.resource_names = ["charger", "water", "food"]
        cfg.game.agent.initial_inventory = {"charger": 10, "water": 8, "food": 6}
        cfg.game.agent.shareable_resources = ["charger", "water"]  # Only charger and water are shareable

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

        # With vibe-based sharing:
        # - Only charger should be shared (half of 10 = 5)
        # - Water should NOT be shared (stays at 8 for both)
        # - Food is not shareable anyway
        assert agent0_after["inventory"][charger_idx] == 5, (
            f"Agent 0 should have 5 charger after vibe-based sharing. Has {agent0_after['inventory'][charger_idx]}"
        )
        assert agent0_after["inventory"][water_idx] == 8, (
            f"Agent 0 should still have 8 water (vibe-based sharing only affects charger). "
            f"Has {agent0_after['inventory'][water_idx]}"
        )
        assert agent0_after["inventory"][food_idx] == 6, (
            f"Agent 0 should still have 6 food (not shareable). Has {agent0_after['inventory'][food_idx]}"
        )

        assert agent1_after["inventory"][charger_idx] == 15, (
            f"Agent 1 should have 15 charger after receiving half. Has {agent1_after['inventory'][charger_idx]}"
        )
        assert agent1_after["inventory"][water_idx] == 8, (
            f"Agent 1 should still have 8 water (not shared via vibe). Has {agent1_after['inventory'][water_idx]}"
        )
        assert agent1_after["inventory"][food_idx] == 6, (
            f"Agent 1 should still have 6 food (not shareable). Has {agent1_after['inventory'][food_idx]}"
        )

    def test_neutral_vibe_shares_all_resources(self):
        """Test that vibe=0 (neutral) still shares all shareable resources (backward compatibility)."""
        # Create a simple environment with 2 agents
        cfg = MettaGridConfig.EmptyRoom(num_agents=2, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#", "#"],
                ["#", "@", "@", "#"],
                ["#", "#", "#", "#"],
            ]
        )

        # Configure resources
        cfg.game.resource_names = ["charger", "water", "food"]
        cfg.game.agent.initial_inventory = {"charger": 10, "water": 8, "food": 6}
        cfg.game.agent.shareable_resources = ["charger", "water"]

        # Enable move action
        cfg.game.actions.move.enabled = True
        cfg.game.actions.noop.enabled = True

        sim = Simulation(cfg)

        # Get resource indices
        charger_idx = sim.resource_names.index("charger")
        water_idx = sim.resource_names.index("water")
        food_idx = sim.resource_names.index("food")

        # Get agent states - vibe should be 0 by default
        grid_objects = sim.grid_objects()
        agents = sorted([obj for obj in grid_objects.values() if "agent_id" in obj], key=lambda x: x["agent_id"])
        agent0 = agents[0]

        # Vibe should be 0 (neutral/default) unless explicitly set
        assert agent0.get("vibe", 0) == 0, "Agent 0 should have neutral vibe (0) by default"

        # Have agent 0 move onto agent 1 with neutral vibe
        # This should share ALL shareable resources (original behavior)
        sim.agent(0).set_action("move_east")
        sim.agent(1).set_action("noop")
        sim.step()

        # Check inventory after sharing
        grid_objects_after = sim.grid_objects()
        agents_after = sorted(
            [obj for obj in grid_objects_after.values() if "agent_id" in obj], key=lambda x: x["agent_id"]
        )
        agent0_after = agents_after[0]
        agent1_after = agents_after[1]

        # With neutral vibe, should share half of ALL shareable resources:
        # Charger: 10 -> 5 (agent 0), 10 -> 15 (agent 1)
        # Water: 8 -> 4 (agent 0), 8 -> 12 (agent 1)
        # Food: unchanged (not shareable)
        assert agent0_after["inventory"][charger_idx] == 5, (
            f"Agent 0 should have 5 charger after sharing all. Has {agent0_after['inventory'][charger_idx]}"
        )
        assert agent0_after["inventory"][water_idx] == 4, (
            f"Agent 0 should have 4 water after sharing all. Has {agent0_after['inventory'][water_idx]}"
        )
        assert agent0_after["inventory"][food_idx] == 6, (
            f"Agent 0 should still have 6 food (not shareable). Has {agent0_after['inventory'][food_idx]}"
        )

        assert agent1_after["inventory"][charger_idx] == 15, (
            f"Agent 1 should have 15 charger after receiving. Has {agent1_after['inventory'][charger_idx]}"
        )
        assert agent1_after["inventory"][water_idx] == 12, (
            f"Agent 1 should have 12 water after receiving. Has {agent1_after['inventory'][water_idx]}"
        )
        assert agent1_after["inventory"][food_idx] == 6, (
            f"Agent 1 should still have 6 food (not shareable). Has {agent1_after['inventory'][food_idx]}"
        )

    def test_vibe_with_non_shareable_resource_fallback(self):
        """Test that vibing a non-shareable resource falls back to sharing all shareable resources."""
        # Create environment
        cfg = MettaGridConfig.EmptyRoom(num_agents=2, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#", "#"],
                ["#", "@", "@", "#"],
                ["#", "#", "#", "#"],
            ]
        )

        # food is a vibe but NOT shareable
        cfg.game.resource_names = ["charger", "water", "food"]
        cfg.game.agent.initial_inventory = {"charger": 10, "water": 8, "food": 6}
        cfg.game.agent.shareable_resources = ["charger", "water"]  # food is not shareable

        cfg.game.actions.move.enabled = True
        cfg.game.actions.noop.enabled = True
        cfg.game.actions.change_vibe.enabled = True
        cfg.game.actions.change_vibe.number_of_vibes = 100

        sim = Simulation(cfg)

        # Get resource indices
        charger_idx = sim.resource_names.index("charger")
        water_idx = sim.resource_names.index("water")
        food_idx = sim.resource_names.index("food")

        # Set agent 0's vibe to "food" (which is not shareable)
        # According to vibes.py, we need to find the right vibe ID
        # For this test, we'll assume there's a vibe action for food or use a different vibe
        # Let's use "up" vibe which doesn't match any resource
        sim.agent(0).set_action("change_vibe_up")
        sim.agent(1).set_action("noop")
        sim.step()

        # Have agent 0 move onto agent 1
        # Since "up" vibe doesn't match any shareable resource, should fall back to sharing all
        sim.agent(0).set_action("move_east")
        sim.agent(1).set_action("noop")
        sim.step()

        # Check inventory after sharing
        grid_objects_after = sim.grid_objects()
        agents_after = sorted(
            [obj for obj in grid_objects_after.values() if "agent_id" in obj], key=lambda x: x["agent_id"]
        )
        agent0_after = agents_after[0]
        agent1_after = agents_after[1]

        # Should fall back to sharing ALL shareable resources
        assert agent0_after["inventory"][charger_idx] == 5, (
            f"Agent 0 should have 5 charger (fallback to sharing all). Has {agent0_after['inventory'][charger_idx]}"
        )
        assert agent0_after["inventory"][water_idx] == 4, (
            f"Agent 0 should have 4 water (fallback to sharing all). Has {agent0_after['inventory'][water_idx]}"
        )
        assert agent1_after["inventory"][charger_idx] == 15
        assert agent1_after["inventory"][water_idx] == 12
