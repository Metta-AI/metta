from mettagrid.config.mettagrid_config import ChestConfig, MettaGridConfig
from mettagrid.simulator import Simulation


class TestChest:
    """Test chest deposit and withdrawal functionality."""

    def test_chest_deposit(self):
        """Test that deposit/withdrawal work with vibe-based transfers."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True)

        cfg.game.resource_names = ["gold"]
        cfg.game.agent.initial_inventory = {"gold": 5}

        # Define vibes for deposit and withdrawal
        cfg.game.vibe_names = ["neutral", "deposit", "withdraw"]

        cfg.game.objects["chest"] = ChestConfig(
            map_char="C",
            name="chest",
            vibe_transfers={
                "deposit": {"gold": 1},  # When showing deposit vibe, deposit 1 gold
                "withdraw": {"gold": -1},  # When showing withdraw vibe, withdraw 1 gold
            },
            resource_limits={"gold": 100},  # Chest can hold up to 100 gold
        )

        cfg = cfg.with_ascii_map(
            [
                ["#", "#", "#", "#", "#"],
                ["#", ".", ".", ".", "#"],
                ["#", ".", "C", ".", "#"],
                ["#", ".", "@", ".", "#"],
                ["#", "#", "#", "#", "#"],
            ]
        )

        # Enable actions
        cfg.game.actions.change_vibe.enabled = True
        cfg.game.actions.change_vibe.number_of_vibes = 3  # 0 (no vibe), 1 (deposit), 2 (withdraw)
        cfg.game.actions.move.enabled = True

        sim = Simulation(cfg)

        gold_idx = sim.resource_names.index("gold")

        # Agent starts at (row=3, col=2), chest is at (row=2, col=2)
        # Change vibe to deposit first
        sim.agent(0).set_action("change_vibe_deposit")
        sim.step()

        # Try to move south (to chest position) - should trigger deposit
        sim.agent(0).set_action("move_south")
        sim.step()

        # Check deposit happened
        grid_objects = sim.grid_objects()
        agent = next(obj for _obj_id, obj in grid_objects.items() if "agent_id" in obj)
        chest = next(obj for _obj_id, obj in grid_objects.items() if obj["type_name"] == "chest")

        assert agent["inventory"].get(gold_idx, 0) == 4, (
            f"Agent should have 4 gold. Has {agent['inventory'].get(gold_idx, 0)}"
        )
        assert chest["inventory"].get(gold_idx, 0) == 1, (
            f"Chest should have 1 gold. Has {chest['inventory'].get(gold_idx, 0)}"
        )

        # Change vibe to withdraw
        sim.agent(0).set_action("change_vibe_withdraw")
        sim.step()

        # Try to move INTO the chest position again to trigger withdrawal
        sim.agent(0).set_action("move_south")
        sim.step()

        # Check withdrawal happened
        grid_objects_after = sim.grid_objects()
        agent_after = next(obj for _obj_id, obj in grid_objects_after.items() if "agent_id" in obj)
        chest_after = next(obj for _obj_id, obj in grid_objects_after.items() if obj["type_name"] == "chest")

        assert agent_after["inventory"].get(gold_idx, 0) == 5, (
            f"Agent should have 5 gold after withdrawal, has {agent_after['inventory'].get(gold_idx, 0)}"
        )
        assert chest_after["inventory"].get(gold_idx, 0) == 0, (
            f"Chest should have 0 gold after withdrawal, has {chest_after['inventory'].get(gold_idx, 0)}"
        )
