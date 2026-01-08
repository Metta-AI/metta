from mettagrid.config.mettagrid_config import ChestConfig, InventoryConfig, MettaGridConfig, ResourceLimitsConfig
from mettagrid.simulator import Simulation


class TestChest:
    """Test chest deposit and withdrawal functionality."""

    def test_chest_deposit(self):
        """Test that deposit/withdrawal work with vibe-based transfers."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True)

        cfg.game.resource_names = ["gold"]
        cfg.game.agent.inventory.initial = {"gold": 5}

        cfg.game.objects["chest"] = ChestConfig(
            vibe_transfers={
                "down": {"gold": 1},  # When showing deposit vibe, deposit 1 gold
                "up": {"gold": -1},  # When showing withdraw vibe, withdraw 1 gold
            },
            inventory=InventoryConfig(
                limits={
                    "gold": ResourceLimitsConfig(limit=100, resources=["gold"]),
                },
            ),
        )

        cfg = cfg.with_ascii_map(
            [
                ["#", "#", "#", "#", "#"],
                ["#", ".", ".", ".", "#"],
                ["#", ".", "C", ".", "#"],
                ["#", ".", "@", ".", "#"],
                ["#", "#", "#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty", "C": "chest"},
        )

        # Enable actions
        cfg.game.actions.change_vibe.enabled = True
        cfg.game.actions.move.enabled = True

        sim = Simulation(cfg)

        gold_idx = sim.resource_names.index("gold")

        sim.agent(0).set_action("change_vibe_down")
        sim.step()

        # Try to move north (to chest position) - should trigger deposit
        sim.agent(0).set_action("move_north")
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
        sim.agent(0).set_action("change_vibe_up")
        sim.step()

        # Try to move INTO the chest position again to trigger withdrawal
        sim.agent(0).set_action("move_north")
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
