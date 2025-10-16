import numpy as np

from mettagrid.config.mettagrid_config import ChestConfig, MettaGridConfig
from mettagrid.core import MettaGridCore
from mettagrid.mettagrid_c import dtype_actions
from mettagrid.test_support.actions import action_index
from mettagrid.test_support.orientation import Orientation


class TestChest:
    """Test chest deposit and withdrawal functionality."""

    def test_chest_deposit(self):
        """Test that deposit/withdrawal only work from configured positions."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True)

        cfg.game.objects["chest"] = ChestConfig(
            type_id=10,
            resource_type="gold",
            map_char="C",
            name="chest",
            position_deltas=[("N", 1), ("S", -1)],  # N=deposit 1, S=withdraw 1
        )

        cfg = cfg.with_ascii_map(
            [
                ["#", "#", "#", "#", "#"],
                ["#", ".", "@", ".", "#"],
                ["#", ".", "C", ".", "#"],
                ["#", ".", ".", ".", "#"],
                ["#", "#", "#", "#", "#"],
            ]
        )

        cfg.game.resource_names = ["gold"]
        cfg.game.agent.initial_inventory = {"gold": 5}

        # Configure chest with specific positions only
        cfg.game.actions.move.enabled = True

        env = MettaGridCore(cfg)
        obs, info = env.reset()

        gold_idx = env.resource_names.index("gold")

        # Agent starts at (3,2), chest is at (2,2)
        # Agent is south of chest (withdrawal position)

        # Try to move south (to chest position) - should trigger deposit
        actions = np.array([action_index(env, "move", Orientation.SOUTH)], dtype=dtype_actions)
        obs, rewards, terminals, truncations, info = env.step(actions)

        # Check deposit happened
        grid_objects = env.grid_objects()
        agent = next(obj for _obj_id, obj in grid_objects.items() if "agent_id" in obj)
        chest = next(obj for _obj_id, obj in grid_objects.items() if obj["type"] == 10)

        assert agent["inventory"].get(gold_idx, 0) == 4, (
            f"Agent should have 4 gold. Has {agent['inventory'].get(gold_idx, 0)}"
        )
        assert chest["inventory"].get(gold_idx, 0) == 1, (
            f"Chest should have 1 gold. Has {chest['inventory'].get(gold_idx, 0)}"
        )

        # Move around to south position to withdraw
        actions = np.array([action_index(env, "move", Orientation.WEST)], dtype=dtype_actions)
        obs, rewards, terminals, truncations, info = env.step(actions)

        # Then south
        actions = np.array([action_index(env, "move", Orientation.SOUTH)], dtype=dtype_actions)
        obs, rewards, terminals, truncations, info = env.step(actions)
        actions = np.array([action_index(env, "move", Orientation.SOUTH)], dtype=dtype_actions)
        obs, rewards, terminals, truncations, info = env.step(actions)

        # Then east to be south of chest
        actions = np.array([action_index(env, "move", Orientation.EAST)], dtype=dtype_actions)
        obs, rewards, terminals, truncations, info = env.step(actions)

        # Now move north to chest (from withdrawal position)
        actions = np.array([action_index(env, "move", Orientation.NORTH)], dtype=dtype_actions)
        obs, rewards, terminals, truncations, info = env.step(actions)

        # Check withdrawal happened
        grid_objects_after = env.grid_objects()
        agent_after = next(obj for _obj_id, obj in grid_objects_after.items() if "agent_id" in obj)
        chest_after = next(obj for _obj_id, obj in grid_objects_after.items() if obj["type"] == 10)

        assert agent_after["inventory"].get(gold_idx, 0) == 5, (
            f"Agent should have 5 gold after withdrawal, has {agent_after['inventory'].get(gold_idx, 0)}"
        )
        assert chest_after["inventory"].get(gold_idx, 0) == 0, (
            f"Chest should have 0 gold after withdrawal, has {chest_after['inventory'].get(gold_idx, 0)}"
        )

    def test_chest_partial_transfers(self):
        """Test that partial transfers work when resources are limited."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True)

        cfg.game.resource_names = ["gold"]
        cfg.game.agent.initial_inventory = {"gold": 3}  # Agent starts with only 3 gold

        # Configure chest with larger deltas than available resources
        cfg.game.objects["chest"] = ChestConfig(
            type_id=10,
            name="chest",
            resource_type="gold",
            map_char="C",
            position_deltas=[("N", 5), ("S", -5)],  # N=deposit 5, S=withdraw 5
            initial_inventory=2,  # Chest starts with 2 gold
        )

        cfg = cfg.with_ascii_map(
            [
                ["#", "#", "#", "#", "#"],
                ["#", ".", "@", ".", "#"],
                ["#", ".", "C", ".", "#"],
                ["#", ".", ".", ".", "#"],
                ["#", "#", "#", "#", "#"],
            ]
        )

        cfg.game.actions.move.enabled = True

        env = MettaGridCore(cfg)
        obs, info = env.reset()

        gold_idx = env.resource_names.index("gold")

        # Agent starts at (1,2), chest is at (2,2)
        # Agent is north of chest (deposit position)

        # Try to deposit 5 gold, but agent only has 3
        actions = np.array([action_index(env, "move", Orientation.SOUTH)], dtype=dtype_actions)
        obs, rewards, terminals, truncations, info = env.step(actions)

        # Check partial deposit happened
        grid_objects = env.grid_objects()
        agent = next(obj for _obj_id, obj in grid_objects.items() if "agent_id" in obj)
        chest = next(obj for _obj_id, obj in grid_objects.items() if obj["type"] == 10)

        assert agent["inventory"].get(gold_idx, 0) == 0, (
            f"Agent should have 0 gold (deposited all 3). Has {agent['inventory'].get(gold_idx, 0)}"
        )
        assert chest["inventory"].get(gold_idx, 0) == 5, (
            f"Chest should have 5 gold (initial 2 + deposited 3). Has {chest['inventory'].get(gold_idx, 0)}"
        )

        # Move around to south position to withdraw
        actions = np.array([action_index(env, "move", Orientation.WEST)], dtype=dtype_actions)
        obs, rewards, terminals, truncations, info = env.step(actions)
        actions = np.array([action_index(env, "move", Orientation.SOUTH)], dtype=dtype_actions)
        obs, rewards, terminals, truncations, info = env.step(actions)
        actions = np.array([action_index(env, "move", Orientation.SOUTH)], dtype=dtype_actions)
        obs, rewards, terminals, truncations, info = env.step(actions)
        actions = np.array([action_index(env, "move", Orientation.EAST)], dtype=dtype_actions)
        obs, rewards, terminals, truncations, info = env.step(actions)

        # Try to withdraw 5 gold, chest has exactly 5
        actions = np.array([action_index(env, "move", Orientation.NORTH)], dtype=dtype_actions)
        obs, rewards, terminals, truncations, info = env.step(actions)

        # Check full withdrawal happened
        grid_objects_after = env.grid_objects()
        agent_after = next(obj for _obj_id, obj in grid_objects_after.items() if "agent_id" in obj)
        chest_after = next(obj for _obj_id, obj in grid_objects_after.items() if obj["type"] == 10)

        assert agent_after["inventory"].get(gold_idx, 0) == 5, (
            f"Agent should have 5 gold after withdrawal, has {agent_after['inventory'].get(gold_idx, 0)}"
        )
        assert chest_after["inventory"].get(gold_idx, 0) == 0, (
            f"Chest should have 0 gold after withdrawal, has {chest_after['inventory'].get(gold_idx, 0)}"
        )

        # Try to withdraw again when chest is empty
        actions = np.array([action_index(env, "move", Orientation.SOUTH)], dtype=dtype_actions)
        obs, rewards, terminals, truncations, info = env.step(actions)
        actions = np.array([action_index(env, "move", Orientation.NORTH)], dtype=dtype_actions)
        obs, rewards, terminals, truncations, info = env.step(actions)

        # Check nothing changed (no resources to withdraw)
        grid_objects_final = env.grid_objects()
        agent_final = next(obj for _obj_id, obj in grid_objects_final.items() if "agent_id" in obj)
        chest_final = next(obj for _obj_id, obj in grid_objects_final.items() if obj["type"] == 10)

        assert agent_final["inventory"].get(gold_idx, 0) == 5, (
            f"Agent should still have 5 gold, has {agent_final['inventory'].get(gold_idx, 0)}"
        )
        assert chest_final["inventory"].get(gold_idx, 0) == 0, (
            f"Chest should still have 0 gold, has {chest_final['inventory'].get(gold_idx, 0)}"
        )
