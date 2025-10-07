import numpy as np

from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.core import MettaGridCore
from mettagrid.mettagrid_c import dtype_actions


class TestResourceLoss:
    """Test resource loss functionality."""

    def test_resource_loss_prob_1_0_causes_complete_loss(self):
        """Test that resource_loss_prob=1.0 causes all items to be lost in the next timestep."""
        # Create a simple environment with resource_loss_prob=1.0
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#"],
                ["#", "@", "#"],
                ["#", "#", "#"],
            ]
        )
        cfg.game.resource_loss_prob = 1.0
        cfg.game.agent.initial_inventory = {"heart": 5, "battery_blue": 3}
        cfg.game.actions.noop.enabled = True
        env = MettaGridCore(cfg)

        # Reset environment
        obs, info = env.reset()

        # Get the agent's inventory through grid_objects
        grid_objects = env.grid_objects()
        agent_obj = None
        for _obj_id, obj in grid_objects.items():
            if "agent_id" in obj:  # This is an agent
                agent_obj = obj
                break

        assert agent_obj is not None, "Should find an agent in grid objects"

        # Verify agent has the items initially
        inventory = agent_obj["inventory"]
        assert inventory[env.resource_names.index("heart")] == 5, "Agent should have 5 hearts initially"
        assert inventory[env.resource_names.index("battery_blue")] == 3, "Agent should have 3 battery_blue initially"

        # Take a step with noop action
        noop_idx = env.action_names.index("noop")
        actions = np.full(env.num_agents, noop_idx, dtype=dtype_actions)

        obs, rewards, terminals, truncations, info = env.step(actions)

        # After one step with resource_loss_prob=1.0, all items should be lost
        grid_objects_after = env.grid_objects()
        agent_obj_after = None
        for _obj_id, obj in grid_objects_after.items():
            if "agent_id" in obj:  # This is an agent
                agent_obj_after = obj
                break

        assert agent_obj_after is not None, "Should find an agent in grid objects after step"
        inventory_after = agent_obj_after["inventory"]
        assert env.resource_names.index("heart") not in inventory_after, "All hearts should be lost after one step"
        assert env.resource_names.index("battery_blue") not in inventory_after, (
            "All battery_blue should be lost after one step"
        )
        assert len(inventory_after) == 0, "Agent should have no items left"

    def test_resource_loss_prob_0_0_causes_no_loss(self):
        """Test that resource_loss_prob=0.0 causes no items to be lost."""
        # Create a simple environment with resource_loss_prob=0.0
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#"],
                ["#", "@", "#"],
                ["#", "#", "#"],
            ]
        )
        cfg.game.resource_loss_prob = 0.0
        cfg.game.agent.initial_inventory = {"heart": 5, "battery_blue": 3}
        cfg.game.actions.noop.enabled = True
        env = MettaGridCore(cfg)

        # Reset environment
        obs, info = env.reset()

        # Get the agent's inventory through grid_objects
        grid_objects = env.grid_objects()
        agent_obj = None
        for _obj_id, obj in grid_objects.items():
            if "agent_id" in obj:  # This is an agent
                agent_obj = obj
                break

        assert agent_obj is not None, "Should find an agent in grid objects"

        # Verify agent has the items initially
        inventory = agent_obj["inventory"]
        assert inventory[env.resource_names.index("heart")] == 5, "Agent should have 5 hearts initially"
        assert inventory[env.resource_names.index("battery_blue")] == 3, "Agent should have 3 battery_blue initially"

        # Take multiple steps with noop action
        noop_idx = env.action_names.index("noop")
        actions = np.full(env.num_agents, noop_idx, dtype=dtype_actions)

        # Take 10 steps
        for _ in range(10):
            obs, rewards, terminals, truncations, info = env.step(actions)

        # After multiple steps with resource_loss_prob=0.0, items should remain
        grid_objects_after = env.grid_objects()
        agent_obj_after = None
        for _obj_id, obj in grid_objects_after.items():
            if "agent_id" in obj:  # This is an agent
                agent_obj_after = obj
                break

        assert agent_obj_after is not None, "Should find an agent in grid objects after steps"
        inventory_after = agent_obj_after["inventory"]
        assert inventory_after[env.resource_names.index("heart")] == 5, "Hearts should remain after multiple steps"
        assert inventory_after[env.resource_names.index("battery_blue")] == 3, (
            "Battery_blue should remain after multiple steps"
        )

    def test_resource_loss_prob_0_5_causes_partial_loss(self):
        """Test that resource_loss_prob=0.5 causes some items to be lost over multiple steps."""
        # Create a simple environment with resource_loss_prob=0.5
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#"],
                ["#", "@", "#"],
                ["#", "#", "#"],
            ]
        )
        cfg.game.actions.noop.enabled = True
        cfg.game.resource_loss_prob = 0.5
        cfg.game.agent.initial_inventory = {"heart": 100}

        # Create environment with resource_loss_prob=0.5 and initial inventory
        env = MettaGridCore(cfg)

        # Reset environment
        obs, info = env.reset()

        # Get the agent's inventory through grid_objects
        grid_objects = env.grid_objects()
        agent_obj = None
        for _obj_id, obj in grid_objects.items():
            if "agent_id" in obj:  # This is an agent
                agent_obj = obj
                break

        assert agent_obj is not None, "Should find an agent in grid objects"

        # Verify agent has the items initially
        inventory = agent_obj["inventory"]
        assert inventory[env.resource_names.index("heart")] == 100, "Agent should have 100 hearts initially"

        # Take multiple steps with noop action
        noop_idx = env.action_names.index("noop")
        actions = np.full(env.num_agents, noop_idx, dtype=dtype_actions)

        initial_count = inventory[env.resource_names.index("heart")]

        # Take 5 steps
        for step in range(5):
            obs, rewards, terminals, truncations, info = env.step(actions)

            # Get current inventory
            grid_objects_current = env.grid_objects()
            agent_obj_current = None
            for _obj_id, obj in grid_objects_current.items():
                if "agent_id" in obj:  # This is an agent
                    agent_obj_current = obj
                    break

            assert agent_obj_current is not None, "Should find an agent in grid objects"
            inventory_current = agent_obj_current["inventory"]
            current_count = inventory_current[env.resource_names.index("heart")]

            # With 50% loss probability, we should see some loss over multiple steps
            # The exact amount is random, but we should see some reduction
            if step > 0:  # After first step
                assert current_count <= initial_count, f"Item count should not increase at step {step}"

            initial_count = current_count

        # After multiple steps, some items should be lost (but not necessarily all)
        assert initial_count < 100, "Some items should be lost after multiple steps with 50% loss probability"
