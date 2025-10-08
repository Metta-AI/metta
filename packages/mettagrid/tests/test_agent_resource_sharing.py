import numpy as np

from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.core import MettaGridCore
from mettagrid.mettagrid_c import dtype_actions
from mettagrid.test_support.actions import action_index
from mettagrid.test_support.orientation import Orientation


class TestAgentResourceSharing:
    """Test agent resource sharing functionality when agents use each other."""

    def test_basic_resource_sharing(self):
        """Test that agents can share resources when one moves onto another."""
        # Create a simple environment with 2 agents
        cfg = MettaGridConfig.EmptyRoom(num_agents=2, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#", "#"],
                ["#", "@", "@", "#"],
                ["#", "#", "#", "#"],
            ]
        )

        # Configure resources and sharing
        cfg.game.resource_names = ["energy", "water", "food"]
        cfg.game.agent.initial_inventory = {"energy": 10, "water": 8, "food": 6}
        cfg.game.agent.shareable_resources = ["energy", "water"]  # Only energy and water are shareable

        # Enable the move action to allow agents to interact
        cfg.game.actions.move.enabled = True
        cfg.game.actions.noop.enabled = True

        env = MettaGridCore(cfg)
        obs, info = env.reset()

        # Get initial state
        grid_objects = env.grid_objects()
        agents = []
        for _obj_id, obj in grid_objects.items():
            if "agent_id" in obj:
                agents.append(obj)

        assert len(agents) == 2, "Should find 2 agents"

        # Check initial inventory
        energy_idx = env.resource_names.index("energy")
        water_idx = env.resource_names.index("water")
        food_idx = env.resource_names.index("food")

        agent0 = agents[0]
        agent1 = agents[1]

        assert agent0["inventory"][energy_idx] == 10, "Agent 0 should start with 10 energy"
        assert agent0["inventory"][water_idx] == 8, "Agent 0 should start with 8 water"
        assert agent0["inventory"][food_idx] == 6, "Agent 0 should start with 6 food"

        assert agent1["inventory"][energy_idx] == 10, "Agent 1 should start with 10 energy"
        assert agent1["inventory"][water_idx] == 8, "Agent 1 should start with 8 water"
        assert agent1["inventory"][food_idx] == 6, "Agent 1 should start with 6 food"

        # Have agent 0 move onto agent 1 to trigger onUse
        # Agent 0 is at position (1,1), Agent 1 is at position (1,2)
        # So agent 0 needs to move to the right (East)
        move_idx = action_index(env, "move", Orientation.EAST)
        noop_idx = action_index(env, "noop")
        actions = np.array([move_idx, noop_idx], dtype=dtype_actions)

        obs, rewards, terminals, truncations, info = env.step(actions)

        # Check inventory after sharing
        grid_objects_after = env.grid_objects()
        agents_after = []
        for _obj_id, obj in grid_objects_after.items():
            if "agent_id" in obj:
                agents_after.append(obj)

        agent0_after = agents_after[0]
        agent1_after = agents_after[1]

        assert (agent0_after["r"], agent0_after["c"]) == (1, 1), "Agent 0 should still be at (1,1)"
        assert (agent1_after["r"], agent1_after["c"]) == (1, 2), "Agent 1 should still be at (1,2)"

        # Agent 0 should have given half of shareable resources to agent 1
        # Energy: 10 -> 5 (agent 0), 10 -> 15 (agent 1)
        # Water: 8 -> 4 (agent 0), 8 -> 12 (agent 1)
        # Food: 6 -> 6 (agent 0, unchanged), 6 -> 6 (agent 1, unchanged)
        assert agent0_after["inventory"][energy_idx] == 5, (
            f"Agent 0 should have 5 energy after sharing. Has {agent0_after['inventory'][energy_idx]}"
        )
        assert agent0_after["inventory"][water_idx] == 4, (
            f"Agent 0 should have 4 water after sharing. Has {agent0_after['inventory'][water_idx]}"
        )
        assert agent0_after["inventory"][food_idx] == 6, (
            f"Agent 0 should still have 6 food (not shareable). Has {agent0_after['inventory'][food_idx]}"
        )

        assert agent1_after["inventory"][energy_idx] == 15, (
            f"Agent 1 should have 15 energy after receiving. Has {agent1_after['inventory'][energy_idx]}"
        )
        assert agent1_after["inventory"][water_idx] == 12, (
            f"Agent 1 should have 12 water after receiving. Has {agent1_after['inventory'][water_idx]}"
        )
        assert agent1_after["inventory"][food_idx] == 6, (
            f"Agent 1 should still have 6 food (not shareable). Has {agent1_after['inventory'][food_idx]}"
        )
