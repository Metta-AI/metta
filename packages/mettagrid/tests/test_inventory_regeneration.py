import numpy as np

from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.core import MettaGridCore
from mettagrid.mettagrid_c import dtype_actions


class TestInventoryRegeneration:
    """Test inventory regeneration functionality."""

    def test_energy_regeneration_basic(self):
        """Test that energy regenerates at the specified interval."""
        # Create a simple environment with energy regeneration
        cfg = MettaGridConfig.EmptyRoom(num_agents=2, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#", "#"],
                ["#", "@", "@", "#"],
                ["#", "#", "#", "#"],
            ]
        )

        # Add energy to resources and configure regeneration
        cfg.game.resource_names = ["energy", "heart", "battery_blue"]
        cfg.game.agent.inventory_regen_amounts = {"energy": 5}  # Regenerate 5 energy
        cfg.game.inventory_regen_interval = 3  # Every 3 timesteps
        cfg.game.agent.initial_inventory = {"energy": 10}  # Start with 10 energy
        cfg.game.actions.noop.enabled = True

        env = MettaGridCore(cfg)

        # Reset environment
        obs, info = env.reset()

        # Get initial energy levels
        grid_objects = env.grid_objects()
        agents = []
        for _obj_id, obj in grid_objects.items():
            if "agent_id" in obj:  # This is an agent
                agents.append(obj)

        assert len(agents) == 2, "Should find 2 agents"

        # Check initial energy
        energy_idx = env.resource_names.index("energy")
        for agent in agents:
            assert agent["inventory"][energy_idx] == 10, "Each agent should start with 10 energy"

        # Take steps and verify regeneration
        noop_idx = env.action_names.index("noop")
        actions = np.full(env.num_agents, noop_idx, dtype=dtype_actions)

        # Step 1: No regeneration yet
        obs, rewards, terminals, truncations, info = env.step(actions)
        grid_objects = env.grid_objects()
        agents = [obj for _obj_id, obj in grid_objects.items() if "agent_id" in obj]
        for agent in agents:
            assert agent["inventory"][energy_idx] == 10, "Energy should not regenerate at step 1"

        # Step 2: No regeneration yet
        obs, rewards, terminals, truncations, info = env.step(actions)
        grid_objects = env.grid_objects()
        agents = [obj for _obj_id, obj in grid_objects.items() if "agent_id" in obj]
        for agent in agents:
            assert agent["inventory"][energy_idx] == 10, "Energy should not regenerate at step 2"

        # Step 3: Regeneration should occur (current_step % 3 == 0)
        obs, rewards, terminals, truncations, info = env.step(actions)
        grid_objects = env.grid_objects()
        agents = [obj for _obj_id, obj in grid_objects.items() if "agent_id" in obj]
        for agent in agents:
            assert agent["inventory"][energy_idx] == 15, "Energy should regenerate to 15 at step 3"

        # Step 4: No regeneration
        obs, rewards, terminals, truncations, info = env.step(actions)
        grid_objects = env.grid_objects()
        agents = [obj for _obj_id, obj in grid_objects.items() if "agent_id" in obj]
        for agent in agents:
            assert agent["inventory"][energy_idx] == 15, "Energy should remain at 15 at step 4"

        # Step 5: No regeneration
        obs, rewards, terminals, truncations, info = env.step(actions)
        grid_objects = env.grid_objects()
        agents = [obj for _obj_id, obj in grid_objects.items() if "agent_id" in obj]
        for agent in agents:
            assert agent["inventory"][energy_idx] == 15, "Energy should remain at 15 at step 5"

        # Step 6: Regeneration should occur again
        obs, rewards, terminals, truncations, info = env.step(actions)
        grid_objects = env.grid_objects()
        agents = [obj for _obj_id, obj in grid_objects.items() if "agent_id" in obj]
        for agent in agents:
            assert agent["inventory"][energy_idx] == 20, "Energy should regenerate to 20 at step 6"

    def test_regeneration_disabled_with_zero_interval(self):
        """Test that regeneration is disabled when interval is 0."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#"],
                ["#", "@", "#"],
                ["#", "#", "#"],
            ]
        )

        cfg.game.resource_names = ["energy"]
        cfg.game.agent.inventory_regen_amounts = {"energy": 5}
        cfg.game.inventory_regen_interval = 0  # Disabled
        cfg.game.agent.initial_inventory = {"energy": 10}
        cfg.game.actions.noop.enabled = True

        env = MettaGridCore(cfg)
        obs, info = env.reset()

        energy_idx = env.resource_names.index("energy")

        # Take many steps
        noop_idx = env.action_names.index("noop")
        actions = np.full(env.num_agents, noop_idx, dtype=dtype_actions)

        for _ in range(10):
            obs, rewards, terminals, truncations, info = env.step(actions)
            grid_objects = env.grid_objects()
            agent = next(obj for _obj_id, obj in grid_objects.items() if "agent_id" in obj)
            assert agent["inventory"][energy_idx] == 10, "Energy should not regenerate with interval=0"

    def test_regeneration_with_resource_limits(self):
        """Test that regeneration respects resource limits."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#"],
                ["#", "@", "#"],
                ["#", "#", "#"],
            ]
        )

        cfg.game.resource_names = ["energy"]
        cfg.game.agent.inventory_regen_amounts = {"energy": 10}  # Try to add 10
        cfg.game.inventory_regen_interval = 1  # Every timestep
        cfg.game.agent.initial_inventory = {"energy": 95}
        cfg.game.agent.resource_limits = {"energy": 100}  # Max 100 energy
        cfg.game.actions.noop.enabled = True

        env = MettaGridCore(cfg)
        obs, info = env.reset()

        energy_idx = env.resource_names.index("energy")

        # Take a step - should regenerate but cap at 100
        noop_idx = env.action_names.index("noop")
        actions = np.full(env.num_agents, noop_idx, dtype=dtype_actions)

        obs, rewards, terminals, truncations, info = env.step(actions)
        grid_objects = env.grid_objects()
        agent = next(obj for _obj_id, obj in grid_objects.items() if "agent_id" in obj)
        assert agent["inventory"][energy_idx] == 100, "Energy should cap at 100 (limit)"

        # Take another step - should stay at 100
        obs, rewards, terminals, truncations, info = env.step(actions)
        grid_objects = env.grid_objects()
        agent = next(obj for _obj_id, obj in grid_objects.items() if "agent_id" in obj)
        assert agent["inventory"][energy_idx] == 100, "Energy should remain at 100"
