from mettagrid.config.mettagrid_config import MettaGridConfig, ResourceLimitsConfig
from mettagrid.simulator import Action, Simulation


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
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty"},
        )

        # Add energy to resources and configure regeneration
        cfg.game.resource_names = ["energy", "heart", "battery_blue"]
        cfg.game.agent.inventory_regen_amounts = {"default": {"energy": 5}}  # Regenerate 5 energy
        cfg.game.inventory_regen_interval = 3  # Every 3 timesteps
        cfg.game.agent.initial_inventory = {"energy": 10}  # Start with 10 energy
        cfg.game.actions.noop.enabled = True

        sim = Simulation(cfg)

        # Reset environment

        # Get initial energy levels
        grid_objects = sim.grid_objects()
        agents = []
        for _obj_id, obj in grid_objects.items():
            if "agent_id" in obj:  # This is an agent
                agents.append(obj)

        assert len(agents) == 2, "Should find 2 agents"

        # Check initial energy
        energy_idx = sim.resource_names.index("energy")
        for agent in agents:
            assert agent["inventory"][energy_idx] == 10, "Each agent should start with 10 energy"

        # Take steps and verify regeneration using SimulationAgent API

        # Step 1: No regeneration yet
        for i in range(sim.num_agents):
            sim.agent(i).set_action(Action(name="noop"))
        sim.step()

        for i in range(sim.num_agents):
            energy = sim.agent(i).inventory.get("energy", 0)
            assert energy == 10, f"Agent {i} energy should not regenerate at step 1, got {energy}"

        # Step 2: No regeneration yet
        for i in range(sim.num_agents):
            sim.agent(i).set_action(Action(name="noop"))
        sim.step()

        for i in range(sim.num_agents):
            energy = sim.agent(i).inventory.get("energy", 0)
            assert energy == 10, f"Agent {i} energy should not regenerate at step 2, got {energy}"

        # Step 3: Regeneration should occur (current_step % 3 == 0)
        for i in range(sim.num_agents):
            sim.agent(i).set_action(Action(name="noop"))
        sim.step()

        for i in range(sim.num_agents):
            energy = sim.agent(i).inventory.get("energy", 0)
            assert energy == 15, f"Agent {i} energy should regenerate to 15 at step 3, got {energy}"

        # Step 4: No regeneration
        for i in range(sim.num_agents):
            sim.agent(i).set_action(Action(name="noop"))
        sim.step()

        for i in range(sim.num_agents):
            energy = sim.agent(i).inventory.get("energy", 0)
            assert energy == 15, f"Agent {i} energy should remain at 15 at step 4, got {energy}"

        # Step 5: No regeneration
        for i in range(sim.num_agents):
            sim.agent(i).set_action(Action(name="noop"))
        sim.step()

        for i in range(sim.num_agents):
            energy = sim.agent(i).inventory.get("energy", 0)
            assert energy == 15, f"Agent {i} energy should remain at 15 at step 5, got {energy}"

        # Step 6: Regeneration should occur again
        for i in range(sim.num_agents):
            sim.agent(i).set_action(Action(name="noop"))
        sim.step()

        for i in range(sim.num_agents):
            energy = sim.agent(i).inventory.get("energy", 0)
            assert energy == 20, f"Agent {i} energy should regenerate to 20 at step 6, got {energy}"

    def test_regeneration_disabled_with_zero_interval(self):
        """Test that regeneration is disabled when interval is 0."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#"],
                ["#", "@", "#"],
                ["#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty"},
        )

        cfg.game.resource_names = ["energy"]
        cfg.game.agent.inventory_regen_amounts = {"default": {"energy": 5}}
        cfg.game.inventory_regen_interval = 0  # Disabled
        cfg.game.agent.initial_inventory = {"energy": 10}
        cfg.game.actions.noop.enabled = True

        sim = Simulation(cfg)

        # Take many steps
        for _ in range(10):
            sim.agent(0).set_action(Action(name="noop"))
            sim.step()
            energy = sim.agent(0).inventory.get("energy", 0)
            assert energy == 10, f"Energy should not regenerate with interval=0, got {energy}"

    def test_regeneration_with_resource_limits(self):
        """Test that regeneration respects resource limits."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#"],
                ["#", "@", "#"],
                ["#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty"},
        )

        cfg.game.resource_names = ["energy"]
        cfg.game.agent.inventory_regen_amounts = {"default": {"energy": 10}}  # Try to add 10
        cfg.game.inventory_regen_interval = 1  # Every timestep
        cfg.game.agent.initial_inventory = {"energy": 95}
        cfg.game.agent.resource_limits = {
            "energy": ResourceLimitsConfig(limit=100, resources=["energy"]),  # Max 100 energy
        }
        cfg.game.actions.noop.enabled = True

        sim = Simulation(cfg)

        # Take a step - should regenerate but cap at 100
        sim.agent(0).set_action(Action(name="noop"))
        sim.step()

        energy = sim.agent(0).inventory.get("energy", 0)
        assert energy == 100, f"Energy should cap at 100 (limit), got {energy}"

        # Take another step - should stay at 100
        sim.agent(0).set_action(Action(name="noop"))
        sim.step()

        energy = sim.agent(0).inventory.get("energy", 0)
        assert energy == 100, f"Energy should remain at 100, got {energy}"

    def test_vibe_dependent_regeneration_default_fallback(self):
        """Test that default vibe is used as fallback when agent's vibe isn't configured."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#"],
                ["#", "@", "#"],
                ["#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty"},
        )

        cfg.game.resource_names = ["energy"]
        # Only configure "default" - agent starts with vibe 0 which is "default"
        cfg.game.agent.inventory_regen_amounts = {"default": {"energy": 5}}
        cfg.game.inventory_regen_interval = 1  # Every timestep
        cfg.game.agent.initial_inventory = {"energy": 10}
        cfg.game.actions.noop.enabled = True

        sim = Simulation(cfg)

        # Take a step - should regenerate using default
        sim.agent(0).set_action(Action(name="noop"))
        sim.step()

        energy = sim.agent(0).inventory.get("energy", 0)
        assert energy == 15, f"Energy should regenerate to 15 using default vibe, got {energy}"

    def test_vibe_dependent_regeneration_specific_vibe(self):
        """Test that vibe-specific regen amounts are used when agent has that vibe."""
        from mettagrid.config.mettagrid_config import ChangeVibeActionConfig

        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#"],
                ["#", "@", "#"],
                ["#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty"},
        )

        cfg.game.resource_names = ["energy"]
        # Configure different regen for default vs charger vibe
        cfg.game.agent.inventory_regen_amounts = {
            "default": {"energy": 5},
            "charger": {"energy": 20},
        }
        cfg.game.inventory_regen_interval = 1  # Every timestep
        cfg.game.agent.initial_inventory = {"energy": 10}
        cfg.game.actions.noop.enabled = True
        cfg.game.actions.change_vibe = ChangeVibeActionConfig(enabled=True, number_of_vibes=10)

        sim = Simulation(cfg)

        # Take a step with default vibe - should get 5 energy
        sim.agent(0).set_action(Action(name="noop"))
        sim.step()

        energy = sim.agent(0).inventory.get("energy", 0)
        assert energy == 15, f"Energy should regenerate to 15 with default vibe, got {energy}"

        # Change to charger vibe
        sim.agent(0).set_action(Action(name="change_vibe_charger"))
        sim.step()

        # Now energy should have regenerated by 20 (charger rate)
        energy = sim.agent(0).inventory.get("energy", 0)
        assert energy == 35, f"Energy should regenerate to 35 with charger vibe, got {energy}"

        # Take another step with charger vibe
        sim.agent(0).set_action(Action(name="noop"))
        sim.step()

        energy = sim.agent(0).inventory.get("energy", 0)
        assert energy == 55, f"Energy should regenerate to 55 with charger vibe, got {energy}"

    def test_vibe_dependent_regeneration_no_regen_for_unconfigured_vibe(self):
        """Test that no regeneration occurs when vibe isn't configured and no default exists."""
        from mettagrid.config.mettagrid_config import ChangeVibeActionConfig

        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#"],
                ["#", "@", "#"],
                ["#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty"},
        )

        cfg.game.resource_names = ["energy"]
        # Only configure "charger" - no default
        cfg.game.agent.inventory_regen_amounts = {
            "charger": {"energy": 20},
        }
        cfg.game.inventory_regen_interval = 1  # Every timestep
        cfg.game.agent.initial_inventory = {"energy": 10}
        cfg.game.actions.noop.enabled = True
        cfg.game.actions.change_vibe = ChangeVibeActionConfig(enabled=True, number_of_vibes=10)

        sim = Simulation(cfg)

        # Take a step with default vibe (vibe 0) - no regeneration since default is not configured
        sim.agent(0).set_action(Action(name="noop"))
        sim.step()

        energy = sim.agent(0).inventory.get("energy", 0)
        assert energy == 10, f"Energy should not regenerate without default config, got {energy}"

        # Change to charger vibe
        sim.agent(0).set_action(Action(name="change_vibe_charger"))
        sim.step()

        # Now energy should have regenerated by 20 (charger rate)
        energy = sim.agent(0).inventory.get("energy", 0)
        assert energy == 30, f"Energy should regenerate to 30 with charger vibe, got {energy}"
