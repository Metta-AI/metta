from mettagrid.config.mettagrid_config import MettaGridConfig, ResourceLimitsConfig
from mettagrid.simulator import Action, Simulation


class TestVibeDependentRegeneration:
    """Test vibe-dependent inventory regeneration functionality."""

    def test_vibe_dependent_regen_different_rates(self):
        """Test that different vibes regenerate resources at different rates."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#"],
                ["#", "@", "#"],
                ["#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty"},
        )

        cfg.game.resource_names = ["energy"]
        # Different regen rates for different vibes
        cfg.game.agent.inventory.regen_amounts = {
            "default": {"energy": 2},  # Default vibe: regenerate 2 energy
            "charger": {"energy": 10},  # Charger vibe: regenerate 10 energy
        }
        cfg.game.inventory_regen_interval = 1  # Every timestep
        cfg.game.agent.inventory.initial = {"energy": 0}
        cfg.game.actions.noop.enabled = True
        cfg.game.actions.change_vibe.enabled = True

        sim = Simulation(cfg)

        # Step 1: Default vibe - should regenerate 2 energy
        sim.agent(0).set_action(Action(name="noop"))
        sim.step()

        energy = sim.agent(0).inventory.get("energy", 0)
        assert energy == 2, f"With default vibe, energy should regenerate to 2, got {energy}"

        # Step 2: Change to charger vibe
        sim.agent(0).set_action("change_vibe_charger")
        sim.step()

        # After changing vibe, regen happens with new vibe rate
        energy = sim.agent(0).inventory.get("energy", 0)
        assert energy == 12, f"With charger vibe, energy should regenerate by 10 (2+10=12), got {energy}"

        # Step 3: Stay on charger vibe - should continue regenerating at 10/step
        sim.agent(0).set_action(Action(name="noop"))
        sim.step()

        energy = sim.agent(0).inventory.get("energy", 0)
        assert energy == 22, f"With charger vibe, energy should be 22 (12+10), got {energy}"

        # Step 4: Change back to default vibe
        sim.agent(0).set_action("change_vibe_default")
        sim.step()

        energy = sim.agent(0).inventory.get("energy", 0)
        assert energy == 24, f"With default vibe, energy should be 24 (22+2), got {energy}"

    def test_vibe_dependent_regen_fallback_to_default(self):
        """Test that unconfigured vibes fall back to 'default' regen."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#"],
                ["#", "@", "#"],
                ["#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty"},
        )

        cfg.game.resource_names = ["energy"]
        # Only configure default vibe - other vibes should fall back to it
        cfg.game.agent.inventory.regen_amounts = {
            "default": {"energy": 5},
        }
        cfg.game.inventory_regen_interval = 1
        cfg.game.agent.inventory.initial = {"energy": 0}
        cfg.game.actions.noop.enabled = True
        cfg.game.actions.change_vibe.enabled = True

        sim = Simulation(cfg)

        # Step 1: Default vibe - should regenerate 5 energy
        sim.agent(0).set_action(Action(name="noop"))
        sim.step()

        energy = sim.agent(0).inventory.get("energy", 0)
        assert energy == 5, f"With default vibe, energy should be 5, got {energy}"

        # Step 2: Change to charger vibe (not configured - should fall back to default)
        sim.agent(0).set_action("change_vibe_charger")
        sim.step()

        energy = sim.agent(0).inventory.get("energy", 0)
        assert energy == 10, f"Unconfigured charger vibe should fall back to default (5+5=10), got {energy}"

        # Step 3: Change to another unconfigured vibe
        sim.agent(0).set_action("change_vibe_carbon_a")
        sim.step()

        energy = sim.agent(0).inventory.get("energy", 0)
        assert energy == 15, f"Unconfigured vibe should fall back to default (10+5=15), got {energy}"

    def test_vibe_dependent_regen_no_default(self):
        """Test that vibes without config and no default get no regen."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#"],
                ["#", "@", "#"],
                ["#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty"},
        )

        cfg.game.resource_names = ["energy"]
        # Only configure charger vibe - no default fallback
        cfg.game.agent.inventory.regen_amounts = {
            "charger": {"energy": 10},
        }
        cfg.game.inventory_regen_interval = 1
        cfg.game.agent.inventory.initial = {"energy": 0}
        cfg.game.actions.noop.enabled = True
        cfg.game.actions.change_vibe.enabled = True

        sim = Simulation(cfg)

        # Step 1: Default vibe (not configured, no fallback) - no regeneration
        sim.agent(0).set_action(Action(name="noop"))
        sim.step()

        energy = sim.agent(0).inventory.get("energy", 0)
        assert energy == 0, f"Unconfigured default vibe should not regenerate, got {energy}"

        # Step 2: Change to charger vibe - should regenerate
        sim.agent(0).set_action("change_vibe_charger")
        sim.step()

        energy = sim.agent(0).inventory.get("energy", 0)
        assert energy == 10, f"Charger vibe should regenerate 10, got {energy}"

        # Step 3: Change back to default - no regeneration
        sim.agent(0).set_action("change_vibe_default")
        sim.step()

        energy = sim.agent(0).inventory.get("energy", 0)
        assert energy == 10, f"Default vibe should not regenerate (still 10), got {energy}"


class TestNegativeRegeneration:
    """Test negative inventory regeneration (decay) functionality."""

    def test_negative_regen_decreases_resource(self):
        """Test that negative regen values decrease resources over time."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#"],
                ["#", "@", "#"],
                ["#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty"},
        )

        cfg.game.resource_names = ["energy"]
        # Negative regen - energy decays by 3 per interval
        cfg.game.agent.inventory.regen_amounts = {
            "default": {"energy": -3},
        }
        cfg.game.inventory_regen_interval = 1  # Every timestep
        cfg.game.agent.inventory.initial = {"energy": 20}
        cfg.game.actions.noop.enabled = True

        sim = Simulation(cfg)

        # Step 1: Energy should decrease by 3
        sim.agent(0).set_action(Action(name="noop"))
        sim.step()

        energy = sim.agent(0).inventory.get("energy", 0)
        assert energy == 17, f"Energy should decay to 17, got {energy}"

        # Step 2: Energy should decrease again
        sim.agent(0).set_action(Action(name="noop"))
        sim.step()

        energy = sim.agent(0).inventory.get("energy", 0)
        assert energy == 14, f"Energy should decay to 14, got {energy}"

        # Step 3: Energy should decrease again
        sim.agent(0).set_action(Action(name="noop"))
        sim.step()

        energy = sim.agent(0).inventory.get("energy", 0)
        assert energy == 11, f"Energy should decay to 11, got {energy}"

    def test_negative_regen_floors_at_zero(self):
        """Test that negative regen doesn't go below zero."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#"],
                ["#", "@", "#"],
                ["#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty"},
        )

        cfg.game.resource_names = ["energy"]
        # Large negative regen to test floor at zero
        cfg.game.agent.inventory.regen_amounts = {
            "default": {"energy": -10},
        }
        cfg.game.inventory_regen_interval = 1
        cfg.game.agent.inventory.initial = {"energy": 5}
        cfg.game.actions.noop.enabled = True

        sim = Simulation(cfg)

        # Step 1: Energy should decay to 0 (not -5)
        sim.agent(0).set_action(Action(name="noop"))
        sim.step()

        energy = sim.agent(0).inventory.get("energy", 0)
        assert energy == 0, f"Energy should floor at 0, got {energy}"

        # Step 2: Energy should stay at 0
        sim.agent(0).set_action(Action(name="noop"))
        sim.step()

        energy = sim.agent(0).inventory.get("energy", 0)
        assert energy == 0, f"Energy should remain at 0, got {energy}"

    def test_vibe_dependent_negative_regen(self):
        """Test that different vibes can have different decay rates."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#"],
                ["#", "@", "#"],
                ["#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty"},
        )

        cfg.game.resource_names = ["energy"]
        # Different decay rates for different vibes
        cfg.game.agent.inventory.regen_amounts = {
            "default": {"energy": -2},  # Slow decay
            "charger": {"energy": 5},  # Actually regenerates
        }
        cfg.game.inventory_regen_interval = 1
        cfg.game.agent.inventory.initial = {"energy": 20}
        cfg.game.actions.noop.enabled = True
        cfg.game.actions.change_vibe.enabled = True

        sim = Simulation(cfg)

        # Step 1: Default vibe - should decay by 2
        sim.agent(0).set_action(Action(name="noop"))
        sim.step()

        energy = sim.agent(0).inventory.get("energy", 0)
        assert energy == 18, f"With default vibe, energy should decay to 18, got {energy}"

        # Step 2: Change to charger vibe - should regenerate
        sim.agent(0).set_action("change_vibe_charger")
        sim.step()

        energy = sim.agent(0).inventory.get("energy", 0)
        assert energy == 23, f"With charger vibe, energy should increase to 23 (18+5), got {energy}"

        # Step 3: Change back to default - should decay again
        sim.agent(0).set_action("change_vibe_default")
        sim.step()

        energy = sim.agent(0).inventory.get("energy", 0)
        assert energy == 21, f"With default vibe, energy should decay to 21 (23-2), got {energy}"


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
        cfg.game.agent.inventory.regen_amounts = {"default": {"energy": 5}}  # Regenerate 5 energy
        cfg.game.inventory_regen_interval = 3  # Every 3 timesteps
        cfg.game.agent.inventory.initial = {"energy": 10}  # Start with 10 energy
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
        cfg.game.agent.inventory.regen_amounts = {"default": {"energy": 5}}
        cfg.game.inventory_regen_interval = 0  # Disabled
        cfg.game.agent.inventory.initial = {"energy": 10}
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
        cfg.game.agent.inventory.regen_amounts = {"default": {"energy": 10}}  # Try to add 10
        cfg.game.inventory_regen_interval = 1  # Every timestep
        cfg.game.agent.inventory.initial = {"energy": 95}
        cfg.game.agent.inventory.limits = {
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
