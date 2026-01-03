"""Test health system functionality."""

from mettagrid.config.mettagrid_config import (
    ClearInventoryMutation,
    HealthConfig,
    MettaGridConfig,
    ResourceDeltaMutation,
)
from mettagrid.simulator import Action, Simulation


class TestHealth:
    """Test health configuration and behavior."""

    def test_health_config_creation(self):
        """Test that health config can be created and accessed."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#"],
                ["#", "@", "#"],
                ["#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty"},
        )

        cfg.game.resource_names = ["battery", "weapon", "shield", "hp"]
        cfg.game.agent.health = HealthConfig(
            health_resource="hp",
            on_damage=[ResourceDeltaMutation(target="actor", deltas={"weapon": -1})],
        )

        # Verify the config is set correctly
        assert cfg.game.agent.health is not None
        assert cfg.game.agent.health.health_resource == "hp"
        assert len(cfg.game.agent.health.on_damage) == 1

    def test_health_triggers_when_zero(self):
        """Test that on_damage triggers when health reaches 0."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#"],
                ["#", "@", "#"],
                ["#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty"},
        )

        cfg.game.resource_names = ["battery", "weapon", "shield", "hp"]
        # Start with hp at 0 so it triggers immediately
        cfg.game.agent.inventory.initial = {"battery": 3, "weapon": 3, "shield": 3, "hp": 0}
        cfg.game.agent.health = HealthConfig(
            health_resource="hp",
            on_damage=[ResourceDeltaMutation(target="actor", deltas={"weapon": -1})],
        )
        cfg.game.actions.noop.enabled = True

        sim = Simulation(cfg)
        agent = sim.agent(0)

        # Check initial inventory
        initial_weapon = agent.inventory.get("weapon", 0)
        assert initial_weapon == 3

        # Take a step - health damage should trigger since hp is 0
        agent.set_action(Action(name="noop"))
        sim.step()

        # Weapon should be reduced by 1 from the on_damage mutation
        new_weapon = agent.inventory.get("weapon", 0)
        assert new_weapon == 2, f"Weapon should be 2 after damage trigger, got {new_weapon}"

    def test_health_not_triggered_when_positive(self):
        """Test that on_damage doesn't trigger when health is positive."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#"],
                ["#", "@", "#"],
                ["#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty"},
        )

        cfg.game.resource_names = ["battery", "weapon", "shield", "hp"]
        cfg.game.agent.inventory.initial = {"battery": 3, "weapon": 3, "shield": 3, "hp": 10}
        cfg.game.agent.health = HealthConfig(
            health_resource="hp",
            on_damage=[ResourceDeltaMutation(target="actor", deltas={"weapon": -1})],
        )
        cfg.game.actions.noop.enabled = True

        sim = Simulation(cfg)
        agent = sim.agent(0)

        # Take a step - health damage should NOT trigger since hp > 0
        agent.set_action(Action(name="noop"))
        sim.step()

        # Weapon should be unchanged
        new_weapon = agent.inventory.get("weapon", 0)
        assert new_weapon == 3, f"Weapon should still be 3, got {new_weapon}"

    def test_health_with_regen(self):
        """Test that health damage works with inventory regeneration."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#"],
                ["#", "@", "#"],
                ["#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty"},
        )

        cfg.game.resource_names = ["battery", "weapon", "shield", "hp"]
        cfg.game.agent.inventory.initial = {"battery": 5, "weapon": 5, "shield": 5, "hp": 3}
        cfg.game.agent.health = HealthConfig(
            health_resource="hp",
            on_damage=[ResourceDeltaMutation(target="actor", deltas={"weapon": -1})],
        )
        # HP decays by 1 each step
        cfg.game.agent.inventory.regen_amounts = {"default": {"hp": -1}}
        cfg.game.inventory_regen_interval = 1
        cfg.game.actions.noop.enabled = True

        sim = Simulation(cfg)
        agent = sim.agent(0)

        # Run 2 steps - hp decreases but doesn't hit 0 yet
        for _ in range(2):
            agent.set_action(Action(name="noop"))
            sim.step()

        # Check hp is decreasing
        hp_after_2 = agent.inventory.get("hp", 0)
        assert hp_after_2 == 1, f"HP should be 1 after 2 steps, got {hp_after_2}"

        # Weapon should still be intact
        weapon_after_2 = agent.inventory.get("weapon", 0)
        assert weapon_after_2 == 5, f"Weapon should be 5 before trigger, got {weapon_after_2}"

        # Step 3 - hp should hit 0 and trigger damage
        agent.set_action(Action(name="noop"))
        sim.step()

        # HP should be 0
        hp_after_3 = agent.inventory.get("hp", 0)
        assert hp_after_3 == 0, f"HP should be 0 after trigger, got {hp_after_3}"

        # Weapon should be reduced
        weapon_after_3 = agent.inventory.get("weapon", 0)
        assert weapon_after_3 == 4, f"Weapon should be 4 after trigger, got {weapon_after_3}"

    def test_health_clear_inventory_on_damage(self):
        """Test using ClearInventoryMutation on damage."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#"],
                ["#", "@", "#"],
                ["#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty"},
        )

        cfg.game.resource_names = ["battery", "weapon", "shield", "hp"]
        cfg.game.agent.inventory.initial = {"battery": 5, "weapon": 5, "shield": 5, "hp": 0}
        cfg.game.agent.inventory.limits = {
            "gear": {"limit": 10, "resources": ["weapon", "shield"]},
            "other": {"limit": 10, "resources": ["battery", "hp"]},
        }
        cfg.game.agent.health = HealthConfig(
            health_resource="hp",
            on_damage=[ClearInventoryMutation(target="actor", limit_name="gear")],
        )
        cfg.game.actions.noop.enabled = True

        sim = Simulation(cfg)
        agent = sim.agent(0)

        # Take a step - should clear gear (weapon and shield)
        agent.set_action(Action(name="noop"))
        sim.step()

        # Gear should be cleared
        assert agent.inventory.get("weapon", 0) == 0
        assert agent.inventory.get("shield", 0) == 0
        # Battery should be unchanged
        assert agent.inventory.get("battery", 0) == 5

    def test_simulation_without_health_config(self):
        """Test that simulation runs without health config."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#"],
                ["#", "@", "#"],
                ["#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty"},
        )

        cfg.game.resource_names = ["battery", "weapon", "shield"]
        cfg.game.agent.inventory.initial = {"battery": 2, "weapon": 2, "shield": 2}
        # No health config set
        cfg.game.actions.noop.enabled = True

        sim = Simulation(cfg)
        agent = sim.agent(0)

        # Take some steps
        for _ in range(10):
            agent.set_action(Action(name="noop"))
            sim.step()

        # Inventory should be unchanged
        assert agent.inventory.get("battery", 0) == 2
        assert agent.inventory.get("weapon", 0) == 2
        assert agent.inventory.get("shield", 0) == 2
