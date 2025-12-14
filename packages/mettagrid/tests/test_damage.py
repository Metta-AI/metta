"""Test damage system functionality."""

from mettagrid.config.mettagrid_config import DamageConfig, MettaGridConfig
from mettagrid.simulator import Action, Simulation


class TestDamage:
    """Test damage configuration and behavior."""

    def test_damage_config_creation(self):
        """Test that damage config can be created and accessed."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#"],
                ["#", "@", "#"],
                ["#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty"},
        )

        cfg.game.resource_names = ["battery", "weapon", "shield", "damage"]
        cfg.game.agent.damage = DamageConfig(
            threshold={"damage": 10},
            resources={"battery": 0, "weapon": 0, "shield": 0},
        )

        # Verify the config is set correctly
        assert cfg.game.agent.damage is not None
        assert cfg.game.agent.damage.threshold == {"damage": 10}
        assert cfg.game.agent.damage.resources == {"battery": 0, "weapon": 0, "shield": 0}

    def test_damage_triggers_when_threshold_reached(self):
        """Test that damage triggers and removes items when threshold is reached."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#"],
                ["#", "@", "#"],
                ["#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty"},
        )

        cfg.game.resource_names = ["battery", "weapon", "shield", "damage"]
        # Start with damage at threshold so it triggers immediately
        cfg.game.agent.inventory.initial = {"battery": 3, "weapon": 3, "shield": 3, "damage": 10}
        cfg.game.agent.damage = DamageConfig(
            threshold={"damage": 10},
            resources={"battery": 0, "weapon": 0, "shield": 0},
        )
        cfg.game.actions.noop.enabled = True

        sim = Simulation(cfg)
        agent = sim.agent(0)

        # Check initial inventory
        initial_battery = agent.inventory.get("battery", 0)
        initial_weapon = agent.inventory.get("weapon", 0)
        initial_shield = agent.inventory.get("shield", 0)
        initial_damage = agent.inventory.get("damage", 0)

        assert initial_battery == 3
        assert initial_weapon == 3
        assert initial_shield == 3
        assert initial_damage == 10

        # Take a step - damage should trigger
        agent.set_action(Action(name="noop"))
        sim.step()

        # Damage should be reset (subtracted by threshold)
        new_damage = agent.inventory.get("damage", 0)
        assert new_damage == 0, f"Damage should be 0 after trigger, got {new_damage}"

        # One of the equipment items should be reduced by 1
        new_battery = agent.inventory.get("battery", 0)
        new_weapon = agent.inventory.get("weapon", 0)
        new_shield = agent.inventory.get("shield", 0)

        total_equipment = new_battery + new_weapon + new_shield
        assert total_equipment == 8, f"Total equipment should be 8 (one removed), got {total_equipment}"

    def test_damage_with_regen(self):
        """Test that damage works with inventory regeneration."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#"],
                ["#", "@", "#"],
                ["#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty"},
        )

        cfg.game.resource_names = ["battery", "weapon", "shield", "damage"]
        cfg.game.agent.inventory.initial = {"battery": 5, "weapon": 5, "shield": 5, "damage": 0}
        cfg.game.agent.damage = DamageConfig(
            threshold={"damage": 5},
            resources={"battery": 0, "weapon": 0, "shield": 0},
        )
        cfg.game.agent.inventory.regen_amounts = {"default": {"damage": 1}}
        cfg.game.inventory_regen_interval = 1
        cfg.game.actions.noop.enabled = True

        sim = Simulation(cfg)
        agent = sim.agent(0)

        # Run 4 steps - damage accumulates but doesn't trigger yet
        for _ in range(4):
            agent.set_action(Action(name="noop"))
            sim.step()

        # Check damage is accumulating
        damage_after_4 = agent.inventory.get("damage", 0)
        assert damage_after_4 == 4, f"Damage should be 4 after 4 steps, got {damage_after_4}"

        # Equipment should still be intact
        total_equipment = (
            agent.inventory.get("battery", 0) + agent.inventory.get("weapon", 0) + agent.inventory.get("shield", 0)
        )
        assert total_equipment == 15, f"Equipment should be 15 before trigger, got {total_equipment}"

        # Step 5 - damage should hit 5 and trigger
        agent.set_action(Action(name="noop"))
        sim.step()

        # Damage should be reset to 0 (5 - 5 = 0)
        damage_after_5 = agent.inventory.get("damage", 0)
        assert damage_after_5 == 0, f"Damage should be 0 after trigger, got {damage_after_5}"

        # One equipment item should be removed
        total_equipment = (
            agent.inventory.get("battery", 0) + agent.inventory.get("weapon", 0) + agent.inventory.get("shield", 0)
        )
        assert total_equipment == 14, f"Equipment should be 14 after trigger, got {total_equipment}"

    def test_damage_respects_minimum(self):
        """Test that damage won't remove items at or below their minimum."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#"],
                ["#", "@", "#"],
                ["#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty"},
        )

        cfg.game.resource_names = ["battery", "weapon", "shield", "damage"]
        # Start with battery=1 (at minimum), weapon=5 (above minimum)
        cfg.game.agent.inventory.initial = {"battery": 1, "weapon": 5, "shield": 0, "damage": 10}
        cfg.game.agent.damage = DamageConfig(
            threshold={"damage": 10},
            resources={"battery": 1, "weapon": 0, "shield": 0},  # battery min=1, weapon min=0
        )
        cfg.game.actions.noop.enabled = True

        sim = Simulation(cfg)
        agent = sim.agent(0)

        # Take a step - damage should trigger, but only weapon can be removed (battery at min)
        agent.set_action(Action(name="noop"))
        sim.step()

        # Battery should still be 1 (protected by minimum)
        new_battery = agent.inventory.get("battery", 0)
        assert new_battery == 1, f"Battery should remain at 1 (minimum), got {new_battery}"

        # Weapon should be reduced
        new_weapon = agent.inventory.get("weapon", 0)
        assert new_weapon == 4, f"Weapon should be 4 after damage, got {new_weapon}"

    def test_simulation_with_damage_config(self):
        """Test that simulation runs with damage config."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#"],
                ["#", "@", "#"],
                ["#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty"},
        )

        cfg.game.resource_names = ["battery", "weapon", "shield", "damage"]
        cfg.game.agent.inventory.initial = {"battery": 2, "weapon": 2, "shield": 2, "damage": 0}
        cfg.game.agent.damage = DamageConfig(
            threshold={"damage": 100},
            resources={"battery": 0, "weapon": 0, "shield": 0},
        )
        cfg.game.actions.noop.enabled = True

        sim = Simulation(cfg)

        # Check initial inventory
        agent = sim.agent(0)
        assert agent.inventory.get("battery", 0) == 2
        assert agent.inventory.get("weapon", 0) == 2
        assert agent.inventory.get("shield", 0) == 2

        # Take some steps - damage should not trigger since we never reach the threshold
        for _ in range(5):
            agent.set_action(Action(name="noop"))
            sim.step()

        # Inventory should be unchanged (no damage threshold reached)
        assert agent.inventory.get("battery", 0) == 2
        assert agent.inventory.get("weapon", 0) == 2
        assert agent.inventory.get("shield", 0) == 2

    def test_damage_not_applied_without_config(self):
        """Test that damage is not applied when damage config is None."""
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
        # No damage config set
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

    def test_weighted_removal_favors_higher_quantities(self):
        """Test that damage removal is weighted by quantity above minimum.

        Items with more excess inventory should be removed more frequently.
        """
        # Run many trials to verify statistical distribution
        removal_counts = {"battery": 0, "weapon": 0, "shield": 0}
        num_trials = 300

        for trial in range(num_trials):
            cfg = MettaGridConfig.EmptyRoom(num_agents=1, with_walls=True).with_ascii_map(
                [
                    ["#", "#", "#"],
                    ["#", "@", "#"],
                    ["#", "#", "#"],
                ],
                char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty"},
            )

            cfg.game.resource_names = ["battery", "weapon", "shield", "damage"]
            # battery=10 (weight 10), weapon=3 (weight 3), shield=1 (weight 1)
            cfg.game.agent.inventory.initial = {"battery": 10, "weapon": 3, "shield": 1, "damage": 10}
            cfg.game.agent.damage = DamageConfig(
                threshold={"damage": 10},
                resources={"battery": 0, "weapon": 0, "shield": 0},
            )
            cfg.game.actions.noop.enabled = True

            # Use different seed for each trial to get different random outcomes
            sim = Simulation(cfg, seed=trial)
            agent = sim.agent(0)

            # Trigger damage
            agent.set_action(Action(name="noop"))
            sim.step()

            # Check which item was removed
            if agent.inventory.get("battery", 0) == 9:
                removal_counts["battery"] += 1
            elif agent.inventory.get("weapon", 0) == 2:
                removal_counts["weapon"] += 1
            elif agent.inventory.get("shield", 0) == 0:
                removal_counts["shield"] += 1

        # Expected weights: battery=10, weapon=3, shield=1, total=14
        # Expected probabilities: battery=10/14≈71%, weapon=3/14≈21%, shield=1/14≈7%
        # With 300 trials, expect roughly: battery≈214, weapon≈64, shield≈21
        # Allow reasonable variance (at least battery > weapon > shield)
        assert removal_counts["battery"] > removal_counts["weapon"], (
            f"Battery (weight 10) should be removed more than weapon (weight 3): "
            f"battery={removal_counts['battery']}, weapon={removal_counts['weapon']}"
        )
        assert removal_counts["weapon"] > removal_counts["shield"], (
            f"Weapon (weight 3) should be removed more than shield (weight 1): "
            f"weapon={removal_counts['weapon']}, shield={removal_counts['shield']}"
        )
        # Battery should be removed at least 50% of the time (expected ~71%)
        assert removal_counts["battery"] >= num_trials * 0.5, (
            f"Battery should be removed at least 50% of the time, got {removal_counts['battery']}/{num_trials}"
        )
