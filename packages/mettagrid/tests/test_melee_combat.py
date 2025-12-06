"""Test melee combat functionality."""

import pytest

from mettagrid.config.mettagrid_c_config import convert_to_cpp_game_config
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.simulator import Simulation


class TestMeleeCombat:
    def test_successful_melee_attack(self):
        cfg = MettaGridConfig.EmptyRoom(num_agents=2, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#", "#"],
                ["#", "@", "@", "#"],
                ["#", "#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent"},
        )

        cfg.game.resource_names = ["laser", "armor", "ore", "heart"]
        cfg.game.agent.initial_inventory = {"laser": 5, "ore": 10}
        cfg.game.agent.freeze_duration = 10

        cfg.game.melee_combat.enabled = True

        cfg.game.actions.move.enabled = True
        cfg.game.actions.noop.enabled = True
        cfg.game.actions.change_vibe.enabled = True
        cfg.game.actions.change_vibe.number_of_vibes = 100

        sim = Simulation(cfg)

        laser_idx = sim.resource_names.index("laser")
        ore_idx = sim.resource_names.index("ore")

        sim.agent(0).set_action("change_vibe_swords")
        sim.agent(1).set_action("noop")
        sim.step()

        grid_objects = sim.grid_objects()
        agents = sorted([obj for obj in grid_objects.values() if "agent_id" in obj], key=lambda x: x["agent_id"])
        agent0 = agents[0]
        agent1_before = agents[1]

        assert agent0["inventory"][laser_idx] == 5, "Agent 0 should have 5 laser"
        assert agent1_before["inventory"][ore_idx] == 10, "Agent 1 should have 10 ore"
        assert agent1_before["freeze_remaining"] == 0, "Agent 1 should not be frozen"

        sim.agent(0).set_action("move_east")
        sim.agent(1).set_action("noop")
        sim.step()

        grid_objects_after = sim.grid_objects()
        agents_after = sorted(
            [obj for obj in grid_objects_after.values() if "agent_id" in obj], key=lambda x: x["agent_id"]
        )
        agent0_after = agents_after[0]
        agent1_after = agents_after[1]

        # Laser consumed (5 -> 4) then stolen from target (4 + 5 = 9)
        assert agent0_after["inventory"][laser_idx] == 9, (
            f"Agent 0 should have 9 laser (5 - 1 consumed + 5 stolen). Has {agent0_after['inventory'][laser_idx]}"
        )

        assert agent1_after["freeze_remaining"] > 0, "Agent 1 should be frozen after attack"

        assert agent0_after["inventory"][ore_idx] > 10, "Agent 0 should have gained ore"
        agent1_ore = agent1_after["inventory"].get(ore_idx, 0)
        assert agent1_ore < 10, f"Agent 1 should have lost ore. Has {agent1_ore}"

    def test_blocked_attack_with_shield(self):
        cfg = MettaGridConfig.EmptyRoom(num_agents=2, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#", "#"],
                ["#", "@", "@", "#"],
                ["#", "#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent"},
        )

        cfg.game.resource_names = ["laser", "armor", "ore", "heart"]
        cfg.game.agent.initial_inventory = {"laser": 5, "armor": 3, "ore": 10}
        cfg.game.agent.freeze_duration = 10

        cfg.game.melee_combat.enabled = True

        cfg.game.actions.move.enabled = True
        cfg.game.actions.noop.enabled = True
        cfg.game.actions.change_vibe.enabled = True
        cfg.game.actions.change_vibe.number_of_vibes = 100

        sim = Simulation(cfg)

        laser_idx = sim.resource_names.index("laser")
        armor_idx = sim.resource_names.index("armor")
        ore_idx = sim.resource_names.index("ore")

        sim.agent(0).set_action("change_vibe_swords")
        sim.agent(1).set_action("change_vibe_shield")
        sim.step()

        grid_objects = sim.grid_objects()
        agents = sorted([obj for obj in grid_objects.values() if "agent_id" in obj], key=lambda x: x["agent_id"])
        agent1_before = agents[1]
        assert agent1_before["inventory"][ore_idx] == 10

        sim.agent(0).set_action("move_east")
        sim.agent(1).set_action("noop")
        sim.step()

        grid_objects_after = sim.grid_objects()
        agents_after = sorted(
            [obj for obj in grid_objects_after.values() if "agent_id" in obj], key=lambda x: x["agent_id"]
        )
        agent0_after = agents_after[0]
        agent1_after = agents_after[1]

        assert agent0_after["inventory"][laser_idx] == 5, "Laser not consumed on blocked attack"
        assert agent1_after["freeze_remaining"] == 0, "Agent 1 should not be frozen (blocked by shield)"
        assert agent1_after["inventory"][armor_idx] == 3, "Armor should not be consumed on defense"
        assert agent1_after["inventory"][ore_idx] == 10, "Agent 1 should keep all ore (attack blocked)"

    def test_defense_fails_wrong_vibe_with_armor(self):
        """Defense requires BOTH correct vibe AND armor. Wrong vibe with armor = defense fails."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=2, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#", "#"],
                ["#", "@", "@", "#"],
                ["#", "#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent"},
        )

        cfg.game.resource_names = ["laser", "armor", "ore", "heart"]
        cfg.game.agent.initial_inventory = {"laser": 5, "armor": 3, "ore": 10}
        cfg.game.agent.freeze_duration = 10

        cfg.game.melee_combat.enabled = True

        cfg.game.actions.move.enabled = True
        cfg.game.actions.noop.enabled = True
        cfg.game.actions.change_vibe.enabled = True
        cfg.game.actions.change_vibe.number_of_vibes = 100

        sim = Simulation(cfg)

        laser_idx = sim.resource_names.index("laser")
        armor_idx = sim.resource_names.index("armor")
        ore_idx = sim.resource_names.index("ore")

        # Attacker sets swords vibe, defender uses WRONG vibe (swords instead of shield)
        sim.agent(0).set_action("change_vibe_swords")
        sim.agent(1).set_action("change_vibe_swords")  # Wrong vibe - should be shield
        sim.step()

        grid_objects = sim.grid_objects()
        agents = sorted([obj for obj in grid_objects.values() if "agent_id" in obj], key=lambda x: x["agent_id"])
        agent1_before = agents[1]
        assert agent1_before["inventory"].get(armor_idx, 0) == 3, "Agent 1 should have armor"

        sim.agent(0).set_action("move_east")
        sim.agent(1).set_action("noop")
        sim.step()

        grid_objects_after = sim.grid_objects()
        agents_after = sorted(
            [obj for obj in grid_objects_after.values() if "agent_id" in obj], key=lambda x: x["agent_id"]
        )
        agent0_after = agents_after[0]
        agent1_after = agents_after[1]

        # Defense fails: wrong vibe, even though target has armor
        assert agent1_after["freeze_remaining"] > 0, "Agent 1 should be frozen (wrong vibe, armor doesn't help)"
        assert agent0_after["inventory"].get(laser_idx, 0) == 9, "Attacker: 5 - 1 consumed + 5 stolen = 9"
        # Armor was stolen by successful attack (defense was never triggered due to wrong vibe)
        assert agent1_after["inventory"].get(armor_idx, 0) == 0, "Armor was stolen by attacker"
        assert agent0_after["inventory"].get(armor_idx, 0) == 6, "Attacker: 3 own + 3 stolen = 6 armor"
        assert agent1_after["inventory"].get(ore_idx, 0) < 10, "Agent 1 should have lost ore (attack succeeded)"

    def test_defense_fails_right_vibe_without_armor(self):
        """Defense requires BOTH correct vibe AND armor. Right vibe without armor = defense fails."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=2, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#", "#"],
                ["#", "@", "@", "#"],
                ["#", "#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent"},
        )

        cfg.game.resource_names = ["laser", "armor", "ore", "heart"]
        # No armor in initial inventory
        cfg.game.agent.initial_inventory = {"laser": 5, "ore": 10}
        cfg.game.agent.freeze_duration = 10

        cfg.game.melee_combat.enabled = True

        cfg.game.actions.move.enabled = True
        cfg.game.actions.noop.enabled = True
        cfg.game.actions.change_vibe.enabled = True
        cfg.game.actions.change_vibe.number_of_vibes = 100

        sim = Simulation(cfg)

        laser_idx = sim.resource_names.index("laser")
        armor_idx = sim.resource_names.index("armor")
        ore_idx = sim.resource_names.index("ore")

        # Attacker sets swords, defender sets correct shield vibe but has no armor
        sim.agent(0).set_action("change_vibe_swords")
        sim.agent(1).set_action("change_vibe_shield")  # Correct vibe, but no armor
        sim.step()

        grid_objects = sim.grid_objects()
        agents = sorted([obj for obj in grid_objects.values() if "agent_id" in obj], key=lambda x: x["agent_id"])
        agent1_before = agents[1]
        assert agent1_before["inventory"].get(armor_idx, 0) == 0, "Agent 1 should have no armor"

        sim.agent(0).set_action("move_east")
        sim.agent(1).set_action("noop")
        sim.step()

        grid_objects_after = sim.grid_objects()
        agents_after = sorted(
            [obj for obj in grid_objects_after.values() if "agent_id" in obj], key=lambda x: x["agent_id"]
        )
        agent0_after = agents_after[0]
        agent1_after = agents_after[1]

        # Defense fails: correct vibe, but no armor
        assert agent1_after["freeze_remaining"] > 0, "Agent 1 should be frozen (no armor, vibe doesn't help alone)"
        assert agent0_after["inventory"].get(laser_idx, 0) == 9, "Attacker: 5 - 1 consumed + 5 stolen = 9"
        assert agent1_after["inventory"].get(ore_idx, 0) < 10, "Agent 1 should have lost ore (attack succeeded)"

    def test_attack_without_laser_fails(self):
        cfg = MettaGridConfig.EmptyRoom(num_agents=2, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#", "#"],
                ["#", "@", "@", "#"],
                ["#", "#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent"},
        )

        cfg.game.resource_names = ["laser", "armor", "ore", "heart"]
        cfg.game.agent.initial_inventory = {"ore": 10}
        cfg.game.agent.freeze_duration = 10

        cfg.game.melee_combat.enabled = True

        cfg.game.actions.move.enabled = True
        cfg.game.actions.noop.enabled = True
        cfg.game.actions.change_vibe.enabled = True
        cfg.game.actions.change_vibe.number_of_vibes = 100

        sim = Simulation(cfg)

        ore_idx = sim.resource_names.index("ore")

        sim.agent(0).set_action("change_vibe_swords")
        sim.agent(1).set_action("noop")
        sim.step()

        sim.agent(0).set_action("move_east")
        sim.agent(1).set_action("noop")
        sim.step()

        grid_objects_after = sim.grid_objects()
        agents_after = sorted(
            [obj for obj in grid_objects_after.values() if "agent_id" in obj], key=lambda x: x["agent_id"]
        )
        agent1_after = agents_after[1]

        assert agent1_after["freeze_remaining"] == 0, "Agent 1 should not be frozen (attacker had no laser)"
        assert agent1_after["inventory"][ore_idx] == 10, "Agent 1 should keep ore (attack failed)"

    def test_friendly_fire_allowed(self):
        cfg = MettaGridConfig.EmptyRoom(num_agents=2, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#", "#"],
                ["#", "@", "@", "#"],
                ["#", "#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent"},
        )

        cfg.game.resource_names = ["laser", "armor", "ore", "heart"]
        cfg.game.agent.initial_inventory = {"laser": 5, "ore": 10}
        cfg.game.agent.freeze_duration = 10
        cfg.game.agent.team_id = 0

        cfg.game.melee_combat.enabled = True

        cfg.game.actions.move.enabled = True
        cfg.game.actions.noop.enabled = True
        cfg.game.actions.change_vibe.enabled = True
        cfg.game.actions.change_vibe.number_of_vibes = 100

        sim = Simulation(cfg)

        sim.agent(0).set_action("change_vibe_swords")
        sim.agent(1).set_action("noop")
        sim.step()

        sim.agent(0).set_action("move_east")
        sim.agent(1).set_action("noop")
        sim.step()

        grid_objects_after = sim.grid_objects()
        agents_after = sorted(
            [obj for obj in grid_objects_after.values() if "agent_id" in obj], key=lambda x: x["agent_id"]
        )
        agent1_after = agents_after[1]

        assert agent1_after["freeze_remaining"] > 0, "Agent 1 should be frozen (friendly fire allowed)"

    def test_melee_combat_disabled_no_attack(self):
        cfg = MettaGridConfig.EmptyRoom(num_agents=2, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#", "#"],
                ["#", "@", "@", "#"],
                ["#", "#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent"},
        )

        cfg.game.resource_names = ["laser", "armor", "ore", "heart"]
        cfg.game.agent.initial_inventory = {"laser": 5, "ore": 10}
        cfg.game.agent.freeze_duration = 10

        assert cfg.game.melee_combat.enabled is False

        cfg.game.actions.move.enabled = True
        cfg.game.actions.noop.enabled = True
        cfg.game.actions.change_vibe.enabled = True
        cfg.game.actions.change_vibe.number_of_vibes = 100

        sim = Simulation(cfg)

        laser_idx = sim.resource_names.index("laser")
        ore_idx = sim.resource_names.index("ore")

        sim.agent(0).set_action("change_vibe_swords")
        sim.agent(1).set_action("noop")
        sim.step()

        sim.agent(0).set_action("move_east")
        sim.agent(1).set_action("noop")
        sim.step()

        grid_objects_after = sim.grid_objects()
        agents_after = sorted(
            [obj for obj in grid_objects_after.values() if "agent_id" in obj], key=lambda x: x["agent_id"]
        )
        agent0_after = agents_after[0]
        agent1_after = agents_after[1]

        assert agent1_after["freeze_remaining"] == 0, "Agent 1 should not be frozen (combat disabled)"
        assert agent0_after["inventory"][laser_idx] == 5, "Laser should not be consumed (combat disabled)"
        assert agent1_after["inventory"][ore_idx] == 10, "Ore should not be stolen (combat disabled)"

    def test_attack_consumes_item_false(self):
        """When attack_consumes_item=False, laser should not be consumed on successful attack."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=2, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#", "#"],
                ["#", "@", "@", "#"],
                ["#", "#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent"},
        )

        cfg.game.resource_names = ["laser", "armor", "ore", "heart"]
        cfg.game.agent.initial_inventory = {"laser": 5, "ore": 10}
        cfg.game.agent.freeze_duration = 10

        cfg.game.melee_combat.enabled = True
        cfg.game.melee_combat.attack_consumes_item = False

        cfg.game.actions.move.enabled = True
        cfg.game.actions.noop.enabled = True
        cfg.game.actions.change_vibe.enabled = True
        cfg.game.actions.change_vibe.number_of_vibes = 100

        sim = Simulation(cfg)

        laser_idx = sim.resource_names.index("laser")
        ore_idx = sim.resource_names.index("ore")

        sim.agent(0).set_action("change_vibe_swords")
        sim.agent(1).set_action("noop")
        sim.step()

        sim.agent(0).set_action("move_east")
        sim.agent(1).set_action("noop")
        sim.step()

        grid_objects_after = sim.grid_objects()
        agents_after = sorted(
            [obj for obj in grid_objects_after.values() if "agent_id" in obj], key=lambda x: x["agent_id"]
        )
        agent0_after = agents_after[0]
        agent1_after = agents_after[1]

        # Attack succeeds but laser NOT consumed, plus 5 stolen from target
        assert agent0_after["inventory"][laser_idx] == 10, (
            f"Agent 0 should have 10 laser (5 own + 5 stolen, none consumed). "
            f"Has {agent0_after['inventory'][laser_idx]}"
        )
        assert agent1_after["freeze_remaining"] > 0, "Agent 1 should be frozen"
        assert agent0_after["inventory"][ore_idx] == 20, "Agent 0 should have 20 ore (10 + 10 stolen)"
        assert agent1_after["inventory"].get(ore_idx, 0) == 0, "Agent 1 should have no ore"

    def test_defense_consumes_item_true(self):
        """When defense_consumes_item=True, armor should be consumed on successful defense."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=2, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#", "#"],
                ["#", "@", "@", "#"],
                ["#", "#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent"},
        )

        cfg.game.resource_names = ["laser", "armor", "ore", "heart"]
        cfg.game.agent.initial_inventory = {"laser": 5, "armor": 3, "ore": 10}
        cfg.game.agent.freeze_duration = 10

        cfg.game.melee_combat.enabled = True
        cfg.game.melee_combat.defense_consumes_item = True

        cfg.game.actions.move.enabled = True
        cfg.game.actions.noop.enabled = True
        cfg.game.actions.change_vibe.enabled = True
        cfg.game.actions.change_vibe.number_of_vibes = 100

        sim = Simulation(cfg)

        laser_idx = sim.resource_names.index("laser")
        armor_idx = sim.resource_names.index("armor")
        ore_idx = sim.resource_names.index("ore")

        sim.agent(0).set_action("change_vibe_swords")
        sim.agent(1).set_action("change_vibe_shield")
        sim.step()

        grid_objects_before = sim.grid_objects()
        agents_before = sorted(
            [obj for obj in grid_objects_before.values() if "agent_id" in obj], key=lambda x: x["agent_id"]
        )
        assert agents_before[1]["inventory"][armor_idx] == 3, "Agent 1 should have 3 armor before attack"

        sim.agent(0).set_action("move_east")
        sim.agent(1).set_action("noop")
        sim.step()

        grid_objects_after = sim.grid_objects()
        agents_after = sorted(
            [obj for obj in grid_objects_after.values() if "agent_id" in obj], key=lambda x: x["agent_id"]
        )
        agent0_after = agents_after[0]
        agent1_after = agents_after[1]

        assert agent1_after["freeze_remaining"] == 0, "Agent 1 should not be frozen (defense succeeded)"
        assert agent0_after["inventory"][laser_idx] == 5, "Attacker laser not consumed (attack was blocked)"
        assert agent1_after["inventory"][armor_idx] == 2, "Defender armor consumed: 3 -> 2"
        assert agent1_after["inventory"][ore_idx] == 10, "Agent 1 should keep all ore"

    def test_soul_bound_resources_not_stolen(self):
        """Soul-bound resources should not be stolen during melee attack."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=2, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#", "#"],
                ["#", "@", "@", "#"],
                ["#", "#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent"},
        )

        cfg.game.resource_names = ["laser", "armor", "ore", "heart"]
        cfg.game.agent.initial_inventory = {"laser": 5, "ore": 10, "heart": 3}
        cfg.game.agent.freeze_duration = 10
        cfg.game.agent.soul_bound_resources = ["heart"]  # Heart cannot be stolen

        cfg.game.melee_combat.enabled = True

        cfg.game.actions.move.enabled = True
        cfg.game.actions.noop.enabled = True
        cfg.game.actions.change_vibe.enabled = True
        cfg.game.actions.change_vibe.number_of_vibes = 100

        sim = Simulation(cfg)

        ore_idx = sim.resource_names.index("ore")
        heart_idx = sim.resource_names.index("heart")

        sim.agent(0).set_action("change_vibe_swords")
        sim.agent(1).set_action("noop")
        sim.step()

        sim.agent(0).set_action("move_east")
        sim.agent(1).set_action("noop")
        sim.step()

        grid_objects_after = sim.grid_objects()
        agents_after = sorted(
            [obj for obj in grid_objects_after.values() if "agent_id" in obj], key=lambda x: x["agent_id"]
        )
        agent0_after = agents_after[0]
        agent1_after = agents_after[1]

        assert agent1_after["freeze_remaining"] > 0, "Agent 1 should be frozen"

        # Ore should be stolen
        assert agent0_after["inventory"][ore_idx] == 20, "Agent 0 should have 20 ore (10 + 10 stolen)"
        assert agent1_after["inventory"].get(ore_idx, 0) == 0, "Agent 1 should have no ore"

        # Heart should NOT be stolen (soul-bound)
        assert agent1_after["inventory"][heart_idx] == 3, "Agent 1 should keep all 3 hearts (soul-bound)"
        assert agent0_after["inventory"][heart_idx] == 3, "Agent 0 should have only own 3 hearts"

    def test_inventory_limits_during_steal(self):
        """Stolen resources should respect attacker's inventory limits."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=2, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#", "#"],
                ["#", "@", "@", "#"],
                ["#", "#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent"},
        )

        cfg.game.resource_names = ["laser", "armor", "ore", "heart"]
        cfg.game.agent.initial_inventory = {"laser": 5, "ore": 8}  # Start with 8 ore
        cfg.game.agent.freeze_duration = 10
        cfg.game.agent.default_resource_limit = 10  # Limit of 10 for all resources

        cfg.game.melee_combat.enabled = True

        cfg.game.actions.move.enabled = True
        cfg.game.actions.noop.enabled = True
        cfg.game.actions.change_vibe.enabled = True
        cfg.game.actions.change_vibe.number_of_vibes = 100

        sim = Simulation(cfg)

        laser_idx = sim.resource_names.index("laser")
        ore_idx = sim.resource_names.index("ore")

        sim.agent(0).set_action("change_vibe_swords")
        sim.agent(1).set_action("noop")
        sim.step()

        grid_objects_before = sim.grid_objects()
        agents_before = sorted(
            [obj for obj in grid_objects_before.values() if "agent_id" in obj], key=lambda x: x["agent_id"]
        )
        assert agents_before[0]["inventory"][ore_idx] == 8, "Agent 0 should start with 8 ore"
        assert agents_before[1]["inventory"][ore_idx] == 8, "Agent 1 should start with 8 ore"

        sim.agent(0).set_action("move_east")
        sim.agent(1).set_action("noop")
        sim.step()

        grid_objects_after = sim.grid_objects()
        agents_after = sorted(
            [obj for obj in grid_objects_after.values() if "agent_id" in obj], key=lambda x: x["agent_id"]
        )
        agent0_after = agents_after[0]
        agent1_after = agents_after[1]

        assert agent1_after["freeze_remaining"] > 0, "Agent 1 should be frozen"

        # Agent 0 has 8 ore, limit is 10, so can only steal 2 more
        assert agent0_after["inventory"][ore_idx] == 10, (
            f"Agent 0 should have 10 ore (8 + 2 stolen, limited). Has {agent0_after['inventory'][ore_idx]}"
        )
        # Agent 1 should only lose 2 ore, keeping 6
        assert agent1_after["inventory"][ore_idx] == 6, (
            f"Agent 1 should have 6 ore (8 - 2 stolen). Has {agent1_after['inventory'].get(ore_idx, 0)}"
        )

        # Laser: 5 in attacker, limit 10, consumes 1, then can steal up to 6, target has 5
        # So: 5 - 1 + 5 = 9 lasers (all target lasers stolen)
        assert agent0_after["inventory"][laser_idx] == 9, (
            f"Agent 0 should have 9 laser (5 - 1 + 5). Has {agent0_after['inventory'][laser_idx]}"
        )


class TestMeleeCombatConfigValidation:
    def test_invalid_attack_vibe_raises_error(self):
        cfg = MettaGridConfig.EmptyRoom(num_agents=2, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#", "#"],
                ["#", "@", "@", "#"],
                ["#", "#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent"},
        )

        cfg.game.resource_names = ["laser", "armor", "ore", "heart"]
        cfg.game.melee_combat.enabled = True
        cfg.game.melee_combat.attack_vibe = "nonexistent_vibe"

        cfg.game.actions.change_vibe.enabled = True
        cfg.game.actions.change_vibe.number_of_vibes = 100

        with pytest.raises(ValueError) as exc_info:
            convert_to_cpp_game_config(cfg.game)

        assert "attack_vibe" in str(exc_info.value)
        assert "nonexistent_vibe" in str(exc_info.value)

    def test_invalid_defense_vibe_raises_error(self):
        cfg = MettaGridConfig.EmptyRoom(num_agents=2, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#", "#"],
                ["#", "@", "@", "#"],
                ["#", "#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent"},
        )

        cfg.game.resource_names = ["laser", "armor", "ore", "heart"]
        cfg.game.melee_combat.enabled = True
        cfg.game.melee_combat.defense_vibe = "nonexistent_vibe"

        cfg.game.actions.change_vibe.enabled = True
        cfg.game.actions.change_vibe.number_of_vibes = 100

        with pytest.raises(ValueError) as exc_info:
            convert_to_cpp_game_config(cfg.game)

        assert "defense_vibe" in str(exc_info.value)
        assert "nonexistent_vibe" in str(exc_info.value)

    def test_invalid_attack_item_raises_error(self):
        cfg = MettaGridConfig.EmptyRoom(num_agents=2, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#", "#"],
                ["#", "@", "@", "#"],
                ["#", "#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent"},
        )

        cfg.game.resource_names = ["armor", "ore", "heart"]
        cfg.game.melee_combat.enabled = True
        cfg.game.melee_combat.attack_item = "laser"

        cfg.game.actions.change_vibe.enabled = True
        cfg.game.actions.change_vibe.number_of_vibes = 100

        with pytest.raises(ValueError) as exc_info:
            convert_to_cpp_game_config(cfg.game)

        assert "attack_item" in str(exc_info.value)
        assert "laser" in str(exc_info.value)

    def test_invalid_defense_item_raises_error(self):
        cfg = MettaGridConfig.EmptyRoom(num_agents=2, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#", "#"],
                ["#", "@", "@", "#"],
                ["#", "#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent"},
        )

        cfg.game.resource_names = ["laser", "ore", "heart"]
        cfg.game.melee_combat.enabled = True
        cfg.game.melee_combat.defense_item = "armor"

        cfg.game.actions.change_vibe.enabled = True
        cfg.game.actions.change_vibe.number_of_vibes = 100

        with pytest.raises(ValueError) as exc_info:
            convert_to_cpp_game_config(cfg.game)

        assert "defense_item" in str(exc_info.value)
        assert "armor" in str(exc_info.value)

    def test_zero_freeze_duration_raises_error(self):
        cfg = MettaGridConfig.EmptyRoom(num_agents=2, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#", "#"],
                ["#", "@", "@", "#"],
                ["#", "#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent"},
        )

        cfg.game.resource_names = ["laser", "armor", "ore", "heart"]
        cfg.game.agent.freeze_duration = 0
        cfg.game.melee_combat.enabled = True

        cfg.game.actions.change_vibe.enabled = True
        cfg.game.actions.change_vibe.number_of_vibes = 100

        with pytest.raises(ValueError) as exc_info:
            convert_to_cpp_game_config(cfg.game)

        assert "freeze_duration" in str(exc_info.value)
        assert "0" in str(exc_info.value)

    def test_disabled_combat_skips_validation(self):
        cfg = MettaGridConfig.EmptyRoom(num_agents=2, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#", "#"],
                ["#", "@", "@", "#"],
                ["#", "#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent"},
        )

        cfg.game.resource_names = ["ore", "heart"]
        cfg.game.melee_combat.enabled = False
        cfg.game.melee_combat.attack_item = "laser"
        cfg.game.melee_combat.defense_item = "armor"

        cfg.game.actions.change_vibe.enabled = True
        cfg.game.actions.change_vibe.number_of_vibes = 100

        cpp_config = convert_to_cpp_game_config(cfg.game)
        assert cpp_config.melee_combat.enabled is False

    def test_zero_freeze_duration_allowed_when_combat_disabled(self):
        """freeze_duration=0 is valid when melee combat is disabled."""
        cfg = MettaGridConfig.EmptyRoom(num_agents=2, with_walls=True).with_ascii_map(
            [
                ["#", "#", "#", "#"],
                ["#", "@", "@", "#"],
                ["#", "#", "#", "#"],
            ],
            char_to_map_name={"#": "wall", "@": "agent.agent"},
        )

        cfg.game.resource_names = ["laser", "armor", "ore", "heart"]
        cfg.game.agent.freeze_duration = 0
        cfg.game.melee_combat.enabled = False

        cfg.game.actions.change_vibe.enabled = True
        cfg.game.actions.change_vibe.number_of_vibes = 100

        cpp_config = convert_to_cpp_game_config(cfg.game)
        assert cpp_config.melee_combat.enabled is False
