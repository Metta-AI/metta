"""Tests for enhanced attack system with armor, weapons, and configurable loot.

The enhanced attack system adds:
- armor_resources: Target resources that reduce incoming damage (weighted)
- weapon_resources: Attacker resources that increase damage (weighted)
- loot: Configurable list of resources to steal (or steal all, or steal nothing)
- Vibe-based bonus: Vibing a resource gives +1 effective armor for that type
"""

import pytest

from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    AttackActionConfig,
    ChangeVibeActionConfig,
    GameConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    ObsConfig,
    ResourceLimitsConfig,
    WallConfig,
)
from mettagrid.simulator import Simulation
from mettagrid.test_support.actions import attack
from mettagrid.test_support.map_builders import ObjectNameMapBuilder


@pytest.fixture
def base_game_config():
    """Base game configuration for attack tests."""
    return GameConfig(
        max_steps=50,
        num_agents=2,
        obs=ObsConfig(width=5, height=5, num_tokens=100),
        resource_names=["energy", "armor", "weapon", "heart", "gold"],
        actions=ActionsConfig(
            noop=NoopActionConfig(enabled=True),
            move=MoveActionConfig(enabled=True),
            attack=AttackActionConfig(enabled=True),
            change_vibe=ChangeVibeActionConfig(enabled=True, number_of_vibes=5),
        ),
        objects={"wall": WallConfig()},
        agent=AgentConfig(
            default_resource_limit=100,
            freeze_duration=5,
            rewards=AgentRewards(),
        ),
    )


def create_sim_with_agents(game_config: GameConfig, attacker_inv: dict, target_inv: dict, seed: int = 42):
    """Create a simulation with two agents side-by-side with specified inventories."""
    # Map layout: [Attacker, Target]
    game_map = [["agent.red", "agent.blue"]]

    game_config.agents = [
        AgentConfig(
            team_id=0,
            initial_inventory=attacker_inv,
            resource_limits={"all": ResourceLimitsConfig(limit=100, resources=game_config.resource_names)},
        ),
        AgentConfig(
            team_id=1,
            initial_inventory=target_inv,
            resource_limits={"all": ResourceLimitsConfig(limit=100, resources=game_config.resource_names)},
        ),
    ]

    cfg = MettaGridConfig(game=game_config)
    cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=game_map)
    return Simulation(cfg, seed=seed)


class TestWeaponArmorSystem:
    """Tests for the weapon vs armor damage calculation."""

    def test_weapon_increases_defense_cost(self, base_game_config):
        """Test that weapon power increases the cost to defend."""
        # Configure: weapon=1 means each weapon unit adds 1 to defense cost
        base_game_config.actions.attack = AttackActionConfig(
            enabled=True,
            defense_resources={"energy": 1},
            weapon_resources={"weapon": 1},
            armor_resources={},
        )

        # Attacker has 3 weapons, target has 2 energy
        # Defense cost = base(1) + weapon_power(3) - armor_power(0) = 4
        # Target needs 4 energy to defend, only has 2 -> attack succeeds
        sim = create_sim_with_agents(
            base_game_config,
            attacker_inv={"weapon": 3},
            target_inv={"energy": 2, "gold": 5},
        )

        result = attack(sim, target_arg=0, agent_idx=0)
        assert result["success"], "Attack should succeed when target can't afford defense"
        assert result["target_frozen"], "Target should be frozen"

    def test_armor_reduces_defense_cost(self, base_game_config):
        """Test that armor power reduces defense cost (but not below base)."""
        # Configure: armor=1 means each armor unit reduces damage by 1
        base_game_config.actions.attack = AttackActionConfig(
            enabled=True,
            defense_resources={"energy": 1},
            weapon_resources={"weapon": 1},
            armor_resources={"armor": 1},
        )

        # Attacker has 3 weapons, target has 3 armor and 1 energy
        # weapon_power = 3, armor_power = 3
        # damage_bonus = max(3-3, 0) = 0
        # Defense cost = base(1) + 0 = 1
        # Target has exactly 1 energy -> defense succeeds
        sim = create_sim_with_agents(
            base_game_config,
            attacker_inv={"weapon": 3},
            target_inv={"energy": 1, "armor": 3, "gold": 5},
        )

        result = attack(sim, target_arg=0, agent_idx=0)
        assert result["success"], "Attack action should execute"
        assert not result.get("target_frozen"), "Target should NOT be frozen (defense succeeded)"

        # Verify energy was consumed
        target_energy = sim.agent(1).inventory.get("energy", 0)
        assert target_energy == 0, f"Target should have 0 energy after defending, has {target_energy}"

    def test_weapon_armor_balance(self, base_game_config):
        """Test complex weapon vs armor balance."""
        # Configure: weapon=2 (each weapon is worth 2), armor=1 (each armor is worth 1)
        base_game_config.actions.attack = AttackActionConfig(
            enabled=True,
            defense_resources={"energy": 2},
            weapon_resources={"weapon": 2},
            armor_resources={"armor": 1},
        )

        # Attacker has 2 weapons (power=4), target has 3 armor (power=3)
        # damage_bonus = max(4-3, 0) = 1
        # Defense cost = base(2) + 1 = 3
        # Target has 3 energy -> defense succeeds
        sim = create_sim_with_agents(
            base_game_config,
            attacker_inv={"weapon": 2},
            target_inv={"energy": 3, "armor": 3},
        )

        result = attack(sim, target_arg=0, agent_idx=0)
        assert result["success"], "Attack action should execute"
        assert not result.get("target_frozen"), "Target should defend successfully with exactly enough energy"

        # Now test with target having only 2 energy (not enough)
        sim2 = create_sim_with_agents(
            base_game_config,
            attacker_inv={"weapon": 2},
            target_inv={"energy": 2, "armor": 3},
        )

        result2 = attack(sim2, target_arg=0, agent_idx=0)
        assert result2["success"], "Attack should succeed"
        assert result2["target_frozen"], "Target should be frozen (not enough energy to defend)"


class TestVibeArmorBonus:
    """Tests for the vibe-based armor bonus."""

    def test_vibe_gives_armor_bonus(self, base_game_config):
        """Test that vibing a resource gives +1 effective armor for that type."""
        # Configure: armor=1 for the "armor" resource
        base_game_config.actions.attack = AttackActionConfig(
            enabled=True,
            defense_resources={"energy": 1},
            weapon_resources={"weapon": 1},
            armor_resources={"armor": 1},
        )

        # First, test without vibe bonus
        # Attacker has 2 weapons (power=2), target has 1 armor (power=1)
        # damage_bonus = max(2-1, 0) = 1
        # Defense cost = base(1) + 1 = 2
        # Target has only 1 energy -> attack succeeds
        sim = create_sim_with_agents(
            base_game_config,
            attacker_inv={"weapon": 2},
            target_inv={"energy": 1, "armor": 1},
        )

        result = attack(sim, target_arg=0, agent_idx=0)
        assert result["target_frozen"], "Without vibe bonus, target should be frozen"

        # Now test WITH vibe bonus
        # If target is vibing "armor", they get +1 effective armor
        # armor_power = 1 (held) + 1 (vibe bonus) = 2
        # damage_bonus = max(2-2, 0) = 0
        # Defense cost = base(1) + 0 = 1
        # Target has 1 energy -> defense succeeds
        base_game_config.agents = [
            AgentConfig(
                team_id=0,
                initial_inventory={"weapon": 2},
                resource_limits={"all": ResourceLimitsConfig(limit=100, resources=base_game_config.resource_names)},
            ),
            AgentConfig(
                team_id=1,
                initial_inventory={"energy": 1, "armor": 1},
                initial_vibe=2,  # Set vibe to "armor" (vibe index matches resource index)
                resource_limits={"all": ResourceLimitsConfig(limit=100, resources=base_game_config.resource_names)},
            ),
        ]

        cfg = MettaGridConfig(game=base_game_config)
        cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=[["agent.red", "agent.blue"]])
        sim2 = Simulation(cfg, seed=42)

        result2 = attack(sim2, target_arg=0, agent_idx=0)
        assert result2["success"], "Attack should execute"
        assert not result2.get("target_frozen"), "With vibe bonus, target should defend successfully"


class TestConfigurableLoot:
    """Tests for configurable loot list."""

    def test_loot_none_steals_all(self, base_game_config):
        """Test that loot=None (default) steals all resources."""
        base_game_config.actions.attack = AttackActionConfig(
            enabled=True,
            loot=None,  # Steal all (default)
        )

        sim = create_sim_with_agents(
            base_game_config,
            attacker_inv={},
            target_inv={"energy": 5, "armor": 3, "gold": 7},
        )

        attack(sim, target_arg=0, agent_idx=0)

        # Attacker should have stolen everything
        attacker = sim.agent(0).inventory
        assert attacker.get("energy", 0) == 5, "Should steal energy"
        assert attacker.get("armor", 0) == 3, "Should steal armor"
        assert attacker.get("gold", 0) == 7, "Should steal gold"

    def test_loot_empty_steals_nothing(self, base_game_config):
        """Test that loot=[] steals nothing."""
        base_game_config.actions.attack = AttackActionConfig(
            enabled=True,
            loot=[],  # Steal nothing
        )

        sim = create_sim_with_agents(
            base_game_config,
            attacker_inv={},
            target_inv={"energy": 5, "gold": 7},
        )

        attack(sim, target_arg=0, agent_idx=0)

        # Attacker should have stolen nothing
        attacker = sim.agent(0).inventory
        target = sim.agent(1).inventory
        assert attacker.get("energy", 0) == 0, "Should NOT steal energy"
        assert attacker.get("gold", 0) == 0, "Should NOT steal gold"
        # Target should keep everything
        assert target.get("energy", 0) == 5, "Target should keep energy"
        assert target.get("gold", 0) == 7, "Target should keep gold"

    def test_loot_specific_resources(self, base_game_config):
        """Test that loot=[specific] only steals specified resources."""
        base_game_config.actions.attack = AttackActionConfig(
            enabled=True,
            loot=["gold", "energy"],  # Only steal gold and energy
        )

        sim = create_sim_with_agents(
            base_game_config,
            attacker_inv={},
            target_inv={"energy": 5, "armor": 3, "gold": 7},
        )

        attack(sim, target_arg=0, agent_idx=0)

        # Attacker should have stolen only gold and energy
        attacker = sim.agent(0).inventory
        target = sim.agent(1).inventory
        assert attacker.get("gold", 0) == 7, "Should steal gold"
        assert attacker.get("energy", 0) == 5, "Should steal energy"
        assert attacker.get("armor", 0) == 0, "Should NOT steal armor"
        # Target should keep armor
        assert target.get("armor", 0) == 3, "Target should keep armor"

    def test_loot_order_matters_with_capacity(self, base_game_config):
        """Test that loot order determines what gets stolen when capacity is limited."""
        base_game_config.actions.attack = AttackActionConfig(
            enabled=True,
            loot=["energy", "gold"],  # Try energy first, then gold
        )

        # Configure attacker with limited capacity for these resources
        base_game_config.agents = [
            AgentConfig(
                team_id=0,
                initial_inventory={},
                resource_limits={
                    "limited": ResourceLimitsConfig(limit=5, resources=["energy", "gold"]),
                    "other": ResourceLimitsConfig(limit=100, resources=["armor", "weapon", "heart"]),
                },
            ),
            AgentConfig(
                team_id=1,
                initial_inventory={"energy": 10, "gold": 10},
                resource_limits={"all": ResourceLimitsConfig(limit=100, resources=base_game_config.resource_names)},
            ),
        ]

        cfg = MettaGridConfig(game=base_game_config)
        cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=[["agent.red", "agent.blue"]])
        sim = Simulation(cfg, seed=42)

        attack(sim, target_arg=0, agent_idx=0)

        # Attacker should have taken energy first (limited to 5 total)
        attacker = sim.agent(0).inventory
        assert attacker.get("energy", 0) == 5, "Should steal all 5 energy (fills capacity)"
        assert attacker.get("gold", 0) == 0, "Should NOT steal gold (no capacity left)"


class TestConfigValidation:
    """Tests for attack configuration validation."""

    def test_loot_resource_validation(self, base_game_config):
        """Test that invalid loot resource names raise errors."""
        base_game_config.actions.attack = AttackActionConfig(
            enabled=True,
            loot=["energy", "nonexistent_resource"],
        )

        cfg = MettaGridConfig(game=base_game_config)
        cfg.game.map_builder = ObjectNameMapBuilder.Config(map_data=[["agent.red", "agent.blue"]])

        with pytest.raises(ValueError) as exc_info:
            Simulation(cfg, seed=42)

        assert "nonexistent_resource" in str(exc_info.value)
        assert "not found in resource_names" in str(exc_info.value)

    def test_config_with_all_features(self, base_game_config):
        """Test that all enhanced attack features work together."""
        base_game_config.actions.attack = AttackActionConfig(
            enabled=True,
            defense_resources={"energy": 1},
            armor_resources={"armor": 1},
            weapon_resources={"weapon": 2},
            loot=["gold"],  # Only steal gold
        )

        # Attacker has 2 weapons (power=4), target has 3 armor (power=3), 2 energy
        # damage_bonus = max(4-3, 0) = 1
        # Defense cost = base(1) + 1 = 2
        # Target has 2 energy -> defense succeeds, but loses 2 energy
        sim = create_sim_with_agents(
            base_game_config,
            attacker_inv={"weapon": 2},
            target_inv={"energy": 2, "armor": 3, "gold": 10},
        )

        result = attack(sim, target_arg=0, agent_idx=0)
        assert result["success"], "Attack action should execute"
        assert not result.get("target_frozen"), "Target should defend successfully"

        # Verify defense cost was consumed
        target = sim.agent(1).inventory
        assert target.get("energy", 0) == 0, "Target should have consumed 2 energy defending"
        assert target.get("gold", 0) == 10, "Target should keep gold (defense succeeded)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
