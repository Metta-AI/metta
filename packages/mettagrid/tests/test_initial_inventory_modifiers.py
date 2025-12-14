"""Test that initial inventory correctly handles modifier dependencies.

When resource limits depend on modifier items (e.g., energy limit depends on battery count),
the modifier items must have high enough limits to be added first.
"""

from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AgentConfig,
    GameConfig,
    InventoryConfig,
    MettaGridConfig,
    NoopActionConfig,
    ObsConfig,
    ResourceLimitsConfig,
)
from mettagrid.map_builder.random_map import RandomMapBuilder
from mettagrid.simulator import Simulation


def _get_agent_inventory(sim: Simulation, agent_id: int) -> dict[int, int]:
    """Get an agent's inventory from grid_objects."""
    for obj in sim.grid_objects().values():
        if obj.get("agent_id") == agent_id:
            return obj.get("inventory", {})
    raise AssertionError(f"Agent {agent_id} not found")


def test_initial_inventory_with_modifier_dependencies():
    """Test that modifier items are correctly applied before dependent items.

    This tests the scenario where:
    - 'tech' modifies the limit for 'weapon', 'shield', 'battery'
    - 'battery' modifies the limit for 'energy'

    Items with no limit dependencies are added first, then dependent items.
    """
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            obs=ObsConfig(width=3, height=3, num_tokens=32),
            actions=ActionsConfig(noop=NoopActionConfig()),
            resource_names=["energy", "carbon", "weapon", "shield", "battery", "tech"],
            agent=AgentConfig(
                inventory=InventoryConfig(
                    initial={
                        "tech": 10,  # Modifier for weapon/shield/battery limit
                        "battery": 4,  # Modifier for energy limit AND depends on tech
                        "energy": 100,  # Depends on battery
                        "weapon": 2,  # Depends on tech
                        "carbon": 50,  # No dependencies
                    },
                    limits={
                        # Energy limit = 0 + battery * 25 = 0 + 4 * 25 = 100
                        "energy": ResourceLimitsConfig(
                            limit=0,
                            resources=["energy"],
                            modifiers={"battery": 25},
                        ),
                        # Tech group limit = 0 + tech * 1 = 0 + 10 * 1 = 10
                        "tech_group": ResourceLimitsConfig(
                            limit=0,
                            resources=["weapon", "shield", "battery"],
                            modifiers={"tech": 1},
                        ),
                        # Tech and carbon have default limit (65535)
                    },
                ),
            ),
            map_builder=RandomMapBuilder.Config(width=5, height=3, agents=1, seed=42),
        )
    )

    sim = Simulation(cfg)
    inv = _get_agent_inventory(sim, agent_id=0)

    # Map resource names to indices
    names = cfg.game.resource_names
    energy_idx = names.index("energy")
    carbon_idx = names.index("carbon")
    weapon_idx = names.index("weapon")
    battery_idx = names.index("battery")
    tech_idx = names.index("tech")

    # Verify all items were added correctly
    assert inv.get(tech_idx, 0) == 10, f"Expected tech=10, got {inv.get(tech_idx, 0)}"
    assert inv.get(battery_idx, 0) == 4, f"Expected battery=4, got {inv.get(battery_idx, 0)}"
    assert inv.get(energy_idx, 0) == 100, f"Expected energy=100, got {inv.get(energy_idx, 0)}"
    assert inv.get(weapon_idx, 0) == 2, f"Expected weapon=2, got {inv.get(weapon_idx, 0)}"
    assert inv.get(carbon_idx, 0) == 50, f"Expected carbon=50, got {inv.get(carbon_idx, 0)}"

    sim.close()


def test_initial_inventory_chained_modifiers():
    """Test a chain of modifier dependencies: A modifies B's limit, B modifies C's limit."""
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            obs=ObsConfig(width=3, height=3, num_tokens=32),
            actions=ActionsConfig(noop=NoopActionConfig()),
            resource_names=["level1", "level2", "level3"],
            agent=AgentConfig(
                inventory=InventoryConfig(
                    initial={
                        "level3": 5,  # No dependencies, provides limit for level2
                        "level2": 3,  # Depends on level3, provides limit for level1
                        "level1": 10,  # Depends on level2
                    },
                    limits={
                        # level1 limit = 0 + level2 * 5 = 0 + 3 * 5 = 15
                        "limit1": ResourceLimitsConfig(
                            limit=0,
                            resources=["level1"],
                            modifiers={"level2": 5},
                        ),
                        # level2 limit = 0 + level3 * 1 = 0 + 5 * 1 = 5
                        "limit2": ResourceLimitsConfig(
                            limit=0,
                            resources=["level2"],
                            modifiers={"level3": 1},
                        ),
                        # level3 has default limit (65535)
                    },
                ),
            ),
            map_builder=RandomMapBuilder.Config(width=5, height=3, agents=1, seed=42),
        )
    )

    sim = Simulation(cfg)
    inv = _get_agent_inventory(sim, agent_id=0)

    names = cfg.game.resource_names
    level1_idx = names.index("level1")
    level2_idx = names.index("level2")
    level3_idx = names.index("level3")

    assert inv.get(level3_idx, 0) == 5, f"Expected level3=5, got {inv.get(level3_idx, 0)}"
    assert inv.get(level2_idx, 0) == 3, f"Expected level2=3, got {inv.get(level2_idx, 0)}"
    assert inv.get(level1_idx, 0) == 10, f"Expected level1=10, got {inv.get(level1_idx, 0)}"

    sim.close()


def test_initial_inventory_exceeds_limit():
    """Test that initial inventory is clamped to effective limit.

    If the initial inventory specifies more than the effective limit allows,
    the actual amount should be clamped.

    Note: This test currently verifies that initial inventory ignores limits,
    which is the current implementation behavior. The ignore_limits feature
    allows initial inventory to exceed normal limits during initialization.
    """
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            obs=ObsConfig(width=3, height=3, num_tokens=32),
            actions=ActionsConfig(noop=NoopActionConfig()),
            resource_names=["modifier", "limited"],
            agent=AgentConfig(
                inventory=InventoryConfig(
                    initial={
                        "modifier": 2,  # Gives limit of 2 * 5 = 10
                        "limited": 20,  # Requested 20, allowed because initial ignores limits
                    },
                    limits={
                        "cap": ResourceLimitsConfig(
                            limit=0,
                            resources=["limited"],
                            modifiers={"modifier": 5},
                        ),
                        # modifier has default limit (65535)
                    },
                ),
            ),
            map_builder=RandomMapBuilder.Config(width=5, height=3, agents=1, seed=42),
        )
    )

    sim = Simulation(cfg)
    inv = _get_agent_inventory(sim, agent_id=0)

    names = cfg.game.resource_names
    modifier_idx = names.index("modifier")
    limited_idx = names.index("limited")

    assert inv.get(modifier_idx, 0) == 2, f"Expected modifier=2, got {inv.get(modifier_idx, 0)}"
    # Initial inventory ignores limits, so the full 20 is added
    assert inv.get(limited_idx, 0) == 20, f"Expected limited=20, got {inv.get(limited_idx, 0)}"

    sim.close()
