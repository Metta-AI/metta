"""Test that initial inventory correctly handles modifier dependencies via inventory_order.

When resource limits depend on modifier items (e.g., energy limit depends on battery count),
the user specifies inventory_order to ensure modifiers are added before dependent items.
"""

from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AgentConfig,
    GameConfig,
    MettaGridConfig,
    NoopActionConfig,
    ObsConfig,
    ResourceLimitsConfig,
)
from mettagrid.map_builder.random import RandomMapBuilder
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

    Without proper handling, if items are processed in the wrong order,
    dependent items may fail to be added because their limit would be 0.
    """
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            obs=ObsConfig(width=3, height=3, num_tokens=32),
            actions=ActionsConfig(noop=NoopActionConfig()),
            resource_names=["energy", "carbon", "weapon", "shield", "battery", "tech"],
            agent=AgentConfig(
                initial_inventory={
                    "tech": 10,  # Modifier for weapon/shield/battery limit
                    "battery": 4,  # Modifier for energy limit AND depends on tech
                    "energy": 100,  # Depends on battery
                    "weapon": 2,  # Depends on tech
                    "carbon": 50,  # No dependencies
                },
                # Explicit order: tech first (modifies battery limit), then battery (modifies energy limit)
                inventory_order=["tech", "battery"],
                resource_limits={
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
                },
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
                initial_inventory={
                    "level3": 5,  # No dependencies, provides limit for level2
                    "level2": 3,  # Depends on level3, provides limit for level1
                    "level1": 10,  # Depends on level2
                },
                # Explicit order: level3 first, then level2 (chained dependencies)
                inventory_order=["level3", "level2"],
                resource_limits={
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
                },
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
    """
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            obs=ObsConfig(width=3, height=3, num_tokens=32),
            actions=ActionsConfig(noop=NoopActionConfig()),
            resource_names=["modifier", "limited"],
            agent=AgentConfig(
                initial_inventory={
                    "modifier": 2,  # Gives limit of 2 * 5 = 10
                    "limited": 20,  # Requested 20, but limit is only 10
                },
                # Explicit order: modifier first so limit is computed correctly
                inventory_order=["modifier"],
                resource_limits={
                    "cap": ResourceLimitsConfig(
                        limit=0,
                        resources=["limited"],
                        modifiers={"modifier": 5},
                    ),
                },
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
    # Limited should be clamped to effective limit (2 * 5 = 10)
    assert inv.get(limited_idx, 0) == 10, f"Expected limited=10 (clamped), got {inv.get(limited_idx, 0)}"

    sim.close()


def test_initial_inventory_without_order():
    """Test that initial inventory still works without specifying inventory_order.

    When inventory_order is not specified (empty), items should still be added
    correctly if there are no dependencies.
    """
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            obs=ObsConfig(width=3, height=3, num_tokens=32),
            actions=ActionsConfig(noop=NoopActionConfig()),
            resource_names=["red", "blue", "green"],
            agent=AgentConfig(
                initial_inventory={
                    "red": 10,
                    "blue": 20,
                    "green": 30,
                },
                # No inventory_order - should still work for independent items
            ),
            map_builder=RandomMapBuilder.Config(width=5, height=3, agents=1, seed=42),
        )
    )

    sim = Simulation(cfg)
    inv = _get_agent_inventory(sim, agent_id=0)

    names = cfg.game.resource_names
    red_idx = names.index("red")
    blue_idx = names.index("blue")
    green_idx = names.index("green")

    assert inv.get(red_idx, 0) == 10, f"Expected red=10, got {inv.get(red_idx, 0)}"
    assert inv.get(blue_idx, 0) == 20, f"Expected blue=20, got {inv.get(blue_idx, 0)}"
    assert inv.get(green_idx, 0) == 30, f"Expected green=30, got {inv.get(green_idx, 0)}"

    sim.close()
