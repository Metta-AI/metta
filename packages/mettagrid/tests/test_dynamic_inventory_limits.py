"""Tests for dynamic inventory limits with modifiers.

This tests the feature where inventory limits can scale based on other items held.
For example, battery limit starts at 0, each gear adds +1 capacity.
"""

import pytest

from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    AgentConfig,
    GameConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    ObsConfig,
    ResourceLimitsConfig,
)
from mettagrid.map_builder.random import RandomMapBuilder
from mettagrid.simulator import Action, Simulation


@pytest.fixture
def sim_with_modifier_limits() -> Simulation:
    """Create a simulation where battery capacity depends on gear count."""
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            obs=ObsConfig(width=3, height=3, num_tokens=32),
            actions=ActionsConfig(noop=NoopActionConfig(), move=MoveActionConfig()),
            resource_names=["gear", "battery", "energy"],
            agent=AgentConfig(
                resource_limits={
                    "gear": ResourceLimitsConfig(resources=["gear"], limit=10),
                    # Battery limit starts at 0, each gear adds +5 capacity
                    "battery": ResourceLimitsConfig(
                        resources=["battery"],
                        limit=0,
                        modifiers={"gear": 5},
                    ),
                    # Energy limit starts at 0, each battery adds +25 capacity
                    "energy": ResourceLimitsConfig(
                        resources=["energy"],
                        limit=0,
                        modifiers={"battery": 25},
                    ),
                }
            ),
            map_builder=RandomMapBuilder.Config(width=5, height=3, agents=1, seed=7),
        )
    )
    return Simulation(cfg)


def _find_agent_id(grid_objects: dict[int, dict]) -> int:
    for obj in grid_objects.values():
        if "agent_id" in obj:
            return int(obj["agent_id"])
    raise AssertionError("No agent found in grid_objects")


def test_dynamic_limit_basic(sim_with_modifier_limits: Simulation):
    """Test that battery capacity scales with gear count."""
    sim = sim_with_modifier_limits

    agent_id = _find_agent_id(sim.grid_objects())
    agent = sim.agent(agent_id)

    # Initially can't add batteries (limit is 0, no gears)
    agent.set_inventory({"battery": 10})
    agent.set_action(Action(name="noop"))
    sim.step()
    assert agent.inventory.get("battery", 0) == 0, "Should not be able to add batteries without gears"

    # Add 2 gears (battery limit becomes 10)
    agent.set_inventory({"gear": 2, "battery": 10})
    agent.set_action(Action(name="noop"))
    sim.step()
    assert agent.inventory.get("gear", 0) == 2
    assert agent.inventory.get("battery", 0) == 10, "Should be able to add 10 batteries with 2 gears"

    # Try to add more batteries than limit allows
    agent.set_inventory({"gear": 2, "battery": 15})
    agent.set_action(Action(name="noop"))
    sim.step()
    assert agent.inventory.get("battery", 0) == 10, "Battery count should be clamped to limit (10)"


def test_dynamic_limit_chain(sim_with_modifier_limits: Simulation):
    """Test chained limits: gear -> battery -> energy."""
    sim = sim_with_modifier_limits

    agent_id = _find_agent_id(sim.grid_objects())
    agent = sim.agent(agent_id)

    # Add 2 gears, 10 batteries (limit from 2*5), then 250 energy (limit from 10*25)
    agent.set_inventory({"gear": 2, "battery": 10, "energy": 250})
    agent.set_action(Action(name="noop"))
    sim.step()

    inv = agent.inventory
    assert inv.get("gear", 0) == 2
    assert inv.get("battery", 0) == 10
    assert inv.get("energy", 0) == 250, f"Expected 250 energy, got {inv.get('energy', 0)}"

    # Try to exceed energy limit
    agent.set_inventory({"gear": 2, "battery": 10, "energy": 300})
    agent.set_action(Action(name="noop"))
    sim.step()

    inv = agent.inventory
    assert inv.get("energy", 0) == 250, "Energy should be clamped to 250 (10 batteries * 25)"


def test_modifiers_without_base_limit():
    """Test that zero base limit with modifier works correctly."""
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            obs=ObsConfig(width=3, height=3, num_tokens=32),
            actions=ActionsConfig(noop=NoopActionConfig(), move=MoveActionConfig()),
            resource_names=["key", "chest"],
            agent=AgentConfig(
                resource_limits={
                    "key": ResourceLimitsConfig(resources=["key"], limit=5),
                    # Can only open chests if you have keys (1 chest capacity per key)
                    "chest": ResourceLimitsConfig(
                        resources=["chest"],
                        limit=0,
                        modifiers={"key": 1},
                    ),
                }
            ),
            map_builder=RandomMapBuilder.Config(width=5, height=3, agents=1, seed=7),
        )
    )
    sim = Simulation(cfg)

    agent_id = _find_agent_id(sim.grid_objects())
    agent = sim.agent(agent_id)

    # Can't add chests without keys
    agent.set_inventory({"chest": 3})
    agent.set_action(Action(name="noop"))
    sim.step()
    assert agent.inventory.get("chest", 0) == 0

    # Add 3 keys, now can add 3 chests
    agent.set_inventory({"key": 3, "chest": 3})
    agent.set_action(Action(name="noop"))
    sim.step()
    assert agent.inventory.get("key", 0) == 3
    assert agent.inventory.get("chest", 0) == 3


def test_resource_limits_config_with_modifiers():
    """Test that ResourceLimitsConfig correctly stores modifiers."""
    config = ResourceLimitsConfig(
        resources=["battery"],
        limit=0,
        modifiers={"gear": 5, "wrench": 3},
    )

    assert config.resources == ["battery"]
    assert config.limit == 0
    assert config.modifiers == {"gear": 5, "wrench": 3}


def test_resource_limits_config_default_modifiers():
    """Test that ResourceLimitsConfig has empty modifiers by default."""
    config = ResourceLimitsConfig(
        resources=["gold"],
        limit=100,
    )

    assert config.resources == ["gold"]
    assert config.limit == 100
    assert config.modifiers == {}
