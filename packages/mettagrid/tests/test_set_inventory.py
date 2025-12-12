import pytest

from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    GameConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    ObsConfig,
)
from mettagrid.map_builder.random_map import RandomMapBuilder
from mettagrid.simulator import Action, Simulation


@pytest.fixture
def sim_with_resources() -> Simulation:
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            obs=ObsConfig(width=3, height=3, num_tokens=32),
            actions=ActionsConfig(noop=NoopActionConfig(), move=MoveActionConfig()),
            resource_names=["red", "blue", "green"],
            map_builder=RandomMapBuilder.Config(width=5, height=3, agents=1, seed=7),
        )
    )
    return Simulation(cfg)


def _find_agent_id(grid_objects: dict[int, dict]) -> int:
    for obj in grid_objects.values():
        if "agent_id" in obj:
            return int(obj["agent_id"])
    raise AssertionError("No agent found in grid_objects")


def test_set_inventory_sets_and_clears(sim_with_resources: Simulation):
    sim = sim_with_resources

    agent_id = _find_agent_id(sim.grid_objects())
    agent = sim.agent(agent_id)

    # 1) Set initial inventory: red=3, blue=2
    inv1 = {"red": 3, "blue": 2}
    agent.set_inventory(inv1)

    # Step to update observations
    agent.set_action(Action(name="noop"))
    sim.step()

    # Verify using the inventory property
    inv_from_property = agent.inventory
    assert inv_from_property.get("red", 0) == 3, f"Expected red=3, got {inv_from_property.get('red', 0)}"
    assert inv_from_property.get("blue", 0) == 2, f"Expected blue=2, got {inv_from_property.get('blue', 0)}"
    assert inv_from_property.get("green", 0) == 0, f"Expected green=0, got {inv_from_property.get('green', 0)}"

    # Also verify through grid_objects for backward compatibility
    # Note: grid_objects returns inventory with integer indices, not resource names
    objs_after_first = sim.grid_objects()
    agent_inventory_1 = None
    for obj in objs_after_first.values():
        if obj.get("agent_id") == agent_id:
            agent_inventory_1 = obj.get("inventory", {})
            break
    assert agent_inventory_1 is not None
    # grid_objects uses resource indices as keys
    red_idx = sim.resource_names.index("red")
    blue_idx = sim.resource_names.index("blue")
    green_idx = sim.resource_names.index("green")
    assert int(agent_inventory_1.get(red_idx, 0)) == 3
    assert int(agent_inventory_1.get(blue_idx, 0)) == 2
    assert int(agent_inventory_1.get(green_idx, 0)) == 0

    # 2) Update inventory with only red=1; blue should be cleared (removed)
    inv2 = {"red": 1}
    agent.set_inventory(inv2)

    # Step to update observations
    agent.set_action(Action(name="noop"))
    sim.step()

    # Verify using the inventory property
    inv_from_property = agent.inventory
    assert inv_from_property.get("red", 0) == 1, f"Expected red=1, got {inv_from_property.get('red', 0)}"
    assert inv_from_property.get("blue", 0) == 0, f"Expected blue=0, got {inv_from_property.get('blue', 0)}"
    assert inv_from_property.get("green", 0) == 0, f"Expected green=0, got {inv_from_property.get('green', 0)}"

    # Also verify through grid_objects
    objs_after_second = sim.grid_objects()
    agent_inventory_2 = None
    for obj in objs_after_second.values():
        if obj.get("agent_id") == agent_id:
            agent_inventory_2 = obj.get("inventory", {})
            break
    assert agent_inventory_2 is not None
    # grid_objects uses resource indices as keys
    assert int(agent_inventory_2.get(red_idx, 0)) == 1
    # blue should be absent or zero
    assert int(agent_inventory_2.get(blue_idx, 0)) == 0
    # green remains zero
    assert int(agent_inventory_2.get(green_idx, 0)) == 0
