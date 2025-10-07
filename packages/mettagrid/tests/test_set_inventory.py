import pytest

from mettagrid.config.mettagrid_config import ActionConfig, ActionsConfig, GameConfig, MettaGridConfig
from mettagrid.core import MettaGridCore
from mettagrid.map_builder.random import RandomMapBuilder


@pytest.fixture
def env_with_resources() -> MettaGridCore:
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            obs_width=3,
            obs_height=3,
            num_observation_tokens=32,
            actions=ActionsConfig(
                noop=ActionConfig(),
                move=ActionConfig(),
                rotate=ActionConfig(),
            ),
            resource_names=["red", "blue", "green"],
            map_builder=RandomMapBuilder.Config(
                width=5,
                height=3,
                agents=1,
                seed=7,
            ),
        )
    )
    return MettaGridCore(cfg)


def _find_agent_id(grid_objects: dict[int, dict]) -> int:
    for obj in grid_objects.values():
        if "agent_id" in obj:
            return int(obj["agent_id"])
    raise AssertionError("No agent found in grid_objects")


def test_set_inventory_sets_and_clears(env_with_resources: MettaGridCore):
    env = env_with_resources
    env.reset()

    # Resolve resource name -> id
    name_to_id = {name: idx for idx, name in enumerate(env.resource_names)}

    agent_id = _find_agent_id(env.grid_objects())

    # 1) Set initial inventory: red=3, blue=2
    inv1 = {name_to_id["red"]: 3, name_to_id["blue"]: 2}
    env.core_env.set_inventory(agent_id, inv1)

    objs_after_first = env.grid_objects()
    # Find agent inventory dict (id -> qty)
    agent_inventory_1 = None
    for obj in objs_after_first.values():
        if obj.get("agent_id") == agent_id:
            agent_inventory_1 = obj.get("inventory", {})
            break
    assert agent_inventory_1 is not None
    assert int(agent_inventory_1.get(name_to_id["red"], 0)) == 3
    assert int(agent_inventory_1.get(name_to_id["blue"], 0)) == 2
    assert int(agent_inventory_1.get(name_to_id["green"], 0)) == 0

    # 2) Update inventory with only red=1; blue should be cleared (removed)
    inv2 = {name_to_id["red"]: 1}
    env.core_env.set_inventory(agent_id, inv2)

    objs_after_second = env.grid_objects()
    agent_inventory_2 = None
    for obj in objs_after_second.values():
        if obj.get("agent_id") == agent_id:
            agent_inventory_2 = obj.get("inventory", {})
            break
    assert agent_inventory_2 is not None
    assert int(agent_inventory_2.get(name_to_id["red"], 0)) == 1
    # blue should be absent or zero
    assert int(agent_inventory_2.get(name_to_id["blue"], 0)) == 0
    # green remains zero
    assert int(agent_inventory_2.get(name_to_id["green"], 0)) == 0
