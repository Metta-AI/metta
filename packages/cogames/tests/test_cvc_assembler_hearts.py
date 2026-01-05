from __future__ import annotations

from collections.abc import Mapping

from cogames.cogs_vs_clips.stations import CvCAssemblerConfig
from mettagrid.config.mettagrid_config import MettaGridConfig, ResourceLimitsConfig
from mettagrid.simulator import Simulation

RESOURCES = [
    "energy",
    "carbon",
    "oxygen",
    "germanium",
    "silicon",
    "heart",
    "decoder",
    "modulator",
    "resonator",
    "scrambler",
]

ASCII_MAP = [
    list("#####"),
    list("#.@.#"),
    list("#@&@#"),
    list("#.@.#"),
    list("#####"),
]

_BASE_STATION = CvCAssemblerConfig()
FIRST_HEART_COST = _BASE_STATION.first_heart_cost
ADDITIONAL_HEART_COST = _BASE_STATION.additional_heart_cost


def _make_simulation(num_agents: int = 4, ascii_map: list[list[str]] = ASCII_MAP) -> Simulation:
    cfg = MettaGridConfig.EmptyRoom(num_agents=num_agents, with_walls=True)
    cfg.game.resource_names = RESOURCES
    cfg.game.agent.inventory.default_limit = 255
    cfg.game.agent.inventory.limits = {name: ResourceLimitsConfig(limit=255, resources=[name]) for name in RESOURCES}
    cfg.game.agent.inventory.initial = {name: 0 for name in RESOURCES}

    cfg.game.actions.noop.enabled = True
    cfg.game.actions.move.enabled = True
    cfg.game.actions.change_vibe.enabled = True

    assembler_cfg = CvCAssemblerConfig(
        first_heart_cost=FIRST_HEART_COST, additional_heart_cost=ADDITIONAL_HEART_COST
    ).station_cfg()
    cfg.game.objects["assembler"] = assembler_cfg

    cfg = cfg.with_ascii_map(
        ascii_map,
        char_to_map_name={"#": "wall", "@": "agent.agent", ".": "empty", "&": "assembler"},
    )
    return Simulation(cfg)


def _agent_positions(sim: Simulation) -> tuple[dict[tuple[int, int], int], tuple[int, int]]:
    assembler_pos: tuple[int, int] | None = None
    positions: dict[tuple[int, int], int] = {}
    for obj in sim.grid_objects().values():
        if "agent_id" in obj:
            positions[(obj["r"], obj["c"])] = obj["agent_id"]
        elif obj.get("type_name") == "assembler":
            assembler_pos = (obj["r"], obj["c"])
    assert assembler_pos is not None, "Assembler missing from map"
    return positions, assembler_pos


def _capture_inventories(sim: Simulation) -> dict[int, dict[str, int]]:
    res_idx = {name: idx for idx, name in enumerate(sim.resource_names)}
    state: dict[int, dict[str, int]] = {}
    for obj in sim.grid_objects().values():
        agent_id = obj.get("agent_id")
        if agent_id is None:
            continue
        raw_inventory: Mapping[int, int] = obj.get("inventory", {})
        state[agent_id] = {name: int(raw_inventory.get(idx, 0)) for name, idx in res_idx.items()}
    return state


def _assign_inventories(sim: Simulation, updates: Mapping[int, dict[str, int]]) -> None:
    for agent_id, inventory in updates.items():
        sim.agent(agent_id).set_inventory(inventory)


def _step(sim: Simulation, actions: Mapping[int, str], default: str = "noop") -> None:
    for agent_id in range(sim.num_agents):
        sim.agent(agent_id).set_action(actions.get(agent_id, default))
    sim.step()


def _move_action(from_pos: tuple[int, int], to_pos: tuple[int, int]) -> str:
    dr = to_pos[0] - from_pos[0]
    dc = to_pos[1] - from_pos[1]
    if dr == 0 and dc == 1:
        return "move_east"
    if dr == 0 and dc == -1:
        return "move_west"
    if dr == 1 and dc == 0:
        return "move_south"
    if dr == -1 and dc == 0:
        return "move_north"
    raise ValueError(f"Unsupported delta from {from_pos} to {to_pos}")


def _expected_inputs(num_hearts: int) -> dict[str, int]:
    assert num_hearts >= 1
    scale = FIRST_HEART_COST + ADDITIONAL_HEART_COST * (num_hearts - 1)
    return {
        "carbon": scale,
        "oxygen": scale,
        "germanium": max(1, scale // 5),
        "silicon": 3 * scale,
        "energy": 0,
    }


def _total(state: Mapping[int, dict[str, int]], resource: str) -> int:
    return sum(agent_state[resource] for agent_state in state.values())


def test_single_heart_recipe_consumes_first_cost_and_pools_resources() -> None:
    sim = _make_simulation()
    try:
        positions, assembler_pos = _agent_positions(sim)
        north = positions[(1, 2)]
        west = positions[(2, 1)]
        east = positions[(2, 3)]
        south = positions[(3, 2)]

        _assign_inventories(
            sim,
            {
                north: {"carbon": 10, "oxygen": 10},
                west: {"carbon": 10, "energy": 5},
                east: {"oxygen": 10, "silicon": 20},
                south: {"germanium": 5, "silicon": 30, "energy": 15},
            },
        )

        _step(sim, {west: "change_vibe_heart_a"})
        before = _capture_inventories(sim)
        _step(sim, {west: _move_action((2, 1), assembler_pos)})
        after = _capture_inventories(sim)

        expected_inputs = _expected_inputs(1)
        for resource, expected in expected_inputs.items():
            assert _total(before, resource) - _total(after, resource) == expected, resource
        assert _total(after, "heart") - _total(before, "heart") == 1

        assert after[north]["carbon"] < before[north]["carbon"]
        assert after[west]["carbon"] < before[west]["carbon"]
        assert after[east]["oxygen"] < before[east]["oxygen"]
        assert after[north]["oxygen"] < before[north]["oxygen"]
        assert after[east]["silicon"] < before[east]["silicon"]
        assert after[south]["silicon"] < before[south]["silicon"]
    finally:
        sim.close()


def test_multi_heart_recipe_uses_additional_cost_and_shared_inventories() -> None:
    sim = _make_simulation()
    try:
        positions, assembler_pos = _agent_positions(sim)
        north = positions[(1, 2)]
        west = positions[(2, 1)]
        east = positions[(2, 3)]
        south = positions[(3, 2)]

        _assign_inventories(
            sim,
            {
                north: {"carbon": 15, "oxygen": 15},
                west: {"carbon": 15, "energy": 10},
                east: {"oxygen": 15, "silicon": 35},
                south: {"germanium": 7, "silicon": 40, "energy": 20},
            },
        )

        _step(
            sim,
            {west: "change_vibe_heart_a", north: "change_vibe_heart_a"},
        )
        before = _capture_inventories(sim)
        _step(sim, {west: _move_action((2, 1), assembler_pos)})
        after = _capture_inventories(sim)

        expected_inputs = _expected_inputs(2)
        for resource, expected in expected_inputs.items():
            assert _total(before, resource) - _total(after, resource) == expected, resource
        assert _total(after, "heart") - _total(before, "heart") == 2

        carbon_consumers = sum(after[agent_id]["carbon"] < before[agent_id]["carbon"] for agent_id in (north, west))
        assert carbon_consumers == 2

        silicon_consumers = sum(after[agent_id]["silicon"] < before[agent_id]["silicon"] for agent_id in (east, south))
        assert silicon_consumers == 2
    finally:
        sim.close()


def test_two_agent_heart_chorus_map_outputs_two_hearts() -> None:
    sim = _make_simulation(num_agents=2)
    try:
        positions, assembler_pos = _agent_positions(sim)
        assert len(positions) == 2
        (first_pos, first_id), (second_pos, second_id) = sorted(positions.items())

        _assign_inventories(
            sim,
            {
                first_id: {"carbon": 15, "oxygen": 5},
                second_id: {"oxygen": 10, "germanium": 3, "silicon": 45},
            },
        )

        _step(sim, {first_id: "change_vibe_heart_a", second_id: "change_vibe_heart_a"})
        before = _capture_inventories(sim)
        _step(sim, {first_id: _move_action(first_pos, assembler_pos)})
        after = _capture_inventories(sim)

        expected_inputs = _expected_inputs(2)
        for resource, expected in expected_inputs.items():
            assert _total(before, resource) - _total(after, resource) == expected, resource
        assert _total(after, "heart") - _total(before, "heart") == 2
        assert after[first_id]["heart"] - before[first_id]["heart"] == 1
        assert after[second_id]["heart"] - before[second_id]["heart"] == 1

        _step(sim, {})
        after_noop = _capture_inventories(sim)
        assert _total(after_noop, "heart") == _total(after, "heart")
    finally:
        sim.close()
