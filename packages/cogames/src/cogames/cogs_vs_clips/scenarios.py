"""Scenario definitions for the Cogs vs Clips environments.

This module now contains curated maps instead of placeholder random layouts. Each scenario declares a
manual grid layout that favours a particular style of play (tight logistics, sprawling exploration,
maze navigation, etc.) so that we can iterate on balance without changing engine code.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from mettagrid.config.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    AssemblerConfig,
    ChangeGlyphActionConfig,
    ChestConfig,
    GameConfig,
    MettaGridConfig,
    WallConfig,
)
from mettagrid.map_builder.map_builder import GameMap, MapBuilder, MapBuilderConfig, map_grid_dtype

from .stations import (
    make_carbon_extractor,
    make_charger,
    make_chest,
    make_core_assembler,
    make_geranium_extractor,
    make_oxygen_extractor,
    make_silicon_extractor,
)
from .stations import (
    resources as RESOURCE_NAMES,
)

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


class StaticGridMapBuilder(MapBuilder):
    """Map builder that renders a pre-computed string grid."""

    class Config(MapBuilderConfig["StaticGridMapBuilder"]):
        grid: list[list[str]]

        @property
        def width(self) -> int:
            return len(self.grid[0]) if self.grid else 0

        @property
        def height(self) -> int:
            return len(self.grid)

    def __init__(self, config: Config):
        self._config = config

    def build(self) -> GameMap:
        grid = np.array(self._config.grid, dtype=map_grid_dtype)
        return GameMap(grid)


@dataclass(frozen=True)
class ScenarioParams:
    """Convenience bundle when generating maps."""

    width: int
    height: int
    fill: str = "wall"


GameObjectConfig = WallConfig | AssemblerConfig | ChestConfig

Grid = npt.NDArray[np.str_]  # type alias for clarity


def _new_grid(params: ScenarioParams) -> Grid:
    return np.full((params.height, params.width), params.fill, dtype=map_grid_dtype)


def _carve_rect(
    grid: Grid,
    *,
    x0: int,
    y0: int,
    width: int,
    height: int,
    fill: str = "empty",
) -> None:
    grid[y0 : y0 + height, x0 : x0 + width] = fill


def _carve_border(grid: Grid, *, thickness: int = 1, value: str = "wall") -> None:
    grid[:thickness, :] = value
    grid[-thickness:, :] = value
    grid[:, :thickness] = value
    grid[:, -thickness:] = value


def _carve_corridor(
    grid: Grid,
    *,
    start: tuple[int, int],
    end: tuple[int, int],
    width: int = 1,
    fill: str = "empty",
) -> None:
    """Carve an L-shaped corridor between `start` and `end`."""

    x0, y0 = start
    x1, y1 = end
    if x0 == x1:
        xi0 = max(0, x0 - width // 2)
        xi1 = min(grid.shape[1], x0 + (width + 1) // 2)
        y_start, y_stop = sorted((y0, y1))
        grid[y_start : y_stop + 1, xi0:xi1] = fill
        return
    if y0 == y1:
        yi0 = max(0, y0 - width // 2)
        yi1 = min(grid.shape[0], y0 + (width + 1) // 2)
        x_start, x_stop = sorted((x0, x1))
        grid[yi0:yi1, x_start : x_stop + 1] = fill
        return

    mid = (x1, y0)
    _carve_corridor(grid, start=start, end=mid, width=width, fill=fill)
    _carve_corridor(grid, start=mid, end=end, width=width, fill=fill)


def _place(grid: Grid, *, x: int, y: int, value: str) -> None:
    if not (0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]):
        raise ValueError(f"Point ({x}, {y}) is out of bounds for grid {grid.shape}")
    grid[y, x] = value


def _place_many(grid: Grid, *, positions: Iterable[tuple[int, int]], value: str) -> None:
    for x, y in positions:
        _place(grid, x=x, y=y, value=value)


def _make_actions(*, allow_item_exchange: bool = False) -> ActionsConfig:
    return ActionsConfig(
        move=ActionConfig(consumed_resources={"energy": 1}),
        noop=ActionConfig(),
        change_glyph=ChangeGlyphActionConfig(number_of_glyphs=16),
        put_items=ActionConfig(enabled=allow_item_exchange),
        get_items=ActionConfig(enabled=allow_item_exchange),
    )


def _make_agent(
    *,
    energy_capacity: int = 100,
    default_limit: int = 10,
    start_energy: int = 100,
) -> AgentConfig:
    return AgentConfig(
        default_resource_limit=default_limit,
        resource_limits={
            "heart": 1,
            "energy": energy_capacity,
        },
        rewards=AgentRewards(
            inventory={"heart": 1},
        ),
        initial_inventory={"energy": start_energy},
    )


def _make_game(
    *,
    grid: Grid,
    objects: dict[str, GameObjectConfig],
    num_cogs: int,
    max_steps: int = 800,
    allow_item_exchange: bool = False,
    energy_capacity: int = 100,
    default_limit: int = 10,
    start_energy: int = 100,
) -> MettaGridConfig:
    return MettaGridConfig(
        game=GameConfig(
            resource_names=list(RESOURCE_NAMES),
            num_agents=num_cogs,
            max_steps=max_steps,
            actions=_make_actions(allow_item_exchange=allow_item_exchange),
            agent=_make_agent(
                energy_capacity=energy_capacity,
                default_limit=default_limit,
                start_energy=start_energy,
            ),
            objects=objects,
            map_builder=StaticGridMapBuilder.Config(grid=grid.tolist()),
        )
    )


# ---------------------------------------------------------------------------
# Scenario generators
# ---------------------------------------------------------------------------


def _tutorial_map(*, num_cogs: int, richer_layout: bool) -> MettaGridConfig:
    params = ScenarioParams(width=18, height=18)
    grid = _new_grid(params)
    _carve_border(grid)

    # Base room with simple four-way access.
    _carve_rect(grid, x0=3, y0=3, width=12, height=12)
    _carve_rect(grid, x0=7, y0=7, width=4, height=4)

    # Short corridors to outer alcoves.
    center_x, center_y = params.width // 2, params.height // 2
    _carve_corridor(grid, start=(center_x, center_y - 2), end=(center_x, 1), width=2)
    _carve_corridor(grid, start=(center_x, center_y + 2), end=(center_x, params.height - 2), width=2)
    _carve_corridor(grid, start=(center_x - 2, center_y), end=(1, center_y), width=2)
    _carve_corridor(grid, start=(center_x + 2, center_y), end=(params.width - 2, center_y), width=2)

    # Object palette.
    objects = {
        "wall": WallConfig(type_id=1),
        "assembler_core": make_core_assembler(name="assembler_core"),
        "central_chest": make_chest(deposit_positions=("N", "S"), withdrawal_positions=("E", "W")),
        "charger_core": make_charger(name="charger_core", energy_output=60),
        "carbon_extractor_basic": make_carbon_extractor(name="carbon_extractor_basic", output_per_cycle=2),
        "oxygen_extractor_basic": make_oxygen_extractor(name="oxygen_extractor_basic", output_per_cycle=12),
        "geranium_extractor_slow": make_geranium_extractor(
            name="geranium_extractor_slow", output_per_cycle=6, cooldown=80
        ),
        "silicon_extractor_basic": make_silicon_extractor(name="silicon_extractor_basic", energy_cost=8),
    }

    # Place base objects.
    _place(grid, x=center_x, y=center_y, value="assembler_core")
    _place(grid, x=center_x + 1, y=center_y, value="central_chest")
    _place_many(
        grid,
        positions=[(center_x - 1, center_y), (center_x, center_y - 1), (center_x, center_y + 1)],
        value="charger_core",
    )

    # Simple resource layout close to base.
    _place_many(
        grid,
        positions=[
            (center_x, 2),
            (center_x, params.height - 3),
            (2, center_y),
            (params.width - 3, center_y),
        ],
        value="carbon_extractor_basic",
    )
    _place_many(
        grid,
        positions=[
            (center_x - 4, 4),
            (center_x + 4, 4),
            (center_x - 4, params.height - 5),
            (center_x + 4, params.height - 5),
        ],
        value="oxygen_extractor_basic",
    )

    if richer_layout:
        _place_many(
            grid,
            positions=[(2, 2), (params.width - 3, 2), (2, params.height - 3), (params.width - 3, params.height - 3)],
            value="geranium_extractor_slow",
        )
        _place_many(
            grid,
            positions=[(center_x - 5, center_y - 5), (center_x + 5, center_y + 5)],
            value="silicon_extractor_basic",
        )

    spawn_offsets = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for idx in range(num_cogs):
        ox, oy = spawn_offsets[idx % len(spawn_offsets)]
        _place(grid, x=center_x + ox, y=center_y + oy, value="agent.agent")

    return _make_game(grid=grid, objects=objects, num_cogs=num_cogs, max_steps=450)


def _machina_small_drop() -> MettaGridConfig:
    params = ScenarioParams(width=26, height=26)
    grid = _new_grid(params)
    _carve_border(grid)

    # Compact base with resource alcoves that are easy to spot.
    base_size = 12
    base_start = (params.width // 2 - base_size // 2, params.height // 2 - base_size // 2)
    _carve_rect(grid, x0=base_start[0], y0=base_start[1], width=base_size, height=base_size)

    # Carve four shallow rooms with short corridors.
    offsets = [(-10, 0), (10, 0), (0, -10), (0, 10)]
    for dx, dy in offsets:
        room_center = (params.width // 2 + dx, params.height // 2 + dy)
        _carve_corridor(
            grid,
            start=(params.width // 2, params.height // 2),
            end=room_center,
            width=3,
        )
        _carve_rect(grid, x0=room_center[0] - 3, y0=room_center[1] - 3, width=6, height=6)

    objects = {
        "wall": WallConfig(type_id=1),
        "assembler_spoke": make_core_assembler(name="assembler_spoke", cooldown=2),
        "team_chest": make_chest(deposit_positions=("E", "W"), withdrawal_positions=("N", "S")),
        "charger_quick": make_charger(name="charger_quick", energy_output=70, cooldown=1),
        "carbon_steady": make_carbon_extractor(name="carbon_steady", output_per_cycle=3, cooldown=2),
        "oxygen_steady": make_oxygen_extractor(name="oxygen_steady", output_per_cycle=15, cooldown=2),
        "geranium_spot": make_geranium_extractor(name="geranium_spot", output_per_cycle=8, cooldown=90),
        "silicon_core": make_silicon_extractor(name="silicon_core", output_per_cycle=2, energy_cost=8, cooldown=2),
    }

    cx, cy = params.width // 2, params.height // 2
    _place(grid, x=cx, y=cy, value="assembler_spoke")
    _place(grid, x=cx + 1, y=cy, value="team_chest")
    _place_many(
        grid,
        positions=[(cx - 1, cy), (cx, cy - 1), (cx, cy + 1), (cx + 2, cy)],
        value="charger_quick",
    )

    carbon_sites = [(cx - 10, cy), (cx + 10, cy), (cx, cy - 10), (cx, cy + 10)]
    oxygen_sites = [(cx - 10, cy - 4), (cx + 10, cy + 4), (cx - 4, cy + 10), (cx + 4, cy - 10)]
    _place_many(grid, positions=carbon_sites, value="carbon_steady")
    _place_many(grid, positions=oxygen_sites, value="oxygen_steady")
    _place_many(grid, positions=[(cx - 10, cy - 10), (cx + 10, cy + 10)], value="geranium_spot")
    _place_many(grid, positions=[(cx + 5, cy + 5), (cx - 5, cy - 5)], value="silicon_core")

    spawn_offsets = [(-2, 0), (-1, 1), (1, -1), (2, 0)]
    for idx, (ox, oy) in enumerate(spawn_offsets):
        _place(grid, x=cx + ox, y=cy + oy, value="agent.agent")
        if idx == 3:
            break

    return _make_game(grid=grid, objects=objects, num_cogs=4, max_steps=600)


def _machina_frontier() -> MettaGridConfig:
    params = ScenarioParams(width=200, height=200)
    grid = _new_grid(params)
    _carve_border(grid, thickness=2)

    center = (params.width // 2, params.height // 2)
    base_half = 15
    _carve_rect(
        grid,
        x0=center[0] - base_half,
        y0=center[1] - base_half,
        width=base_half * 2,
        height=base_half * 2,
    )

    # Inner plaza inside the base for quick manoeuvring.
    _carve_rect(
        grid,
        x0=center[0] - 8,
        y0=center[1] - 8,
        width=16,
        height=16,
    )

    # Corridors fanning out to far rooms.
    spokes = {
        "north": (center[0], 30),
        "south": (center[0], params.height - 31),
        "west": (30, center[1]),
        "east": (params.width - 31, center[1]),
    }

    for point in spokes.values():
        _carve_corridor(grid, start=center, end=point, width=5)
        _carve_rect(grid, x0=point[0] - 10, y0=point[1] - 10, width=20, height=20)

    # Edge chambers with higher-tier resources.
    edge_rooms = [
        (20, 20),
        (params.width - 40, 20),
        (20, params.height - 40),
        (params.width - 40, params.height - 40),
    ]
    for ex, ey in edge_rooms:
        _carve_rect(grid, x0=ex, y0=ey, width=20, height=20)

    # Connect spokes to edge chambers via meandering paths for "room-to-room" feel.
    _carve_corridor(grid, start=spokes["north"], end=(edge_rooms[0][0] + 10, edge_rooms[0][1] + 20), width=3)
    _carve_corridor(grid, start=spokes["north"], end=(edge_rooms[1][0] + 10, edge_rooms[1][1] + 20), width=3)
    _carve_corridor(grid, start=spokes["south"], end=(edge_rooms[2][0] + 10, edge_rooms[2][1] + 0), width=3)
    _carve_corridor(grid, start=spokes["south"], end=(edge_rooms[3][0] + 10, edge_rooms[3][1] + 0), width=3)

    objects = {
        "wall": WallConfig(type_id=1),
        "assembler_vault": make_core_assembler(name="assembler_vault", energy_cost=4, cooldown=2),
        "grand_chest": make_chest(
            deposit_positions=("N", "S"),
            withdrawal_positions=("E", "W"),
        ),
        "charger_core": make_charger(name="charger_core", energy_output=90, cooldown=1, type_id=5),
        "charger_waypoint": make_charger(name="charger_waypoint", energy_output=120, cooldown=2, type_id=38),
        "carbon_sparse": make_carbon_extractor(
            name="carbon_sparse", output_per_cycle=2, cooldown=5, type_id=32, tags=("frontier",)
        ),
        "carbon_rich": make_carbon_extractor(
            name="carbon_rich",
            output_per_cycle=6,
            cooldown=2,
            energy_cost=2,
            type_id=33,
            tags=("edge",),
        ),
        "oxygen_dense": make_oxygen_extractor(name="oxygen_dense", output_per_cycle=20, cooldown=2, type_id=34),
        "geranium_far": make_geranium_extractor(
            name="geranium_far", output_per_cycle=20, cooldown=140, type_id=35, tags=("edge",)
        ),
        "silicon_focus": make_silicon_extractor(
            name="silicon_focus", output_per_cycle=3, energy_cost=14, cooldown=3, type_id=36
        ),
    }

    cx, cy = center
    _place(grid, x=cx, y=cy, value="assembler_vault")
    _place(grid, x=cx + 2, y=cy, value="grand_chest")
    charger_sites = [
        (cx - 1, cy),
        (cx + 1, cy),
        (cx, cy - 1),
        (cx, cy + 1),
        (cx - 6, cy - 6),
        (cx + 6, cy + 6),
    ]
    _place_many(grid, positions=charger_sites, value="charger_core")

    # Waypoint chargers along the corridors.
    for point in spokes.values():
        _place(grid, x=point[0], y=point[1], value="charger_waypoint")

    # Resource gradient: sparse nodes near base, richer nodes near the edge.
    medium_band = 50
    carbon_sparse_positions = [
        (cx, cy - medium_band),
        (cx, cy + medium_band),
        (cx - medium_band, cy),
        (cx + medium_band, cy),
    ]
    _place_many(grid, positions=carbon_sparse_positions, value="carbon_sparse")

    carbon_rich_positions = [
        (edge_rooms[0][0] + 5, edge_rooms[0][1] + 5),
        (edge_rooms[1][0] + 15, edge_rooms[1][1] + 5),
        (edge_rooms[2][0] + 5, edge_rooms[2][1] + 15),
        (edge_rooms[3][0] + 15, edge_rooms[3][1] + 15),
    ]
    _place_many(grid, positions=carbon_rich_positions, value="carbon_rich")

    oxygen_positions = [
        (cx, 15),
        (cx, params.height - 16),
        (15, cy),
        (params.width - 16, cy),
    ]
    _place_many(grid, positions=oxygen_positions, value="oxygen_dense")

    geranium_positions = [
        (edge_rooms[0][0] + 10, 10),
        (edge_rooms[1][0] + 10, 10),
        (edge_rooms[2][0] + 10, params.height - 11),
        (edge_rooms[3][0] + 10, params.height - 11),
    ]
    _place_many(grid, positions=geranium_positions, value="geranium_far")

    silicon_positions = [
        (spokes["north"][0], spokes["north"][1] - 8),
        (spokes["south"][0], spokes["south"][1] + 8),
        (spokes["west"][0] - 8, spokes["west"][1]),
        (spokes["east"][0] + 8, spokes["east"][1]),
    ]
    _place_many(grid, positions=silicon_positions, value="silicon_focus")

    spawn_offsets = [(-3, -3), (-3, 3), (3, -3), (3, 3)]
    for ox, oy in spawn_offsets:
        _place(grid, x=cx + ox, y=cy + oy, value="agent.agent")

    # TODO: Place clip spawn points once the infestation system is exposed.

    return _make_game(
        grid=grid,
        objects=objects,
        num_cogs=4,
        max_steps=2000,
        energy_capacity=140,
        default_limit=16,
        start_energy=110,
    )


def _machina_maze() -> MettaGridConfig:
    params = ScenarioParams(width=72, height=72)
    grid = _new_grid(params)
    _carve_border(grid)

    base_origin = (params.width // 2 - 4, 4)
    _carve_rect(grid, x0=base_origin[0], y0=base_origin[1], width=8, height=8)

    # Single-file corridor out of the base into a winding maze.
    _carve_corridor(grid, start=(params.width // 2, 11), end=(params.width // 2, params.height - 6), width=1)

    # Snaking horizontal bands.
    for row in range(14, params.height - 6, 6):
        if (row // 6) % 2 == 0:
            _carve_corridor(grid, start=(2, row), end=(params.width - 3, row), width=1)
            _carve_corridor(grid, start=(params.width - 3, row), end=(params.width - 3, row + 3), width=1)
        else:
            _carve_corridor(grid, start=(params.width - 3, row), end=(2, row), width=1)
            _carve_corridor(grid, start=(2, row), end=(2, row + 3), width=1)

    # Dead-end rooms only reachable near the end.
    dead_ends = [(5, params.height - 8), (params.width - 6, params.height - 8)]
    for dx, dy in dead_ends:
        _carve_rect(grid, x0=dx, y0=dy, width=5, height=5)

    objects = {
        "wall": WallConfig(type_id=1),
        "assembler_escape": make_core_assembler(name="assembler_escape", cooldown=3),
        "maze_chest": make_chest(deposit_positions=("S",), withdrawal_positions=("N",)),
        "charger_lane": make_charger(name="charger_lane", energy_output=60, cooldown=2),
        "carbon_lane": make_carbon_extractor(name="carbon_lane", output_per_cycle=2, cooldown=3),
        "oxygen_lane": make_oxygen_extractor(name="oxygen_lane", output_per_cycle=14, cooldown=3),
        "geranium_cache": make_geranium_extractor(name="geranium_cache", output_per_cycle=10, cooldown=120),
        "silicon_cache": make_silicon_extractor(name="silicon_cache", output_per_cycle=2, energy_cost=12, cooldown=4),
    }

    center_x = params.width // 2
    _place(grid, x=center_x, y=base_origin[1] + 2, value="assembler_escape")
    _place(grid, x=center_x + 1, y=base_origin[1] + 2, value="maze_chest")
    _place(grid, x=center_x - 1, y=base_origin[1] + 2, value="charger_lane")
    _place(grid, x=center_x, y=base_origin[1] + 3, value="charger_lane")

    path_positions = [(center_x, row) for row in range(15, params.height - 6, 6)]
    _place_many(grid, positions=path_positions, value="charger_lane")

    _place_many(
        grid,
        positions=[(5, params.height - 7), (params.width - 6, params.height - 7)],
        value="geranium_cache",
    )
    _place_many(
        grid,
        positions=[(params.width // 2, params.height - 8)],
        value="silicon_cache",
    )
    _place_many(grid, positions=[(5, 15), (params.width - 6, 15)], value="carbon_lane")
    _place_many(grid, positions=[(10, 21), (params.width - 11, 27)], value="oxygen_lane")

    for offset in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        _place(grid, x=center_x + offset[0], y=base_origin[1] + 4 + offset[1], value="agent.agent")

    return _make_game(grid=grid, objects=objects, num_cogs=4, max_steps=900)


def _machina_branching() -> MettaGridConfig:
    params = ScenarioParams(width=96, height=64)
    grid = _new_grid(params)
    _carve_border(grid)

    center = (params.width // 2, params.height - 12)
    _carve_rect(grid, x0=center[0] - 6, y0=center[1] - 6, width=12, height=12)

    branch_points = [
        (center[0], center[1] - 14),
        (center[0] - 20, center[1] - 30),
        (center[0] + 20, center[1] - 30),
        (center[0] - 30, 12),
        (center[0] + 30, 12),
    ]
    path = [center, *branch_points]
    for start, end in zip(path, path[1:], strict=False):
        _carve_corridor(grid, start=start, end=end, width=3)
        _carve_rect(grid, x0=end[0] - 4, y0=end[1] - 4, width=8, height=8)

    leaf_nodes = [
        (12, 12),
        (params.width - 13, 12),
        (12, params.height // 2),
        (params.width - 13, params.height // 2),
    ]
    for node in leaf_nodes:
        _carve_corridor(grid, start=branch_points[-1], end=node, width=2)
        _carve_rect(grid, x0=node[0] - 3, y0=node[1] - 3, width=6, height=6)

    objects = {
        "wall": WallConfig(type_id=1),
        "assembler_branch": make_core_assembler(name="assembler_branch", cooldown=3),
        "branch_chest": make_chest(deposit_positions=("E", "W"), withdrawal_positions=("N", "S")),
        "charger_branch": make_charger(name="charger_branch", energy_output=65, cooldown=2),
        "carbon_branch": make_carbon_extractor(name="carbon_branch", output_per_cycle=3, cooldown=3, type_id=39),
        "oxygen_branch": make_oxygen_extractor(name="oxygen_branch", output_per_cycle=16, cooldown=3, type_id=40),
        "geranium_leaf": make_geranium_extractor(name="geranium_leaf", output_per_cycle=12, cooldown=100, type_id=41),
        "silicon_leaf": make_silicon_extractor(
            name="silicon_leaf", output_per_cycle=2, energy_cost=12, cooldown=3, type_id=42
        ),
    }

    _place(grid, x=center[0], y=center[1], value="assembler_branch")
    _place(grid, x=center[0] + 1, y=center[1], value="branch_chest")
    _place_many(
        grid,
        positions=[(center[0] - 1, center[1]), (center[0], center[1] - 1), (center[0], center[1] + 1)],
        value="charger_branch",
    )

    _place_many(
        grid,
        positions=[(x, y) for x, y in branch_points[1:3]],
        value="carbon_branch",
    )
    _place_many(grid, positions=[branch_points[3], branch_points[4]], value="oxygen_branch")
    _place_many(grid, positions=leaf_nodes[:2], value="geranium_leaf")
    _place_many(grid, positions=leaf_nodes[2:], value="silicon_leaf")

    spawn_offsets = [(-2, 0), (2, 0), (0, -2), (0, 2)]
    for ox, oy in spawn_offsets:
        _place(grid, x=center[0] + ox, y=center[1] + oy, value="agent.agent")

    return _make_game(grid=grid, objects=objects, num_cogs=4, max_steps=850)


def _machina_bottleneck() -> MettaGridConfig:
    params = ScenarioParams(width=48, height=48)
    grid = _new_grid(params)
    _carve_border(grid)

    base = (params.width // 2, params.height // 2)
    _carve_rect(grid, x0=base[0] - 5, y0=base[1] - 5, width=10, height=10)

    # One-width corridors leading out of the base.
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        for step in range(1, 16):
            _place(grid, x=base[0] + dx * step, y=base[1] + dy * step, value="empty")

    # Wider rooms at the ends.
    end_points = [
        (base[0] - 15, base[1]),
        (base[0] + 15, base[1]),
        (base[0], base[1] - 15),
        (base[0], base[1] + 15),
    ]
    for ex, ey in end_points:
        _carve_rect(grid, x0=ex - 4, y0=ey - 4, width=8, height=8)

    objects = {
        "wall": WallConfig(type_id=1),
        "assembler_lock": make_core_assembler(name="assembler_lock"),
        "lock_chest": make_chest(deposit_positions=("E", "W"), withdrawal_positions=("N", "S")),
        "charger_gate": make_charger(name="charger_gate", energy_output=55, cooldown=1),
        "carbon_gate": make_carbon_extractor(name="carbon_gate", output_per_cycle=3, cooldown=2, type_id=43),
        "oxygen_gate": make_oxygen_extractor(name="oxygen_gate", output_per_cycle=12, cooldown=2, type_id=44),
        "geranium_gate": make_geranium_extractor(name="geranium_gate", output_per_cycle=9, cooldown=110, type_id=45),
        "silicon_gate": make_silicon_extractor(
            name="silicon_gate", output_per_cycle=2, energy_cost=10, cooldown=2, type_id=46
        ),
    }

    _place(grid, x=base[0], y=base[1], value="assembler_lock")
    _place(grid, x=base[0] + 1, y=base[1], value="lock_chest")
    _place_many(
        grid,
        positions=[(base[0] - 1, base[1]), (base[0], base[1] - 1), (base[0], base[1] + 1)],
        value="charger_gate",
    )

    _place_many(grid, positions=end_points, value="carbon_gate")
    _place_many(grid, positions=[(ex, ey + 3) for ex, ey in end_points], value="oxygen_gate")
    _place_many(grid, positions=[(ex, ey - 3) for ex, ey in end_points], value="geranium_gate")
    _place_many(grid, positions=[(ex + 2, ey) for ex, ey in end_points], value="silicon_gate")

    spawn_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for ox, oy in spawn_offsets:
        _place(grid, x=base[0] + ox, y=base[1] + oy, value="agent.agent")

    return _make_game(grid=grid, objects=objects, num_cogs=4, max_steps=700)


def _machina_quarters() -> MettaGridConfig:
    params = ScenarioParams(width=24, height=24)
    grid = _new_grid(params)
    _carve_border(grid)

    # Extremely tight base; only a 6x6 manoeuvring area.
    _carve_rect(grid, x0=9, y0=9, width=6, height=6)
    _carve_rect(grid, x0=10, y0=10, width=4, height=4)

    # Resources squeezed around the outside.
    objects = {
        "wall": WallConfig(type_id=1),
        "assembler_core": make_core_assembler(name="assembler_core"),
        "micro_chest": make_chest(deposit_positions=("E",), withdrawal_positions=("W",)),
        "charger_micro": make_charger(name="charger_micro", energy_output=50, cooldown=1),
        "carbon_micro": make_carbon_extractor(name="carbon_micro", output_per_cycle=2, cooldown=2),
        "oxygen_micro": make_oxygen_extractor(name="oxygen_micro", output_per_cycle=10, cooldown=2),
        "geranium_micro": make_geranium_extractor(name="geranium_micro", output_per_cycle=6, cooldown=120),
        "silicon_micro": make_silicon_extractor(name="silicon_micro", output_per_cycle=1, energy_cost=8, cooldown=2),
    }

    _place(grid, x=12, y=12, value="assembler_core")
    _place(grid, x=13, y=12, value="micro_chest")
    _place_many(
        grid,
        positions=[(11, 12), (12, 11), (12, 13)],
        value="charger_micro",
    )

    resource_ring = [
        (8, 12),
        (16, 12),
        (12, 8),
        (12, 16),
        (8, 10),
        (16, 10),
        (8, 14),
        (16, 14),
    ]
    _place_many(grid, positions=resource_ring[:4], value="carbon_micro")
    _place_many(grid, positions=resource_ring[4:], value="oxygen_micro")
    _place_many(grid, positions=[(7, 7), (17, 17)], value="geranium_micro")
    _place_many(grid, positions=[(7, 17), (17, 7)], value="silicon_micro")

    spawn_offsets = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
    for ox, oy in spawn_offsets:
        _place(grid, x=12 + ox, y=12 + oy, value="agent.agent")

    return _make_game(grid=grid, objects=objects, num_cogs=4, max_steps=500)


def _machina_open_fields() -> MettaGridConfig:
    params = ScenarioParams(width=56, height=48, fill="empty")
    grid = _new_grid(params)
    _carve_border(grid, thickness=2)

    base_center = (params.width // 2, params.height // 2)
    _carve_rect(grid, x0=base_center[0] - 4, y0=base_center[1] - 4, width=8, height=8)

    objects = {
        "wall": WallConfig(type_id=1),
        "assembler_field": make_core_assembler(name="assembler_field", cooldown=2),
        "field_chest": make_chest(deposit_positions=("N", "S"), withdrawal_positions=("E", "W")),
        "charger_field": make_charger(name="charger_field", energy_output=65, cooldown=1),
        "carbon_field": make_carbon_extractor(name="carbon_field", output_per_cycle=4, cooldown=2, type_id=47),
        "oxygen_field": make_oxygen_extractor(name="oxygen_field", output_per_cycle=18, cooldown=2, type_id=48),
        "silicon_field": make_silicon_extractor(
            name="silicon_field", output_per_cycle=3, energy_cost=10, cooldown=3, type_id=49
        ),
        "geranium_rare": make_geranium_extractor(
            name="geranium_rare", output_per_cycle=15, cooldown=160, type_id=50, tags=("rare",)
        ),
    }

    cx, cy = base_center
    _place(grid, x=cx, y=cy, value="assembler_field")
    _place(grid, x=cx + 1, y=cy, value="field_chest")
    _place_many(
        grid,
        positions=[(cx - 1, cy), (cx, cy - 1), (cx, cy + 1)],
        value="charger_field",
    )

    # Scatter abundant carbon/oxygen near the center.
    carbon_positions = [
        (cx + dx, cy + dy) for dx in range(-12, 13, 4) for dy in range(-8, 9, 4) if abs(dx) + abs(dy) <= 12
    ]
    _place_many(grid, positions=carbon_positions, value="carbon_field")

    oxygen_positions = [(cx + dx, cy + dy) for dx in range(-14, 15, 6) for dy in range(-10, 11, 6)]
    _place_many(grid, positions=oxygen_positions, value="oxygen_field")

    silicon_positions = [(cx + dx, cy + dy) for dx in range(-18, 19, 9) for dy in (-15, 0, 15)]
    _place_many(grid, positions=silicon_positions, value="silicon_field")

    # Rare germanium nodes at the far corners with heavy cooldowns.
    _place_many(
        grid,
        positions=[(3, 3), (params.width - 4, 3), (3, params.height - 4), (params.width - 4, params.height - 4)],
        value="geranium_rare",
    )

    spawn_offsets = [(-2, 0), (2, 0), (0, -2), (0, 2)]
    for ox, oy in spawn_offsets:
        _place(grid, x=cx + ox, y=cy + oy, value="agent.agent")

    return _make_game(
        grid=grid,
        objects=objects,
        num_cogs=4,
        max_steps=700,
        allow_item_exchange=True,
        default_limit=18,
    )


def _machina_duality() -> MettaGridConfig:
    params = ScenarioParams(width=68, height=42)
    grid = _new_grid(params)
    _carve_border(grid)

    mid_x = params.width // 2
    _carve_rect(grid, x0=mid_x - 6, y0=params.height // 2 - 6, width=12, height=12)

    # Left half: open fields with low-yield resources.
    _carve_rect(grid, x0=2, y0=2, width=mid_x - 8, height=params.height - 4, fill="empty")

    # Right half: layered maze with deeper corridors.
    for column in range(mid_x + 4, params.width - 2, 4):
        _carve_corridor(grid, start=(column, 2), end=(column, params.height - 3), width=2)
    for row in range(4, params.height - 4, 6):
        _carve_corridor(grid, start=(mid_x + 4, row), end=(params.width - 3, row), width=2)

    objects = {
        "wall": WallConfig(type_id=1),
        "assembler_dual": make_core_assembler(name="assembler_dual", cooldown=2),
        "duality_chest": make_chest(deposit_positions=("N", "S"), withdrawal_positions=("E", "W")),
        "charger_dual": make_charger(name="charger_dual", energy_output=70, cooldown=1),
        "carbon_easy": make_carbon_extractor(name="carbon_easy", output_per_cycle=3, cooldown=2, type_id=51),
        "carbon_maze": make_carbon_extractor(
            name="carbon_maze", output_per_cycle=5, cooldown=3, energy_cost=2, type_id=52
        ),
        "oxygen_easy": make_oxygen_extractor(name="oxygen_easy", output_per_cycle=15, cooldown=2, type_id=53),
        "oxygen_maze": make_oxygen_extractor(name="oxygen_maze", output_per_cycle=20, cooldown=3, type_id=54),
        "silicon_easy": make_silicon_extractor(
            name="silicon_easy", output_per_cycle=2, energy_cost=8, cooldown=2, type_id=55
        ),
        "silicon_maze": make_silicon_extractor(
            name="silicon_maze", output_per_cycle=3, energy_cost=12, cooldown=3, type_id=56
        ),
        "geranium_maze": make_geranium_extractor(name="geranium_maze", output_per_cycle=16, cooldown=140),
    }

    cx, cy = mid_x, params.height // 2
    _place(grid, x=cx, y=cy, value="assembler_dual")
    _place(grid, x=cx + 1, y=cy, value="duality_chest")
    _place_many(
        grid,
        positions=[(cx - 1, cy), (cx, cy - 1), (cx, cy + 1)],
        value="charger_dual",
    )

    easy_positions = [(4 + dx, 6 + dy) for dx in range(0, mid_x - 12, 6) for dy in range(0, params.height - 12, 6)]
    _place_many(grid, positions=easy_positions[::2], value="carbon_easy")
    _place_many(grid, positions=easy_positions[1::2], value="oxygen_easy")
    _place_many(grid, positions=[pos for pos in easy_positions if (pos[0] + pos[1]) % 3 == 0], value="silicon_easy")

    maze_positions = [
        (column, row) for column in range(mid_x + 4, params.width - 3, 4) for row in range(4, params.height - 4, 6)
    ]
    _place_many(grid, positions=maze_positions[::3], value="carbon_maze")
    _place_many(grid, positions=maze_positions[1::3], value="oxygen_maze")
    _place_many(grid, positions=maze_positions[2::3], value="silicon_maze")
    _place_many(
        grid,
        positions=[(params.width - 5, params.height // 3), (params.width - 5, 2 * params.height // 3)],
        value="geranium_maze",
    )

    spawn_offsets = [(-2, -2), (-2, 2), (2, -2), (2, 2)]
    for ox, oy in spawn_offsets:
        _place(grid, x=cx + ox, y=cy + oy, value="agent.agent")

    return _make_game(grid=grid, objects=objects, num_cogs=4, max_steps=780)


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------


SCENARIO_BUILDERS: Final[dict[str, Callable[[], MettaGridConfig]]] = {
    # Tutorials
    "assembler_1_simple": lambda: _tutorial_map(num_cogs=1, richer_layout=False),
    "assembler_1_complex": lambda: _tutorial_map(num_cogs=1, richer_layout=True),
    "assembler_2_simple": lambda: _tutorial_map(num_cogs=4, richer_layout=False),
    "assembler_2_complex": lambda: _tutorial_map(num_cogs=4, richer_layout=True),
    # Machina collection
    "machina_1": _machina_small_drop,
    "machina_2": _machina_frontier,
    "machina_maze": _machina_maze,
    "machina_branching": _machina_branching,
    "machina_bottleneck": _machina_bottleneck,
    "machina_quarters": _machina_quarters,
    "machina_open_fields": _machina_open_fields,
    "machina_duality": _machina_duality,
}


def make_game(num_cogs: int = 4, **legacy_kwargs) -> MettaGridConfig:
    """Legacy helper retained for backwards compatibility.

    Historically this factory accepted a long list of object counts. We now rely on curated scenarios,
    so those keyword arguments are ignored for the moment (recorded here so we remember to surface richer
    hooks later). Users can select explicit layouts via the ``scenario`` keyword.
    """

    scenario_name = legacy_kwargs.pop("scenario", None)
    if legacy_kwargs:
        # TODO: expose knobs that let us mutate handcrafted layouts without rebuilding the grid.
        unused = ", ".join(sorted(legacy_kwargs.keys()))
        raise ValueError(f"Legacy parameters ({unused}) are no longer supported. Pass scenario='machina_2' etc.")

    if scenario_name is None:
        scenario_name = "machina_1" if num_cogs <= 2 else "machina_2"

    if scenario_name not in SCENARIO_BUILDERS:
        raise ValueError(f"Unknown scenario '{scenario_name}'")
    return SCENARIO_BUILDERS[scenario_name]()


def games() -> dict[str, MettaGridConfig]:
    return {name: builder() for name, builder in SCENARIO_BUILDERS.items()}
