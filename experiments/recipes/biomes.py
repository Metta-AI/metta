from typing import List

import numpy as np
from cogames.cogs_vs_clips import stations as cvc_stations
from metta.sim.simulation_config import SimulationConfig
from mettagrid import MettaGridConfig
from mettagrid.builder.envs import make_navigation
from mettagrid.config.config import Config
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.random.int import IntConstantDistribution
from mettagrid.mapgen.scene import ChildrenAction, Scene
from mettagrid.mapgen.scenes.base_hub import BaseHub, BaseHubParams
from mettagrid.mapgen.scenes.biome_caves import BiomeCaves, BiomeCavesParams
from mettagrid.mapgen.scenes.biome_city import BiomeCity, BiomeCityParams
from mettagrid.mapgen.scenes.biome_desert import BiomeDesert, BiomeDesertParams
from mettagrid.mapgen.scenes.biome_forest import BiomeForest, BiomeForestParams
from mettagrid.mapgen.scenes.bsp import BSP, BSPLayout, BSPLayoutParams, BSPParams
from mettagrid.mapgen.scenes.fill_area import FillArea, FillAreaParams
from mettagrid.mapgen.scenes.layout import Layout, LayoutArea, LayoutParams
from mettagrid.mapgen.scenes.make_connected import MakeConnected, MakeConnectedParams
from mettagrid.mapgen.scenes.maze import Maze, MazeParams
from mettagrid.mapgen.scenes.quadrants import Quadrants, QuadrantsParams
from mettagrid.mapgen.scenes.radial_maze import RadialMaze, RadialMazeParams
from mettagrid.mapgen.types import AreaWhere
from pydantic import Field


def areas_overlap(
    area1: tuple[int, int, int, int], area2: tuple[int, int, int, int]
) -> bool:
    """Check if two areas overlap. Areas are (x, y, width, height)."""
    x1, y1, w1, h1 = area1
    x2, y2, w2, h2 = area2

    # Check if rectangles overlap
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)


def validate_placement(scene, areas_to_check: list[tuple[int, int, int, int]]) -> bool:
    """Validate that areas don't overlap with existing areas."""
    for existing_area in areas_to_check:
        if areas_overlap(existing_area, (scene.x, scene.y, scene.width, scene.height)):
            return False
    return True


class CollisionSafeLayout(Scene[LayoutParams]):
    """Layout scene with collision detection and retry logic."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.placed_areas = []

    def render(self):
        for area in self.params.areas:
            if area.width > self.width or area.height > self.height:
                raise ValueError(
                    f"Area {area} is too large for grid {self.width}x{self.height}"
                )

            # Try different placements until we find a valid one
            max_attempts = 10
            for attempt in range(max_attempts):
                if area.placement == "center":
                    x = (self.width - area.width) // 2
                    y = (self.height - area.height) // 2

                    # Check if this placement would overlap with existing areas
                    if validate_placement(
                        type(
                            "MockScene",
                            (),
                            {
                                "x": x,
                                "y": y,
                                "width": area.width,
                                "height": area.height,
                            },
                        )(),
                        self.placed_areas,
                    ):
                        self.make_area(x, y, area.width, area.height, tags=[area.tag])
                        self.placed_areas.append((x, y, area.width, area.height))
                        break
                    else:
                        # Try random offset placement as fallback
                        offset_x = self.rng.integers(
                            -min(50, self.width - area.width),
                            min(50, self.width - area.width) + 1,
                        )
                        offset_y = self.rng.integers(
                            -min(50, self.height - area.height),
                            min(50, self.height - area.height) + 1,
                        )
                        x = max(
                            0,
                            min(
                                self.width - area.width,
                                (self.width - area.width) // 2 + offset_x,
                            ),
                        )
                        y = max(
                            0,
                            min(
                                self.height - area.height,
                                (self.height - area.height) // 2 + offset_y,
                            ),
                        )

                        if validate_placement(
                            type(
                                "MockScene",
                                (),
                                {
                                    "x": x,
                                    "y": y,
                                    "width": area.width,
                                    "height": area.height,
                                },
                            )(),
                            self.placed_areas,
                        ):
                            self.make_area(
                                x, y, area.width, area.height, tags=[area.tag]
                            )
                            self.placed_areas.append((x, y, area.width, area.height))
                            break
                else:
                    raise ValueError(f"Unknown placement: {area.placement}")
            else:
                # If we couldn't place after max attempts, raise an error
                raise ValueError(
                    f"Could not place area {area} after {max_attempts} attempts due to collisions"
                )


def _add_extractor_objects(env: MettaGridConfig) -> None:
    objects = env.game.objects
    objects.setdefault("charger", cvc_stations.charger())
    objects.setdefault("carbon_extractor", cvc_stations.carbon_extractor())
    objects.setdefault("oxygen_extractor", cvc_stations.oxygen_extractor())
    objects.setdefault("germanium_extractor", cvc_stations.germanium_extractor())
    objects.setdefault("silicon_extractor", cvc_stations.silicon_extractor())
    objects.setdefault("chest", cvc_stations.chest())
    # depleted variants
    objects.setdefault("carbon_ex_dep", cvc_stations.carbon_ex_dep())
    objects.setdefault("oxygen_ex_dep", cvc_stations.oxygen_ex_dep())
    objects.setdefault("germanium_ex_dep", cvc_stations.germanium_ex_dep())
    objects.setdefault("silicon_ex_dep", cvc_stations.silicon_ex_dep())


def _linspace_positions(count: int, interior_size: int) -> list[int]:
    if count <= 0:
        return []
    if interior_size <= 0:
        raise ValueError("interior_size must be positive")

    if count >= interior_size:
        return [i for i in range(1, interior_size + 1)]

    step = (interior_size + 1) / (count + 1)
    return [
        1 + max(0, min(interior_size - 1, round(step * (i + 1)))) for i in range(count)
    ]


class UniformExtractorParams(Config):
    rows: int = 4
    cols: int = 4
    jitter: int = 1
    padding: int = 1
    clear_existing: bool = False
    frame_with_walls: bool = False
    target_coverage: float | None = None
    extractor_names: list[str] = Field(
        default_factory=lambda: [
            "carbon_extractor",
            "oxygen_extractor",
            "germanium_extractor",
            "silicon_extractor",
            "charger",
        ]
    )


class UniformExtractorScene(Scene[UniformExtractorParams]):
    """Place extractor stations on a jittered uniform grid."""

    def render(self):
        params = self.params
        if self.width < 3 or self.height < 3:
            raise ValueError("Extractor map must be at least 3x3 to fit border walls")

        padding = max(0, params.padding)
        row_min = padding
        row_max = self.height - padding - 1
        col_min = padding
        col_max = self.width - padding - 1

        if row_min > row_max or col_min > col_max:
            return

        if params.clear_existing:
            # Start from an empty canvas when requested (used for dedicated showcase maps).
            self.grid[:, :] = "empty"
            if params.frame_with_walls:
                self.grid[0, :] = "wall"
                self.grid[-1, :] = "wall"
                self.grid[:, 0] = "wall"
                self.grid[:, -1] = "wall"

        interior_width = self.width - 2
        interior_height = self.height - 2

        spacing = padding + 1

        def carve_and_place(center_row: int, center_col: int, name: str) -> None:
            for rr in range(center_row - padding, center_row + padding + 1):
                if rr < 0 or rr >= self.height:
                    continue
                for cc in range(center_col - padding, center_col + padding + 1):
                    if cc < 0 or cc >= self.width:
                        continue
                    if rr == center_row and cc == center_col:
                        self.grid[rr, cc] = name
                    else:
                        self.grid[rr, cc] = "empty"

        def can_place(
            center_row: int, center_col: int, centers: list[tuple[int, int]]
        ) -> bool:
            return not any(
                abs(center_row - r0) <= padding and abs(center_col - c0) <= padding
                for r0, c0 in centers
            )

        extractor_names = params.extractor_names or ["carbon_extractor"]

        if params.target_coverage is not None:
            available_height = row_max - row_min + 1
            available_width = col_max - col_min + 1
            if available_height <= 0 or available_width <= 0:
                return

            max_rows = max(0, (available_height + spacing - 1) // spacing)
            max_cols = max(0, (available_width + spacing - 1) // spacing)
            max_possible = max_rows * max_cols
            if max_possible == 0:
                return

            desired = int(params.target_coverage * interior_width * interior_height)
            placement_goal = min(max_possible, max(1, desired))

            valid_row_starts = [
                row_min + offset
                for offset in range(spacing)
                if row_min + offset <= row_max
            ]
            valid_col_starts = [
                col_min + offset
                for offset in range(spacing)
                if col_min + offset <= col_max
            ]
            if not valid_row_starts or not valid_col_starts:
                return

            start_row = int(self.rng.choice(valid_row_starts))
            start_col = int(self.rng.choice(valid_col_starts))

            rows = list(range(start_row, row_max + 1, spacing))
            cols = list(range(start_col, col_max + 1, spacing))
            positions = [(r, c) for r in rows for c in cols]
            if not positions:
                return

            positions = positions[:max_possible]
            permutation = self.rng.permutation(len(positions))
            positions = [positions[i] for i in permutation]
            positions = positions[:placement_goal]

            assignments = [
                extractor_names[i % len(extractor_names)] for i in range(len(positions))
            ]
            assignment_perm = self.rng.permutation(len(assignments))
            assignments = [assignments[i] for i in assignment_perm]

            placed_centers_tc: list[tuple[int, int]] = []
            for (row, col), name in zip(positions, assignments):
                if not can_place(row, col, placed_centers_tc):
                    continue
                carve_and_place(row, col, name)
                placed_centers_tc.append((row, col))
            return

        row_positions = _linspace_positions(params.rows, interior_height)
        col_positions = _linspace_positions(params.cols, interior_width)

        if not row_positions or not col_positions:
            raise ValueError("rows and cols must be positive for extractor placement")

        # Deduplicate potential overlaps caused by rounding.
        raw_positions = [(row, col) for row in row_positions for col in col_positions]
        positions: list[tuple[int, int]] = []
        seen = set()
        for row, col in raw_positions:
            if (row, col) not in seen:
                seen.add((row, col))
                positions.append((row, col))

        if not positions:
            return

        extractor_names = params.extractor_names or ["carbon_extractor"]
        assignments = [
            extractor_names[i % len(extractor_names)] for i in range(len(positions))
        ]
        # Shuffle assignments so a new seed changes extractor distribution.
        self.rng.shuffle(assignments)

        jitter = max(0, params.jitter)
        placed_centers: list[tuple[int, int]] = []
        for (base_row, base_col), name in zip(positions, assignments):
            row = int(min(row_max, max(row_min, base_row)))
            col = int(min(col_max, max(col_min, base_col)))
            attempts = max(1, 8 if jitter else 1)
            placement: tuple[int, int] | None = None
            for _ in range(attempts):
                offset_row = int(
                    np.clip(
                        row + (self.rng.integers(-jitter, jitter + 1) if jitter else 0),
                        row_min,
                        row_max,
                    )
                )
                offset_col = int(
                    np.clip(
                        col + (self.rng.integers(-jitter, jitter + 1) if jitter else 0),
                        col_min,
                        col_max,
                    )
                )
                if not (
                    row_min <= offset_row <= row_max
                    and col_min <= offset_col <= col_max
                ):
                    continue
                if not can_place(offset_row, offset_col, placed_centers):
                    continue
                placement = (offset_row, offset_col)
                break
            if placement is None:
                continue
            row, col = placement
            carve_and_place(row, col, name)
            placed_centers.append((row, col))


def make_extractor_showcase() -> MettaGridConfig:
    env = make_navigation(num_agents=4)
    _add_extractor_objects(env)

    resources = set(env.game.resource_names)
    resources.update({"energy", "carbon", "oxygen", "germanium", "silicon", "energy"})
    env.game.resource_names = sorted(resources)

    env.game.map_builder = MapGen.Config(
        width=25,
        height=25,
        root=UniformExtractorScene.factory(
            UniformExtractorParams(
                rows=4,
                cols=4,
                jitter=1,
                clear_existing=True,
                frame_with_walls=True,
                extractor_names=[
                    "carbon_extractor",
                    "oxygen_extractor",
                    "germanium_extractor",
                    "silicon_extractor",
                    "charger",
                ],
            )
        ),
    )

    return env


def make_basehub_showcase() -> MettaGridConfig:
    env = make_navigation(num_agents=4)
    _add_extractor_objects(env)

    env.game.map_builder = MapGen.Config(
        width=21,
        height=21,
        root=Layout.factory(
            LayoutParams(
                areas=[
                    LayoutArea(
                        width=21,
                        height=21,
                        placement="center",
                        tag="sanctum.tight",
                    )
                ]
            ),
            children_actions=[
                ChildrenAction(
                    scene=FillArea.factory(FillAreaParams(value="empty")),
                    where=AreaWhere(tags=["sanctum.tight"]),
                ),
                ChildrenAction(
                    scene=BaseHub.factory(
                        BaseHubParams(
                            include_inner_wall=False,
                            layout="tight",
                            corner_objects=[
                                "carbon_ex_dep",
                                "oxygen_ex_dep",
                                "germanium_ex_dep",
                                "silicon_ex_dep",
                            ],
                        )
                    ),
                    where=AreaWhere(tags=["sanctum.tight"]),
                    limit=1,
                    order_by="last",
                ),
            ],
        ),
    )

    return env


def make_mettagrid(
    width: int = 500, height: int = 500
) -> tuple[
    MettaGridConfig,
    MettaGridConfig,
    MettaGridConfig,
    MettaGridConfig,
    MettaGridConfig,
    MettaGridConfig,
    MettaGridConfig,
    MettaGridConfig,
    MettaGridConfig,
    MettaGridConfig,
    MettaGridConfig,
]:
    desert_noise = make_navigation(num_agents=4)
    desert_noise.game.map_builder = MapGen.Config(
        width=width,
        height=height,
        root=BiomeDesert.factory(
            BiomeDesertParams(dune_period=8, ridge_width=2, angle=0.4, noise_prob=0.6)
        ),
    )

    city = make_navigation(num_agents=4)
    city.game.map_builder = MapGen.Config(
        width=width,
        height=height,
        root=BiomeCity.factory(
            BiomeCityParams(pitch=10, road_width=2, jitter=2, place_prob=0.9)
        ),
    )

    forest = make_navigation(num_agents=4)
    forest.game.map_builder = MapGen.Config(
        width=width,
        height=height,
        root=BiomeForest.factory(
            BiomeForestParams(clumpiness=4, seed_prob=0.05, growth_prob=0.6)
        ),
    )

    caves = make_navigation(num_agents=4)
    caves.game.map_builder = MapGen.Config(
        width=width,
        height=height,
        root=BiomeCaves.factory(
            BiomeCavesParams(fill_prob=0.45, steps=4, birth_limit=5, death_limit=3)
        ),
    )

    astroid = make_navigation(num_agents=4)
    _add_extractor_objects(astroid)
    astroid_resources = set(astroid.game.resource_names)
    astroid_resources.update({"energy", "carbon", "oxygen", "germanium", "silicon"})
    astroid.game.resource_names = sorted(astroid_resources)
    astroid.game.map_builder = MapGen.Config(
        width=width,
        height=height,
        root=BiomeCaves.factory(
            BiomeCavesParams(
                fill_prob=0.4,
                steps=5,
                birth_limit=5,
                death_limit=3,
            ),
            children_actions=[
                # Sanctum outpost: orderly ring around central hub
                ChildrenAction(
                    scene=Layout.factory(
                        LayoutParams(
                            areas=[
                                LayoutArea(
                                    width=120,
                                    height=120,
                                    placement="center",
                                    tag="sanctum.outpost",
                                )
                            ]
                        ),
                        children_actions=[
                            ChildrenAction(
                                scene=FillArea.factory(FillAreaParams(value="empty")),
                                where=AreaWhere(tags=["sanctum.outpost"]),
                            ),
                            ChildrenAction(
                                scene=BiomeCity.factory(
                                    BiomeCityParams(
                                        pitch=12,
                                        road_width=3,
                                        jitter=1,
                                        place_prob=0.6,
                                    )
                                ),
                                where=AreaWhere(tags=["sanctum.outpost"]),
                            ),
                            ChildrenAction(
                                scene=BiomeCaves.factory(
                                    BiomeCavesParams(
                                        fill_prob=0.25,
                                        steps=2,
                                        birth_limit=4,
                                        death_limit=3,
                                    )
                                ),
                                where=AreaWhere(tags=["sanctum.outpost"]),
                            ),
                        ],
                    ),
                    where="full",
                    order_by="first",
                    lock="sanctum.outpost",
                    limit=1,
                ),
                # Stencil a constellation of pockets across the caves using BSP zones.
                ChildrenAction(
                    scene=BSPLayout.factory(
                        BSPLayoutParams(area_count=12),
                        children_actions=[
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=90,
                                                height=90,
                                                placement="center",
                                                tag="astroid.desert",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(tags=["astroid.desert"]),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeDesert.factory(
                                                BiomeDesertParams(
                                                    dune_period=8,
                                                    ridge_width=2,
                                                    angle=0.4,
                                                    noise_prob=0.6,
                                                )
                                            ),
                                            where=AreaWhere(tags=["astroid.desert"]),
                                        ),
                                        ChildrenAction(
                                            scene=Layout.factory(
                                                LayoutParams(
                                                    areas=[
                                                        LayoutArea(
                                                            width=30,
                                                            height=30,
                                                            placement="center",
                                                            tag="astroid.radial",
                                                        )
                                                    ]
                                                ),
                                                children_actions=[
                                                    ChildrenAction(
                                                        scene=FillArea.factory(
                                                            FillAreaParams(
                                                                value="empty"
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=["astroid.radial"]
                                                        ),
                                                    ),
                                                    ChildrenAction(
                                                        scene=RadialMaze.factory(
                                                            RadialMazeParams(
                                                                arms=10,
                                                                arm_width=2,
                                                                arm_length=28,
                                                                fill_background=False,
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=["astroid.radial"]
                                                        ),
                                                        limit=1,
                                                    ),
                                                ],
                                            ),
                                            where=AreaWhere(tags=["astroid.desert"]),
                                            limit=1,
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                                lock="astroid.zone",
                            ),
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=60,
                                                height=60,
                                                placement="center",
                                                tag="astroid.forest",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(tags=["astroid.forest"]),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeForest.factory(
                                                BiomeForestParams(
                                                    clumpiness=4,
                                                    seed_prob=0.05,
                                                    growth_prob=0.6,
                                                )
                                            ),
                                            where=AreaWhere(tags=["astroid.forest"]),
                                        ),
                                        ChildrenAction(
                                            scene=Layout.factory(
                                                LayoutParams(
                                                    areas=[
                                                        LayoutArea(
                                                            width=24,
                                                            height=24,
                                                            placement="center",
                                                            tag="astroid.bsp",
                                                        )
                                                    ]
                                                ),
                                                children_actions=[
                                                    ChildrenAction(
                                                        scene=FillArea.factory(
                                                            FillAreaParams(
                                                                value="empty"
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=["astroid.bsp"]
                                                        ),
                                                    ),
                                                    ChildrenAction(
                                                        scene=BSP.factory(
                                                            BSPParams(
                                                                rooms=8,
                                                                min_room_size=3,
                                                                min_room_size_ratio=0.4,
                                                                max_room_size_ratio=0.8,
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=["astroid.bsp"]
                                                        ),
                                                        limit=1,
                                                    ),
                                                ],
                                            ),
                                            where=AreaWhere(tags=["astroid.forest"]),
                                            limit=1,
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                                lock="astroid.zone",
                            ),
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=60,
                                                height=60,
                                                placement="center",
                                                tag="astroid.city.maze1",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.city.maze1"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeCity.factory(
                                                BiomeCityParams(
                                                    pitch=10,
                                                    road_width=2,
                                                    jitter=2,
                                                    place_prob=0.9,
                                                )
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.city.maze1"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=Layout.factory(
                                                LayoutParams(
                                                    areas=[
                                                        LayoutArea(
                                                            width=30,
                                                            height=30,
                                                            placement="center",
                                                            tag="astroid.maze",
                                                        )
                                                    ]
                                                ),
                                                children_actions=[
                                                    ChildrenAction(
                                                        scene=FillArea.factory(
                                                            FillAreaParams(
                                                                value="empty"
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=["astroid.maze"]
                                                        ),
                                                    ),
                                                    ChildrenAction(
                                                        scene=Maze.factory(
                                                            MazeParams(
                                                                algorithm="dfs",
                                                                room_size=IntConstantDistribution(
                                                                    value=2
                                                                ),
                                                                wall_size=IntConstantDistribution(
                                                                    value=1
                                                                ),
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=["astroid.maze"]
                                                        ),
                                                        limit=1,
                                                    ),
                                                ],
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.city.maze1"]
                                            ),
                                            limit=1,
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                                lock="astroid.zone",
                            ),
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=72,
                                                height=72,
                                                placement="center",
                                                tag="astroid.city.maze2",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.city.maze2"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeCity.factory(
                                                BiomeCityParams(
                                                    pitch=10,
                                                    road_width=2,
                                                    jitter=2,
                                                    place_prob=0.9,
                                                )
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.city.maze2"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=Layout.factory(
                                                LayoutParams(
                                                    areas=[
                                                        LayoutArea(
                                                            width=56,
                                                            height=56,
                                                            placement="center",
                                                            tag="astroid.maze.kruskal",
                                                        )
                                                    ]
                                                ),
                                                children_actions=[
                                                    ChildrenAction(
                                                        scene=FillArea.factory(
                                                            FillAreaParams(
                                                                value="empty"
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=[
                                                                "astroid.maze.kruskal"
                                                            ]
                                                        ),
                                                    ),
                                                    ChildrenAction(
                                                        scene=Maze.factory(
                                                            MazeParams(
                                                                algorithm="kruskal",
                                                                room_size=IntConstantDistribution(
                                                                    value=4
                                                                ),
                                                                wall_size=IntConstantDistribution(
                                                                    value=2
                                                                ),
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=[
                                                                "astroid.maze.kruskal"
                                                            ]
                                                        ),
                                                        limit=1,
                                                    ),
                                                ],
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.city.maze2"]
                                            ),
                                            limit=1,
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                                lock="astroid.zone",
                            ),
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=160,
                                                height=100,
                                                placement="center",
                                                tag="astroid.bsp.crucible",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.bsp.crucible"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeCaves.factory(
                                                BiomeCavesParams(
                                                    fill_prob=0.35,
                                                    steps=3,
                                                    birth_limit=4,
                                                    death_limit=2,
                                                )
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.bsp.crucible"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=Layout.factory(
                                                LayoutParams(
                                                    areas=[
                                                        LayoutArea(
                                                            width=96,
                                                            height=72,
                                                            placement="center",
                                                            tag="astroid.bsp.crucible.core",
                                                        )
                                                    ]
                                                ),
                                                children_actions=[
                                                    ChildrenAction(
                                                        scene=FillArea.factory(
                                                            FillAreaParams(
                                                                value="empty"
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=[
                                                                "astroid.bsp.crucible.core"
                                                            ]
                                                        ),
                                                    ),
                                                    ChildrenAction(
                                                        scene=BSP.factory(
                                                            BSPParams(
                                                                rooms=40,
                                                                min_room_size=8,
                                                                min_room_size_ratio=0.5,
                                                                max_room_size_ratio=0.7,
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=[
                                                                "astroid.bsp.crucible.core"
                                                            ]
                                                        ),
                                                        limit=1,
                                                    ),
                                                ],
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.bsp.crucible"]
                                            ),
                                            limit=1,
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                                lock="astroid.zone",
                            ),
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=108,
                                                height=108,
                                                placement="center",
                                                tag="astroid.radial.alt",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.radial.alt"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeForest.factory(
                                                BiomeForestParams(
                                                    clumpiness=5,
                                                    seed_prob=0.08,
                                                    growth_prob=0.65,
                                                )
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.radial.alt"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=Layout.factory(
                                                LayoutParams(
                                                    areas=[
                                                        LayoutArea(
                                                            width=68,
                                                            height=68,
                                                            placement="center",
                                                            tag="astroid.radial.alt.core",
                                                        )
                                                    ]
                                                ),
                                                children_actions=[
                                                    ChildrenAction(
                                                        scene=FillArea.factory(
                                                            FillAreaParams(
                                                                value="empty"
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=[
                                                                "astroid.radial.alt.core"
                                                            ]
                                                        ),
                                                    ),
                                                    ChildrenAction(
                                                        scene=RadialMaze.factory(
                                                            RadialMazeParams(
                                                                arms=12,
                                                                arm_width=4,
                                                                arm_length=64,
                                                                fill_background=False,
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=[
                                                                "astroid.radial.alt.core"
                                                            ]
                                                        ),
                                                        limit=1,
                                                    ),
                                                ],
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.radial.alt"]
                                            ),
                                            limit=1,
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                                lock="astroid.zone",
                            ),
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=80,
                                                height=80,
                                                placement="center",
                                                tag="astroid.maze.prim",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(tags=["astroid.maze.prim"]),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeCity.factory(
                                                BiomeCityParams(
                                                    pitch=12,
                                                    road_width=3,
                                                    jitter=1,
                                                    place_prob=0.7,
                                                )
                                            ),
                                            where=AreaWhere(tags=["astroid.maze.prim"]),
                                        ),
                                        ChildrenAction(
                                            scene=Layout.factory(
                                                LayoutParams(
                                                    areas=[
                                                        LayoutArea(
                                                            width=56,
                                                            height=56,
                                                            placement="center",
                                                            tag="astroid.maze.prim.grid",
                                                        )
                                                    ]
                                                ),
                                                children_actions=[
                                                    ChildrenAction(
                                                        scene=FillArea.factory(
                                                            FillAreaParams(
                                                                value="empty"
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=[
                                                                "astroid.maze.prim.grid"
                                                            ]
                                                        ),
                                                    ),
                                                    ChildrenAction(
                                                        scene=Maze.factory(
                                                            MazeParams(
                                                                algorithm="dfs",
                                                                room_size=IntConstantDistribution(
                                                                    value=6
                                                                ),
                                                                wall_size=IntConstantDistribution(
                                                                    value=2
                                                                ),
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=[
                                                                "astroid.maze.prim.grid"
                                                            ]
                                                        ),
                                                        limit=1,
                                                    ),
                                                ],
                                            ),
                                            where=AreaWhere(tags=["astroid.maze.prim"]),
                                            limit=1,
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                                lock="astroid.zone",
                            ),
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=88,
                                                height=88,
                                                placement="center",
                                                tag="astroid.maze.unicursal",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.maze.unicursal"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeDesert.factory(
                                                BiomeDesertParams(
                                                    dune_period=6,
                                                    ridge_width=3,
                                                    angle=0.55,
                                                    noise_prob=0.45,
                                                )
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.maze.unicursal"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=Layout.factory(
                                                LayoutParams(
                                                    areas=[
                                                        LayoutArea(
                                                            width=64,
                                                            height=64,
                                                            placement="center",
                                                            tag="astroid.maze.unicursal.grid",
                                                        )
                                                    ]
                                                ),
                                                children_actions=[
                                                    ChildrenAction(
                                                        scene=FillArea.factory(
                                                            FillAreaParams(
                                                                value="empty"
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=[
                                                                "astroid.maze.unicursal.grid"
                                                            ]
                                                        ),
                                                    ),
                                                    ChildrenAction(
                                                        scene=Maze.factory(
                                                            MazeParams(
                                                                algorithm="kruskal",
                                                                room_size=IntConstantDistribution(
                                                                    value=4
                                                                ),
                                                                wall_size=IntConstantDistribution(
                                                                    value=4
                                                                ),
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=[
                                                                "astroid.maze.unicursal.grid"
                                                            ]
                                                        ),
                                                        limit=1,
                                                    ),
                                                ],
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.maze.unicursal"]
                                            ),
                                            limit=1,
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                                lock="astroid.zone",
                            ),
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=144,
                                                height=96,
                                                placement="center",
                                                tag="astroid.forest.glade",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.forest.glade"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeForest.factory(
                                                BiomeForestParams(
                                                    clumpiness=3,
                                                    seed_prob=0.12,
                                                    growth_prob=0.55,
                                                )
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.forest.glade"]
                                            ),
                                            limit=1,
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                                lock="astroid.zone",
                            ),
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=180,
                                                height=100,
                                                placement="center",
                                                tag="astroid.forest.mire",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.forest.mire"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeForest.factory(
                                                BiomeForestParams(
                                                    clumpiness=6,
                                                    seed_prob=0.04,
                                                    growth_prob=0.7,
                                                )
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.forest.mire"]
                                            ),
                                            limit=1,
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                                lock="astroid.zone",
                            ),
                        ],
                    ),
                    where="full",
                    limit=1,
                ),
                # Central sanctum retained as the cave nexus, stamped last.
                ChildrenAction(
                    scene=Layout.factory(
                        LayoutParams(
                            areas=[
                                LayoutArea(
                                    width=15,
                                    height=15,
                                    placement="center",
                                    tag="sanctum.center",
                                )
                            ]
                        ),
                        children_actions=[
                            ChildrenAction(
                                scene=BaseHub.factory(
                                    BaseHubParams(
                                        altar_object="altar",
                                        include_inner_wall=True,
                                        corner_objects=[
                                            "carbon_ex_dep",  # TL
                                            "oxygen_ex_dep",  # TR
                                            "germanium_ex_dep",  # BL
                                            "silicon_ex_dep",  # BR
                                        ],
                                    )
                                ),
                                where=AreaWhere(tags=["sanctum.center"]),
                                limit=1,
                            )
                        ],
                    ),
                    where="full",
                    order_by="last",
                    lock="sanctum",
                    limit=1,
                ),
                ChildrenAction(
                    scene=MakeConnected.factory(MakeConnectedParams()),
                    where="full",
                    order_by="last",
                    lock="connect",
                    limit=1,
                ),
                ChildrenAction(
                    scene=UniformExtractorScene.factory(
                        UniformExtractorParams(
                            target_coverage=0.05,
                            jitter=0,
                            extractor_names=[
                                "carbon_extractor",
                                "oxygen_extractor",
                                "germanium_extractor",
                                "silicon_extractor",
                                "charger",
                            ],
                        )
                    ),
                    where="full",
                    order_by="last",
                    lock="sanctum_extractors",
                    limit=1,
                ),
            ],
        ),
    )

    astroid_big = make_navigation(num_agents=4)
    _add_extractor_objects(astroid_big)
    big_resources = set(astroid_big.game.resource_names)
    big_resources.update({"energy", "carbon", "oxygen", "germanium", "silicon"})
    astroid_big.game.resource_names = sorted(big_resources)
    astroid_big.game.map_builder = MapGen.Config(
        width=1000,
        height=1000,
        root=BiomeCaves.factory(
            BiomeCavesParams(
                fill_prob=0.4,
                steps=5,
                birth_limit=5,
                death_limit=3,
            ),
            children_actions=[
                # Sanctum outpost: orderly ring around central hub
                ChildrenAction(
                    scene=Layout.factory(
                        LayoutParams(
                            areas=[
                                LayoutArea(
                                    width=120,
                                    height=120,
                                    placement="center",
                                    tag="sanctum.outpost",
                                )
                            ]
                        ),
                        children_actions=[
                            ChildrenAction(
                                scene=FillArea.factory(FillAreaParams(value="empty")),
                                where=AreaWhere(tags=["sanctum.outpost"]),
                            ),
                            ChildrenAction(
                                scene=BiomeCity.factory(
                                    BiomeCityParams(
                                        pitch=12,
                                        road_width=3,
                                        jitter=1,
                                        place_prob=0.6,
                                    )
                                ),
                                where=AreaWhere(tags=["sanctum.outpost"]),
                            ),
                            ChildrenAction(
                                scene=BiomeCaves.factory(
                                    BiomeCavesParams(
                                        fill_prob=0.25,
                                        steps=2,
                                        birth_limit=4,
                                        death_limit=3,
                                    )
                                ),
                                where=AreaWhere(tags=["sanctum.outpost"]),
                            ),
                        ],
                    ),
                    where="full",
                    order_by="first",
                    lock="sanctum.outpost",
                    limit=1,
                ),
                # Stencil a constellation of pockets across the caves using BSP zones.
                ChildrenAction(
                    scene=BSPLayout.factory(
                        BSPLayoutParams(area_count=30),
                        children_actions=[
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=180,
                                                height=180,
                                                placement="center",
                                                tag="astroid.desert",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(tags=["astroid.desert"]),
                                        ),
                                        ChildrenAction(
                                            scene=Layout.factory(
                                                LayoutParams(
                                                    areas=[
                                                        LayoutArea(
                                                            width=100,
                                                            height=100,
                                                            placement="center",
                                                            tag="astroid.desert.oasis",
                                                        )
                                                    ]
                                                ),
                                                children_actions=[
                                                    ChildrenAction(
                                                        scene=FillArea.factory(
                                                            FillAreaParams(
                                                                value="empty"
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=[
                                                                "astroid.desert.oasis"
                                                            ]
                                                        ),
                                                    ),
                                                    ChildrenAction(
                                                        scene=BiomeDesert.factory(
                                                            BiomeDesertParams(
                                                                dune_period=12,
                                                                ridge_width=4,
                                                                angle=0.3,
                                                                noise_prob=0.3,
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=[
                                                                "astroid.desert.oasis"
                                                            ]
                                                        ),
                                                    ),
                                                    ChildrenAction(
                                                        scene=Layout.factory(
                                                            LayoutParams(
                                                                areas=[
                                                                    LayoutArea(
                                                                        width=50,
                                                                        height=50,
                                                                        placement="center",
                                                                        tag="astroid.desert.radial",
                                                                    )
                                                                ]
                                                            ),
                                                            children_actions=[
                                                                ChildrenAction(
                                                                    scene=FillArea.factory(
                                                                        FillAreaParams(
                                                                            value="empty"
                                                                        )
                                                                    ),
                                                                    where=AreaWhere(
                                                                        tags=[
                                                                            "astroid.desert.radial"
                                                                        ]
                                                                    ),
                                                                ),
                                                                ChildrenAction(
                                                                    scene=RadialMaze.factory(
                                                                        RadialMazeParams(
                                                                            arms=8,
                                                                            arm_width=3,
                                                                            arm_length=48,
                                                                            fill_background=False,
                                                                        )
                                                                    ),
                                                                    where=AreaWhere(
                                                                        tags=[
                                                                            "astroid.desert.radial"
                                                                        ]
                                                                    ),
                                                                    limit=1,
                                                                ),
                                                            ],
                                                        ),
                                                        where=AreaWhere(
                                                            tags=[
                                                                "astroid.desert.oasis"
                                                            ]
                                                        ),
                                                        limit=1,
                                                    ),
                                                ],
                                            ),
                                            where=AreaWhere(tags=["astroid.desert"]),
                                            limit=1,
                                        ),
                                        ChildrenAction(
                                            scene=BiomeDesert.factory(
                                                BiomeDesertParams(
                                                    dune_period=8,
                                                    ridge_width=2,
                                                    angle=0.4,
                                                    noise_prob=0.6,
                                                )
                                            ),
                                            where=AreaWhere(tags=["astroid.desert"]),
                                        ),
                                        ChildrenAction(
                                            scene=Layout.factory(
                                                LayoutParams(
                                                    areas=[
                                                        LayoutArea(
                                                            width=60,
                                                            height=60,
                                                            placement="center",
                                                            tag="astroid.radial",
                                                        )
                                                    ]
                                                ),
                                                children_actions=[
                                                    ChildrenAction(
                                                        scene=FillArea.factory(
                                                            FillAreaParams(
                                                                value="empty"
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=["astroid.radial"]
                                                        ),
                                                    ),
                                                    ChildrenAction(
                                                        scene=RadialMaze.factory(
                                                            RadialMazeParams(
                                                                arms=10,
                                                                arm_width=2,
                                                                arm_length=28,
                                                                fill_background=False,
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=["astroid.radial"]
                                                        ),
                                                        limit=1,
                                                    ),
                                                ],
                                            ),
                                            where=AreaWhere(tags=["astroid.desert"]),
                                            limit=1,
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                                lock="astroid.zone",
                            ),
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=120,
                                                height=120,
                                                placement="center",
                                                tag="astroid.forest",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(tags=["astroid.forest"]),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeForest.factory(
                                                BiomeForestParams(
                                                    clumpiness=4,
                                                    seed_prob=0.05,
                                                    growth_prob=0.6,
                                                )
                                            ),
                                            where=AreaWhere(tags=["astroid.forest"]),
                                        ),
                                        ChildrenAction(
                                            scene=Layout.factory(
                                                LayoutParams(
                                                    areas=[
                                                        LayoutArea(
                                                            width=110,
                                                            height=90,
                                                            placement="center",
                                                            tag="astroid.forest.ancient",
                                                        )
                                                    ]
                                                ),
                                                children_actions=[
                                                    ChildrenAction(
                                                        scene=FillArea.factory(
                                                            FillAreaParams(
                                                                value="empty"
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=[
                                                                "astroid.forest.ancient"
                                                            ]
                                                        ),
                                                    ),
                                                    ChildrenAction(
                                                        scene=BiomeForest.factory(
                                                            BiomeForestParams(
                                                                clumpiness=7,
                                                                seed_prob=0.03,
                                                                growth_prob=0.8,
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=[
                                                                "astroid.forest.ancient"
                                                            ]
                                                        ),
                                                    ),
                                                    ChildrenAction(
                                                        scene=Layout.factory(
                                                            LayoutParams(
                                                                areas=[
                                                                    LayoutArea(
                                                                        width=60,
                                                                        height=60,
                                                                        placement="center",
                                                                        tag="astroid.forest.bsp",
                                                                    )
                                                                ]
                                                            ),
                                                            children_actions=[
                                                                ChildrenAction(
                                                                    scene=FillArea.factory(
                                                                        FillAreaParams(
                                                                            value="empty"
                                                                        )
                                                                    ),
                                                                    where=AreaWhere(
                                                                        tags=[
                                                                            "astroid.forest.bsp"
                                                                        ]
                                                                    ),
                                                                ),
                                                                ChildrenAction(
                                                                    scene=BSP.factory(
                                                                        BSPParams(
                                                                            rooms=12,
                                                                            min_room_size=8,
                                                                            min_room_size_ratio=0.6,
                                                                            max_room_size_ratio=0.9,
                                                                        )
                                                                    ),
                                                                    where=AreaWhere(
                                                                        tags=[
                                                                            "astroid.forest.bsp"
                                                                        ]
                                                                    ),
                                                                    limit=1,
                                                                ),
                                                            ],
                                                        ),
                                                        where=AreaWhere(
                                                            tags=[
                                                                "astroid.forest.ancient"
                                                            ]
                                                        ),
                                                        limit=1,
                                                    ),
                                                ],
                                            ),
                                            where=AreaWhere(tags=["astroid.forest"]),
                                            limit=1,
                                        ),
                                        ChildrenAction(
                                            scene=Layout.factory(
                                                LayoutParams(
                                                    areas=[
                                                        LayoutArea(
                                                            width=48,
                                                            height=48,
                                                            placement="center",
                                                            tag="astroid.bsp",
                                                        )
                                                    ]
                                                ),
                                                children_actions=[
                                                    ChildrenAction(
                                                        scene=FillArea.factory(
                                                            FillAreaParams(
                                                                value="empty"
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=["astroid.bsp"]
                                                        ),
                                                    ),
                                                    ChildrenAction(
                                                        scene=BSP.factory(
                                                            BSPParams(
                                                                rooms=16,
                                                                min_room_size=6,
                                                                min_room_size_ratio=0.4,
                                                                max_room_size_ratio=0.8,
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=["astroid.bsp"]
                                                        ),
                                                        limit=1,
                                                    ),
                                                ],
                                            ),
                                            where=AreaWhere(tags=["astroid.forest"]),
                                            limit=1,
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                                lock="astroid.zone",
                            ),
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=120,
                                                height=120,
                                                placement="center",
                                                tag="astroid.city.maze1",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.city.maze1"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeCity.factory(
                                                BiomeCityParams(
                                                    pitch=10,
                                                    road_width=2,
                                                    jitter=2,
                                                    place_prob=0.9,
                                                )
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.city.maze1"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=Layout.factory(
                                                LayoutParams(
                                                    areas=[
                                                        LayoutArea(
                                                            width=130,
                                                            height=80,
                                                            placement="center",
                                                            tag="astroid.city.metropolis",
                                                        )
                                                    ]
                                                ),
                                                children_actions=[
                                                    ChildrenAction(
                                                        scene=FillArea.factory(
                                                            FillAreaParams(
                                                                value="empty"
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=[
                                                                "astroid.city.metropolis"
                                                            ]
                                                        ),
                                                    ),
                                                    ChildrenAction(
                                                        scene=BiomeCity.factory(
                                                            BiomeCityParams(
                                                                pitch=8,
                                                                road_width=4,
                                                                jitter=3,
                                                                place_prob=0.95,
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=[
                                                                "astroid.city.metropolis"
                                                            ]
                                                        ),
                                                    ),
                                                    ChildrenAction(
                                                        scene=Layout.factory(
                                                            LayoutParams(
                                                                areas=[
                                                                    LayoutArea(
                                                                        width=70,
                                                                        height=50,
                                                                        placement="center",
                                                                        tag="astroid.city.maze",
                                                                    )
                                                                ]
                                                            ),
                                                            children_actions=[
                                                                ChildrenAction(
                                                                    scene=FillArea.factory(
                                                                        FillAreaParams(
                                                                            value="empty"
                                                                        )
                                                                    ),
                                                                    where=AreaWhere(
                                                                        tags=[
                                                                            "astroid.city.maze"
                                                                        ]
                                                                    ),
                                                                ),
                                                                ChildrenAction(
                                                                    scene=Maze.factory(
                                                                        MazeParams(
                                                                            algorithm="dfs",
                                                                            room_size=IntConstantDistribution(
                                                                                value=5
                                                                            ),
                                                                            wall_size=IntConstantDistribution(
                                                                                value=3
                                                                            ),
                                                                        )
                                                                    ),
                                                                    where=AreaWhere(
                                                                        tags=[
                                                                            "astroid.city.maze"
                                                                        ]
                                                                    ),
                                                                    limit=1,
                                                                ),
                                                            ],
                                                        ),
                                                        where=AreaWhere(
                                                            tags=[
                                                                "astroid.city.metropolis"
                                                            ]
                                                        ),
                                                        limit=1,
                                                    ),
                                                ],
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.city.maze1"]
                                            ),
                                            limit=1,
                                        ),
                                        ChildrenAction(
                                            scene=Layout.factory(
                                                LayoutParams(
                                                    areas=[
                                                        LayoutArea(
                                                            width=60,
                                                            height=60,
                                                            placement="center",
                                                            tag="astroid.maze",
                                                        )
                                                    ]
                                                ),
                                                children_actions=[
                                                    ChildrenAction(
                                                        scene=FillArea.factory(
                                                            FillAreaParams(
                                                                value="empty"
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=["astroid.maze"]
                                                        ),
                                                    ),
                                                    ChildrenAction(
                                                        scene=Maze.factory(
                                                            MazeParams(
                                                                algorithm="dfs",
                                                                room_size=IntConstantDistribution(
                                                                    value=4
                                                                ),
                                                                wall_size=IntConstantDistribution(
                                                                    value=2
                                                                ),
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=["astroid.maze"]
                                                        ),
                                                        limit=1,
                                                    ),
                                                ],
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.city.maze1"]
                                            ),
                                            limit=1,
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                                lock="astroid.zone",
                            ),
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=36,
                                                height=36,
                                                placement="center",
                                                tag="astroid.city.maze2",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.city.maze2"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeCity.factory(
                                                BiomeCityParams(
                                                    pitch=10,
                                                    road_width=2,
                                                    jitter=2,
                                                    place_prob=0.9,
                                                )
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.city.maze2"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=Layout.factory(
                                                LayoutParams(
                                                    areas=[
                                                        LayoutArea(
                                                            width=28,
                                                            height=28,
                                                            placement="center",
                                                            tag="astroid.maze.kruskal",
                                                        )
                                                    ]
                                                ),
                                                children_actions=[
                                                    ChildrenAction(
                                                        scene=FillArea.factory(
                                                            FillAreaParams(
                                                                value="empty"
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=[
                                                                "astroid.maze.kruskal"
                                                            ]
                                                        ),
                                                    ),
                                                    ChildrenAction(
                                                        scene=Maze.factory(
                                                            MazeParams(
                                                                algorithm="kruskal",
                                                                room_size=IntConstantDistribution(
                                                                    value=2
                                                                ),
                                                                wall_size=IntConstantDistribution(
                                                                    value=1
                                                                ),
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=[
                                                                "astroid.maze.kruskal"
                                                            ]
                                                        ),
                                                        limit=1,
                                                    ),
                                                ],
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.city.maze2"]
                                            ),
                                            limit=1,
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                                lock="astroid.zone",
                            ),
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=80,
                                                height=50,
                                                placement="center",
                                                tag="astroid.bsp.crucible",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.bsp.crucible"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeCaves.factory(
                                                BiomeCavesParams(
                                                    fill_prob=0.35,
                                                    steps=3,
                                                    birth_limit=4,
                                                    death_limit=2,
                                                )
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.bsp.crucible"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=Layout.factory(
                                                LayoutParams(
                                                    areas=[
                                                        LayoutArea(
                                                            width=48,
                                                            height=36,
                                                            placement="center",
                                                            tag="astroid.bsp.crucible.core",
                                                        )
                                                    ]
                                                ),
                                                children_actions=[
                                                    ChildrenAction(
                                                        scene=FillArea.factory(
                                                            FillAreaParams(
                                                                value="empty"
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=[
                                                                "astroid.bsp.crucible.core"
                                                            ]
                                                        ),
                                                    ),
                                                    ChildrenAction(
                                                        scene=BSP.factory(
                                                            BSPParams(
                                                                rooms=20,
                                                                min_room_size=4,
                                                                min_room_size_ratio=0.5,
                                                                max_room_size_ratio=0.7,
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=[
                                                                "astroid.bsp.crucible.core"
                                                            ]
                                                        ),
                                                        limit=1,
                                                    ),
                                                ],
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.bsp.crucible"]
                                            ),
                                            limit=1,
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                                lock="astroid.zone",
                            ),
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=54,
                                                height=54,
                                                placement="center",
                                                tag="astroid.radial.alt",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.radial.alt"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeForest.factory(
                                                BiomeForestParams(
                                                    clumpiness=5,
                                                    seed_prob=0.08,
                                                    growth_prob=0.65,
                                                )
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.radial.alt"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=Layout.factory(
                                                LayoutParams(
                                                    areas=[
                                                        LayoutArea(
                                                            width=34,
                                                            height=34,
                                                            placement="center",
                                                            tag="astroid.radial.alt.core",
                                                        )
                                                    ]
                                                ),
                                                children_actions=[
                                                    ChildrenAction(
                                                        scene=FillArea.factory(
                                                            FillAreaParams(
                                                                value="empty"
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=[
                                                                "astroid.radial.alt.core"
                                                            ]
                                                        ),
                                                    ),
                                                    ChildrenAction(
                                                        scene=RadialMaze.factory(
                                                            RadialMazeParams(
                                                                arms=12,
                                                                arm_width=2,
                                                                arm_length=32,
                                                                fill_background=False,
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=[
                                                                "astroid.radial.alt.core"
                                                            ]
                                                        ),
                                                        limit=1,
                                                    ),
                                                ],
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.radial.alt"]
                                            ),
                                            limit=1,
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                                lock="astroid.zone",
                            ),
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=40,
                                                height=40,
                                                placement="center",
                                                tag="astroid.maze.prim",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(tags=["astroid.maze.prim"]),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeCity.factory(
                                                BiomeCityParams(
                                                    pitch=12,
                                                    road_width=3,
                                                    jitter=1,
                                                    place_prob=0.7,
                                                )
                                            ),
                                            where=AreaWhere(tags=["astroid.maze.prim"]),
                                        ),
                                        ChildrenAction(
                                            scene=Layout.factory(
                                                LayoutParams(
                                                    areas=[
                                                        LayoutArea(
                                                            width=28,
                                                            height=28,
                                                            placement="center",
                                                            tag="astroid.maze.prim.grid",
                                                        )
                                                    ]
                                                ),
                                                children_actions=[
                                                    ChildrenAction(
                                                        scene=FillArea.factory(
                                                            FillAreaParams(
                                                                value="empty"
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=[
                                                                "astroid.maze.prim.grid"
                                                            ]
                                                        ),
                                                    ),
                                                    ChildrenAction(
                                                        scene=Maze.factory(
                                                            MazeParams(
                                                                algorithm="dfs",
                                                                room_size=IntConstantDistribution(
                                                                    value=3
                                                                ),
                                                                wall_size=IntConstantDistribution(
                                                                    value=1
                                                                ),
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=[
                                                                "astroid.maze.prim.grid"
                                                            ]
                                                        ),
                                                        limit=1,
                                                    ),
                                                ],
                                            ),
                                            where=AreaWhere(tags=["astroid.maze.prim"]),
                                            limit=1,
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                                lock="astroid.zone",
                            ),
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=44,
                                                height=44,
                                                placement="center",
                                                tag="astroid.maze.unicursal",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.maze.unicursal"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeDesert.factory(
                                                BiomeDesertParams(
                                                    dune_period=6,
                                                    ridge_width=3,
                                                    angle=0.55,
                                                    noise_prob=0.45,
                                                )
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.maze.unicursal"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=Layout.factory(
                                                LayoutParams(
                                                    areas=[
                                                        LayoutArea(
                                                            width=32,
                                                            height=32,
                                                            placement="center",
                                                            tag="astroid.maze.unicursal.grid",
                                                        )
                                                    ]
                                                ),
                                                children_actions=[
                                                    ChildrenAction(
                                                        scene=FillArea.factory(
                                                            FillAreaParams(
                                                                value="empty"
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=[
                                                                "astroid.maze.unicursal.grid"
                                                            ]
                                                        ),
                                                    ),
                                                    ChildrenAction(
                                                        scene=Maze.factory(
                                                            MazeParams(
                                                                algorithm="kruskal",
                                                                room_size=IntConstantDistribution(
                                                                    value=2
                                                                ),
                                                                wall_size=IntConstantDistribution(
                                                                    value=2
                                                                ),
                                                            )
                                                        ),
                                                        where=AreaWhere(
                                                            tags=[
                                                                "astroid.maze.unicursal.grid"
                                                            ]
                                                        ),
                                                        limit=1,
                                                    ),
                                                ],
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.maze.unicursal"]
                                            ),
                                            limit=1,
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                                lock="astroid.zone",
                            ),
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=72,
                                                height=48,
                                                placement="center",
                                                tag="astroid.forest.glade",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.forest.glade"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeForest.factory(
                                                BiomeForestParams(
                                                    clumpiness=3,
                                                    seed_prob=0.12,
                                                    growth_prob=0.55,
                                                )
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.forest.glade"]
                                            ),
                                            limit=1,
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                                lock="astroid.zone",
                            ),
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=90,
                                                height=50,
                                                placement="center",
                                                tag="astroid.forest.mire",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.forest.mire"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeForest.factory(
                                                BiomeForestParams(
                                                    clumpiness=6,
                                                    seed_prob=0.04,
                                                    growth_prob=0.7,
                                                )
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.forest.mire"]
                                            ),
                                            limit=1,
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                                lock="astroid.zone",
                            ),
                        ],
                    ),
                    where="full",
                    order_by="first",
                    limit=1,
                ),
                # Second layer: dungeons fill their zones
                ChildrenAction(
                    scene=BSPLayout.factory(
                        BSPLayoutParams(area_count=40),
                        children_actions=[
                            # Medium BSP dungeon - fills zone
                            ChildrenAction(
                                scene=BSP.factory(
                                    BSPParams(
                                        rooms=15,
                                        min_room_size=5,
                                        min_room_size_ratio=0.4,
                                        max_room_size_ratio=0.7,
                                    )
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                            ),
                            # Radial maze - fills zone
                            ChildrenAction(
                                scene=RadialMaze.factory(
                                    RadialMazeParams(
                                        arms=8,
                                        arm_width=3,
                                        arm_length=50,
                                        fill_background=False,
                                    )
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                            ),
                            # DFS maze - fills zone
                            ChildrenAction(
                                scene=Maze.factory(
                                    MazeParams(
                                        algorithm="dfs",
                                        room_size=IntConstantDistribution(value=4),
                                        wall_size=IntConstantDistribution(value=2),
                                    )
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                            ),
                            # Kruskal maze - fills zone
                            ChildrenAction(
                                scene=Maze.factory(
                                    MazeParams(
                                        algorithm="kruskal",
                                        room_size=IntConstantDistribution(value=5),
                                        wall_size=IntConstantDistribution(value=3),
                                    )
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                            ),
                            # BSP variant - fills zone
                            ChildrenAction(
                                scene=BSP.factory(
                                    BSPParams(
                                        rooms=12,
                                        min_room_size=4,
                                        min_room_size_ratio=0.3,
                                        max_room_size_ratio=0.6,
                                    )
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                            ),
                            # Radial variant - fills zone
                            ChildrenAction(
                                scene=RadialMaze.factory(
                                    RadialMazeParams(
                                        arms=10,
                                        arm_width=2,
                                        arm_length=45,
                                        fill_background=False,
                                    )
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                            ),
                            # Dense DFS - fills zone
                            ChildrenAction(
                                scene=Maze.factory(
                                    MazeParams(
                                        algorithm="dfs",
                                        room_size=IntConstantDistribution(value=2),
                                        wall_size=IntConstantDistribution(value=1),
                                    )
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                            ),
                            # BSP compact - fills zone
                            ChildrenAction(
                                scene=BSP.factory(
                                    BSPParams(
                                        rooms=10,
                                        min_room_size=4,
                                        min_room_size_ratio=0.3,
                                        max_room_size_ratio=0.5,
                                    )
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                            ),
                            # Radial with many arms - fills zone
                            ChildrenAction(
                                scene=RadialMaze.factory(
                                    RadialMazeParams(
                                        arms=12,
                                        arm_width=3,
                                        arm_length=60,
                                        fill_background=False,
                                    )
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                            ),
                            # Kruskal wide - fills zone
                            ChildrenAction(
                                scene=Maze.factory(
                                    MazeParams(
                                        algorithm="kruskal",
                                        room_size=IntConstantDistribution(value=4),
                                        wall_size=IntConstantDistribution(value=3),
                                    )
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                            ),
                            # BSP large - fills zone
                            ChildrenAction(
                                scene=BSP.factory(
                                    BSPParams(
                                        rooms=18,
                                        min_room_size=6,
                                        min_room_size_ratio=0.35,
                                        max_room_size_ratio=0.65,
                                    )
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                            ),
                            # DFS wide - fills zone
                            ChildrenAction(
                                scene=Maze.factory(
                                    MazeParams(
                                        algorithm="dfs",
                                        room_size=IntConstantDistribution(value=5),
                                        wall_size=IntConstantDistribution(value=3),
                                    )
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                            ),
                            # Radial focused - fills zone
                            ChildrenAction(
                                scene=RadialMaze.factory(
                                    RadialMazeParams(
                                        arms=6,
                                        arm_width=3,
                                        arm_length=40,
                                        fill_background=False,
                                    )
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                            ),
                            # Kruskal dense - fills zone
                            ChildrenAction(
                                scene=Maze.factory(
                                    MazeParams(
                                        algorithm="kruskal",
                                        room_size=IntConstantDistribution(value=3),
                                        wall_size=IntConstantDistribution(value=2),
                                    )
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                            ),
                            # BSP spread - fills zone
                            ChildrenAction(
                                scene=BSP.factory(
                                    BSPParams(
                                        rooms=14,
                                        min_room_size=5,
                                        min_room_size_ratio=0.35,
                                        max_room_size_ratio=0.6,
                                    )
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                            ),
                        ],
                    ),
                    where="full",
                    order_by="first",
                    limit=1,
                ),
                # Third layer: biome patches fill their zones
                ChildrenAction(
                    scene=BSPLayout.factory(
                        BSPLayoutParams(area_count=50),
                        children_actions=[
                            # Desert patch 1 - wide dunes
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=85,
                                                height=55,
                                                placement="center",
                                                tag="astroid.biome.desert1",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.biome.desert1"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeDesert.factory(
                                                BiomeDesertParams(
                                                    dune_period=15,
                                                    ridge_width=5,
                                                    angle=0.25,
                                                    noise_prob=0.2,
                                                )
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.biome.desert1"]
                                            ),
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                            ),
                            # Forest patch 1 - dense growth
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=70,
                                                height=70,
                                                placement="center",
                                                tag="astroid.biome.forest1",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.biome.forest1"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeForest.factory(
                                                BiomeForestParams(
                                                    clumpiness=8,
                                                    seed_prob=0.02,
                                                    growth_prob=0.85,
                                                )
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.biome.forest1"]
                                            ),
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                            ),
                            # City patch 1 - wide streets
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=65,
                                                height=65,
                                                placement="center",
                                                tag="astroid.biome.city1",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.biome.city1"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeCity.factory(
                                                BiomeCityParams(
                                                    pitch=15,
                                                    road_width=4,
                                                    jitter=3,
                                                    place_prob=0.85,
                                                )
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.biome.city1"]
                                            ),
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                            ),
                            # Caves patch 1 - sparse
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=60,
                                                height=60,
                                                placement="center",
                                                tag="astroid.biome.caves1",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.biome.caves1"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeCaves.factory(
                                                BiomeCavesParams(
                                                    fill_prob=0.3,
                                                    steps=6,
                                                    birth_limit=6,
                                                    death_limit=2,
                                                )
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.biome.caves1"]
                                            ),
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                            ),
                            # Desert patch 2 - narrow ridges
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=75,
                                                height=60,
                                                placement="center",
                                                tag="astroid.biome.desert2",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.biome.desert2"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeDesert.factory(
                                                BiomeDesertParams(
                                                    dune_period=5,
                                                    ridge_width=1,
                                                    angle=0.7,
                                                    noise_prob=0.8,
                                                )
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.biome.desert2"]
                                            ),
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                            ),
                            # Forest patch 2 - sparse scattered
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=80,
                                                height=50,
                                                placement="center",
                                                tag="astroid.biome.forest2",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.biome.forest2"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeForest.factory(
                                                BiomeForestParams(
                                                    clumpiness=2,
                                                    seed_prob=0.15,
                                                    growth_prob=0.45,
                                                )
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.biome.forest2"]
                                            ),
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                            ),
                            # City patch 2 - dense narrow
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=55,
                                                height=75,
                                                placement="center",
                                                tag="astroid.biome.city2",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.biome.city2"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeCity.factory(
                                                BiomeCityParams(
                                                    pitch=6,
                                                    road_width=1,
                                                    jitter=0,
                                                    place_prob=0.98,
                                                )
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.biome.city2"]
                                            ),
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                            ),
                            # Caves patch 2 - dense
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=70,
                                                height=55,
                                                placement="center",
                                                tag="astroid.biome.caves2",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.biome.caves2"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeCaves.factory(
                                                BiomeCavesParams(
                                                    fill_prob=0.55,
                                                    steps=2,
                                                    birth_limit=3,
                                                    death_limit=4,
                                                )
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.biome.caves2"]
                                            ),
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                            ),
                            # Desert patch 3 - medium everything
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=60,
                                                height=80,
                                                placement="center",
                                                tag="astroid.biome.desert3",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.biome.desert3"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeDesert.factory(
                                                BiomeDesertParams(
                                                    dune_period=10,
                                                    ridge_width=3,
                                                    angle=0.5,
                                                    noise_prob=0.5,
                                                )
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.biome.desert3"]
                                            ),
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                            ),
                            # Forest patch 3 - medium clumps
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=68,
                                                height=68,
                                                placement="center",
                                                tag="astroid.biome.forest3",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.biome.forest3"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeForest.factory(
                                                BiomeForestParams(
                                                    clumpiness=5,
                                                    seed_prob=0.07,
                                                    growth_prob=0.65,
                                                )
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.biome.forest3"]
                                            ),
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                            ),
                            # Forest patch 4 - tendrils (very clumpy, low seed)
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=90,
                                                height=45,
                                                placement="center",
                                                tag="astroid.biome.forest4",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.biome.forest4"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeForest.factory(
                                                BiomeForestParams(
                                                    clumpiness=12,
                                                    seed_prob=0.01,
                                                    growth_prob=0.75,
                                                    dither_prob=0.25,
                                                )
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.biome.forest4"]
                                            ),
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                            ),
                            # Forest patch 5 - scattered islands (no clump, high seed)
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=55,
                                                height=85,
                                                placement="center",
                                                tag="astroid.biome.forest5",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.biome.forest5"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeForest.factory(
                                                BiomeForestParams(
                                                    clumpiness=1,
                                                    seed_prob=0.2,
                                                    growth_prob=0.4,
                                                    dither_prob=0.3,
                                                )
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.biome.forest5"]
                                            ),
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                            ),
                            # Forest patch 6 - wispy (extreme clump, very low growth)
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=75,
                                                height=75,
                                                placement="center",
                                                tag="astroid.biome.forest6",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.biome.forest6"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeForest.factory(
                                                BiomeForestParams(
                                                    clumpiness=15,
                                                    seed_prob=0.08,
                                                    growth_prob=0.3,
                                                    dither_prob=0.35,
                                                )
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.biome.forest6"]
                                            ),
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                            ),
                            # City patch 3 - chaotic
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=72,
                                                height=62,
                                                placement="center",
                                                tag="astroid.biome.city3",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.biome.city3"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeCity.factory(
                                                BiomeCityParams(
                                                    pitch=8,
                                                    road_width=2,
                                                    jitter=4,
                                                    place_prob=0.75,
                                                )
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.biome.city3"]
                                            ),
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                            ),
                            # Caves patch 3 - winding
                            ChildrenAction(
                                scene=Layout.factory(
                                    LayoutParams(
                                        areas=[
                                            LayoutArea(
                                                width=65,
                                                height=65,
                                                placement="center",
                                                tag="astroid.biome.caves3",
                                            )
                                        ]
                                    ),
                                    children_actions=[
                                        ChildrenAction(
                                            scene=FillArea.factory(
                                                FillAreaParams(value="empty")
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.biome.caves3"]
                                            ),
                                        ),
                                        ChildrenAction(
                                            scene=BiomeCaves.factory(
                                                BiomeCavesParams(
                                                    fill_prob=0.42,
                                                    steps=7,
                                                    birth_limit=4,
                                                    death_limit=3,
                                                )
                                            ),
                                            where=AreaWhere(
                                                tags=["astroid.biome.caves3"]
                                            ),
                                        ),
                                    ],
                                ),
                                where=AreaWhere(tags=["zone"]),
                                limit=1,
                                order_by="random",
                            ),
                        ],
                    ),
                    where="full",
                    order_by="first",
                    limit=1,
                ),
                # Central sanctum retained as the cave nexus, stamped last.
                ChildrenAction(
                    scene=Layout.factory(
                        LayoutParams(
                            areas=[
                                LayoutArea(
                                    width=15,
                                    height=15,
                                    placement="center",
                                    tag="sanctum.center",
                                )
                            ]
                        ),
                        children_actions=[
                            ChildrenAction(
                                scene=BaseHub.factory(
                                    BaseHubParams(
                                        altar_object="assembler",
                                        include_inner_wall=True,
                                        corner_objects=[
                                            "carbon_ex_dep",  # TL
                                            "oxygen_ex_dep",  # TR
                                            "germanium_ex_dep",  # BL
                                            "silicon_ex_dep",  # BR
                                        ],
                                    )
                                ),
                                where=AreaWhere(tags=["sanctum.center"]),
                                limit=1,
                            )
                        ],
                    ),
                    where="full",
                    order_by="last",
                    lock="sanctum",
                    limit=1,
                ),
                ChildrenAction(
                    scene=MakeConnected.factory(MakeConnectedParams()),
                    where="full",
                    order_by="last",
                    lock="connect",
                    limit=1,
                ),
                ChildrenAction(
                    scene=UniformExtractorScene.factory(
                        UniformExtractorParams(
                            target_coverage=0.15,
                            jitter=0,
                            extractor_names=[
                                "carbon_extractor",
                                "oxygen_extractor",
                                "germanium_extractor",
                                "silicon_extractor",
                            ],
                        )
                    ),
                    where="full",
                    order_by="last",
                    lock="sanctum_extractors",
                    limit=1,
                ),
            ],
        ),
    )

    env = make_navigation(num_agents=4)
    # reuse simple action config and objects (altar removed later if needed)
    env.game.map_builder = MapGen.Config(
        width=width,
        height=height,
        root=Quadrants.factory(
            params=QuadrantsParams(base_size=6),  # Creates 4 quadrants for biomes (0-3)
            # Dungeons use BSP layout to create 3 bounded zones
            children_actions=[
                # Top-left: City biome
                ChildrenAction(
                    scene=BiomeCity.factory(
                        BiomeCityParams(
                            pitch=10, road_width=2, jitter=2, place_prob=0.9
                        )
                    ),
                    where=AreaWhere(tags=["quadrant", "quadrant.0"]),
                    order_by="first",
                    lock="biome",
                    limit=1,
                ),
                # Top-right: Forest biome
                ChildrenAction(
                    scene=BiomeForest.factory(
                        BiomeForestParams(clumpiness=4, seed_prob=0.05, growth_prob=0.6)
                    ),
                    where=AreaWhere(tags=["quadrant", "quadrant.1"]),
                    order_by="first",
                    lock="biome",
                    limit=1,
                ),
                # Bottom-left: Caves biome
                ChildrenAction(
                    scene=BiomeCaves.factory(
                        BiomeCavesParams(
                            fill_prob=0.45, steps=4, birth_limit=5, death_limit=3
                        )
                    ),
                    where=AreaWhere(tags=["quadrant", "quadrant.2"]),
                    order_by="first",
                    lock="biome",
                    limit=1,
                ),
                # Fractal-style dungeon embedded in caves quadrant
                ChildrenAction(
                    scene=Layout.factory(
                        LayoutParams(
                            areas=[
                                LayoutArea(
                                    width=50,
                                    height=50,
                                    placement="center",
                                    tag="dz_fractal",
                                )
                            ]
                        ),
                        children_actions=[
                            ChildrenAction(
                                scene=Maze.factory(
                                    MazeParams(
                                        algorithm="kruskal",
                                        room_size=IntConstantDistribution(value=2),
                                        wall_size=IntConstantDistribution(value=1),
                                    )
                                ),
                                where=AreaWhere(tags=["dz_fractal"]),
                                limit=1,
                            )
                        ],
                    ),
                    where=AreaWhere(tags=["quadrant", "quadrant.2"]),
                    order_by="last",
                    lock="dungeon",
                    limit=1,
                ),
                # Bottom-right: Desert biome
                ChildrenAction(
                    scene=BiomeDesert.factory(
                        BiomeDesertParams(
                            dune_period=8, ridge_width=2, angle=0.4, noise_prob=0.6
                        )
                    ),
                    where=AreaWhere(tags=["quadrant", "quadrant.3"]),
                    order_by="first",
                    lock="biome",
                    limit=1,
                ),
                # BSP Dungeon - centered in city quadrant
                ChildrenAction(
                    scene=Layout.factory(
                        LayoutParams(
                            areas=[
                                LayoutArea(
                                    width=70,
                                    height=60,
                                    placement="center",
                                    tag="dz_bsp",
                                )
                            ]
                        ),
                        children_actions=[
                            ChildrenAction(
                                scene=BSP.factory(
                                    BSPParams(
                                        rooms=8,
                                        min_room_size=3,
                                        min_room_size_ratio=0.4,
                                        max_room_size_ratio=0.8,
                                    )
                                ),
                                where=AreaWhere(tags=["dz_bsp"]),
                                limit=1,
                            )
                        ],
                    ),
                    where=AreaWhere(tags=["quadrant", "quadrant.0"]),
                    order_by="last",
                    lock="dungeon",
                    limit=1,
                ),
                # Maze Dungeon - centered in forest quadrant
                ChildrenAction(
                    scene=Layout.factory(
                        LayoutParams(
                            areas=[
                                LayoutArea(
                                    width=50,
                                    height=50,
                                    placement="center",
                                    tag="dz_maze",
                                )
                            ]
                        ),
                        children_actions=[
                            ChildrenAction(
                                scene=Maze.factory(
                                    MazeParams(
                                        algorithm="dfs",
                                        room_size=IntConstantDistribution(value=3),
                                        wall_size=IntConstantDistribution(value=1),
                                    )
                                ),
                                where=AreaWhere(tags=["dz_maze"]),
                                limit=1,
                            )
                        ],
                    ),
                    where=AreaWhere(tags=["quadrant", "quadrant.1"]),
                    order_by="last",
                    lock="dungeon",
                    limit=1,
                ),
                # Radial Maze Dungeon - centered in desert quadrant
                ChildrenAction(
                    scene=Layout.factory(
                        LayoutParams(
                            areas=[
                                LayoutArea(
                                    width=80,
                                    height=80,
                                    placement="center",
                                    tag="dz_radial",
                                )
                            ]
                        ),
                        children_actions=[
                            ChildrenAction(
                                scene=RadialMaze.factory(
                                    RadialMazeParams(
                                        arms=10,
                                        arm_width=2,
                                        arm_length=40,
                                        fill_background=False,
                                    )
                                ),
                                where=AreaWhere(tags=["dz_radial"]),
                                limit=1,
                            )
                        ],
                    ),
                    where=AreaWhere(tags=["quadrant", "quadrant.3"]),
                    order_by="last",
                    lock="dungeon",
                    limit=1,
                ),
                # Global connectivity pass - runs after everything to connect dungeons and biomes
                ChildrenAction(
                    scene=MakeConnected.factory(MakeConnectedParams()),
                    where="full",
                    order_by="last",
                    lock="connect",
                    limit=1,
                ),
                # Central sanctum stamp: carve centered area then place BaseHub inside it
                ChildrenAction(
                    scene=Layout.factory(
                        LayoutParams(
                            areas=[
                                LayoutArea(
                                    width=15,
                                    height=15,
                                    placement="center",
                                    tag="sanctum.center",
                                )
                            ]
                        ),
                        children_actions=[
                            ChildrenAction(
                                scene=BaseHub.factory(
                                    BaseHubParams(
                                        altar_object="altar",
                                        include_inner_wall=True,
                                        corner_objects=[
                                            "carbon_ex_dep",  # TL
                                            "oxygen_ex_dep",  # TR
                                            "germanium_ex_dep",  # BL
                                            "silicon_ex_dep",  # BR
                                        ],
                                    )
                                ),
                                where=AreaWhere(tags=["sanctum.center"]),
                                limit=1,
                            )
                        ],
                    ),
                    where="full",
                    order_by="last",
                    lock="sanctum",
                    limit=1,
                ),
            ],
        ),
    )

    # BSP Dungeon environment - standalone roguelike dungeon (15x15 to 25x25)
    bsp_dungeon = make_navigation(num_agents=4)
    bsp_dungeon.game.map_builder = MapGen.Config(
        width=width,
        height=height,
        root=BSP.factory(
            BSPParams(
                rooms=10,
                min_room_size=4,
                min_room_size_ratio=0.4,
                max_room_size_ratio=0.9,
            )
        ),
    )

    # Radial Maze Dungeon environment - standalone fractal maze (12x12 to 20x20)
    radial_maze = make_navigation(num_agents=4)
    radial_maze.game.map_builder = MapGen.Config(
        width=width,
        height=height,
        root=RadialMaze.factory(
            RadialMazeParams(arms=8, arm_width=3, arm_length=15, fill_background=False)
        ),
    )

    extractor_showcase = make_extractor_showcase()

    return (
        env,
        desert_noise,
        city,
        caves,
        forest,
        bsp_dungeon,
        radial_maze,
        astroid,
        astroid_big,
        extractor_showcase,
        make_basehub_showcase(),
    )


def make_sanctum_caves_test() -> MettaGridConfig:
    env = make_navigation(num_agents=4)
    _add_extractor_objects(env)

    env.game.map_builder = MapGen.Config(
        width=50,
        height=50,
        root=BiomeCaves.factory(
            BiomeCavesParams(
                fill_prob=0.45,
                steps=4,
                birth_limit=5,
                death_limit=3,
            ),
            children_actions=[
                # Spread resource extractors across the map uniformly
                ChildrenAction(
                    scene=UniformExtractorScene.factory(
                        UniformExtractorParams(
                            target_coverage=0.02,
                            jitter=0,
                            extractor_names=[
                                "carbon_extractor",
                                "oxygen_extractor",
                                "germanium_extractor",
                                "silicon_extractor",
                                "charger",
                            ],
                        )
                    ),
                    where="full",
                    order_by="first",
                    limit=1,
                ),
                # Stamp a sanctum hub in the center (last so it isn't overwritten)
                ChildrenAction(
                    scene=Layout.factory(
                        LayoutParams(
                            areas=[
                                LayoutArea(
                                    width=15,
                                    height=15,
                                    placement="center",
                                    tag="sanctum.center",
                                )
                            ]
                        ),
                        children_actions=[
                            ChildrenAction(
                                scene=BaseHub.factory(
                                    BaseHubParams(
                                        altar_object="altar",
                                        include_inner_wall=True,
                                        corner_objects=[
                                            "carbon_ex_dep",
                                            "oxygen_ex_dep",
                                            "germanium_ex_dep",
                                            "silicon_ex_dep",
                                        ],
                                    )
                                ),
                                where=AreaWhere(tags=["sanctum.center"]),
                                limit=1,
                            )
                        ],
                    ),
                    where="full",
                    order_by="last",
                    lock="sanctum",
                    limit=1,
                ),
                ChildrenAction(
                    scene=MakeConnected.factory(MakeConnectedParams()),
                    where="full",
                    order_by="last",
                    lock="connect",
                    limit=1,
                ),
            ],
        ),
    )

    return env


def make_evals() -> List[SimulationConfig]:
    (
        env,
        desert_noise,
        city,
        caves,
        forest,
        bsp_dungeon,
        radial_maze,
        astroid,
        astroid_big,
        extractor_showcase,
        basehub_showcase,
    ) = make_mettagrid()
    sanctum_caves_test = make_sanctum_caves_test()
    return [
        SimulationConfig(suite="biomes", name="biomes_quadrants", env=env),
        SimulationConfig(suite="biomes", name="desert_noise", env=desert_noise),
        SimulationConfig(suite="biomes", name="city", env=city),
        SimulationConfig(suite="biomes", name="caves", env=caves),
        SimulationConfig(suite="biomes", name="forest", env=forest),
        SimulationConfig(suite="biomes", name="bsp_dungeon", env=bsp_dungeon),
        SimulationConfig(suite="biomes", name="radial_maze", env=radial_maze),
        SimulationConfig(suite="biomes", name="astroid", env=astroid),
        SimulationConfig(suite="biomes", name="astroid_big", env=astroid_big),
        SimulationConfig(
            suite="biomes", name="extractor_showcase", env=extractor_showcase
        ),
        SimulationConfig(
            suite="biomes", name="sanctum_caves_test", env=sanctum_caves_test
        ),
        SimulationConfig(suite="biomes", name="basehub_showcase", env=basehub_showcase),
    ]
