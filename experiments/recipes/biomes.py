import math
import random
from typing import Callable, List, Optional

from cogames.cogs_vs_clips import stations as cvc_stations
from metta.sim.simulation_config import SimulationConfig
from mettagrid import MettaGridConfig
from mettagrid.builder.envs import make_navigation
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
from mettagrid.mapgen.scenes.uniform_extractors import (
    UniformExtractorParams,
    UniformExtractorScene,
)
from mettagrid.mapgen.types import AreaWhere


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


def _compute_scale(width: int, height: int) -> float:
    base_area = 500 * 500
    current_area = width * height
    return math.sqrt(current_area / base_area)


def _scale_int(
    base: int, scale: float, minimum: int, maximum: Optional[int] = None
) -> int:
    value = max(minimum, int(round(base * scale)))
    if maximum is not None:
        value = min(value, maximum)
    return value


def _rand_tag(prefix: str, rng: random.Random) -> str:
    return f"{prefix}.{rng.randrange(10_000_000)}"


def _build_zone_actions(
    rng: random.Random,
    scale: float,
    count: int,
    max_width: int,
    max_height: int,
) -> list[ChildrenAction]:
    max_feature_width = max(32, max_width // 3)
    max_feature_height = max(32, max_height // 3)

    def _wrap_layout(
        width: int,
        height: int,
        tag: str,
        nested_children: list[ChildrenAction],
        background: Optional[str] = "empty",
    ):
        clamped_width = max(12, min(width, max_feature_width))
        clamped_height = max(12, min(height, max_feature_height))
        children: list[ChildrenAction] = []
        if background:
            children.append(
                ChildrenAction(
                    scene=FillArea.factory(FillAreaParams(value=background)),
                    where=AreaWhere(tags=[tag]),
                )
            )
        children.extend(nested_children)
        return Layout.factory(
            LayoutParams(
                areas=[
                    LayoutArea(
                        width=clamped_width,
                        height=clamped_height,
                        placement="center",
                        tag=tag,
                    )
                ]
            ),
            children_actions=children,
        )

    def zone_desert_radial(randomizer: random.Random, factor: float) -> ChildrenAction:
        outer_tag = _rand_tag("scalable.desert", randomizer)
        inner_tag = f"{outer_tag}.radial"
        outer_size = min(_scale_int(90, factor, 36, 220), max_feature_width)
        inner_size = max(12, min(outer_size - 6, _scale_int(30, factor, 14, 160)))
        inner_scene = _wrap_layout(
            inner_size,
            inner_size,
            inner_tag,
            [
                ChildrenAction(
                    scene=RadialMaze.factory(
                        RadialMazeParams(
                            arms=_scale_int(10, factor, 6, 18),
                            arm_width=_scale_int(2, factor, 1, 5),
                            arm_length=_scale_int(28, factor, 12, 90),
                            fill_background=False,
                        )
                    ),
                    where=AreaWhere(tags=[inner_tag]),
                    limit=1,
                )
            ],
        )
        outer_children = [
            ChildrenAction(
                scene=BiomeDesert.factory(
                    BiomeDesertParams(
                        dune_period=_scale_int(8, factor, 4, 24),
                        ridge_width=_scale_int(2, factor, 1, 7),
                        angle=randomizer.uniform(0.3, 0.6),
                        noise_prob=randomizer.uniform(0.35, 0.7),
                    )
                ),
                where=AreaWhere(tags=[outer_tag]),
            ),
            ChildrenAction(
                scene=inner_scene,
                where=AreaWhere(tags=[outer_tag]),
                limit=1,
            ),
        ]
        return ChildrenAction(
            scene=_wrap_layout(
                outer_size, outer_size, outer_tag, outer_children, background="empty"
            ),
            where=AreaWhere(tags=["zone"]),
            order_by="random",
            lock="scalable.zone",
            limit=1,
        )

    def zone_forest_bsp(randomizer: random.Random, factor: float) -> ChildrenAction:
        outer_tag = _rand_tag("scalable.forest", randomizer)
        inner_tag = f"{outer_tag}.bsp"
        outer_width = min(_scale_int(80, factor, 30, 180), max_feature_width)
        outer_height = min(_scale_int(70, factor, 30, 160), max_feature_height)
        inner_width = max(12, min(outer_width - 6, _scale_int(40, factor, 18, 140)))
        inner_height = max(12, min(outer_height - 6, _scale_int(34, factor, 18, 140)))
        inner_scene = _wrap_layout(
            inner_width,
            inner_height,
            inner_tag,
            [
                ChildrenAction(
                    scene=BSP.factory(
                        BSPParams(
                            rooms=_scale_int(12, factor, 6, 32),
                            min_room_size=_scale_int(5, factor, 3, 12),
                            min_room_size_ratio=0.3,
                            max_room_size_ratio=0.7,
                        )
                    ),
                    where=AreaWhere(tags=[inner_tag]),
                    limit=1,
                )
            ],
        )
        outer_children = [
            ChildrenAction(
                scene=BiomeForest.factory(
                    BiomeForestParams(
                        clumpiness=_scale_int(5, factor, 3, 14),
                        seed_prob=randomizer.uniform(0.03, 0.12),
                        growth_prob=randomizer.uniform(0.55, 0.75),
                        dither_prob=min(0.4, randomizer.uniform(0.0, 0.3)),
                    )
                ),
                where=AreaWhere(tags=[outer_tag]),
            ),
            ChildrenAction(
                scene=inner_scene,
                where=AreaWhere(tags=[outer_tag]),
                limit=1,
            ),
        ]
        return ChildrenAction(
            scene=_wrap_layout(
                outer_width,
                outer_height,
                outer_tag,
                outer_children,
                background="empty",
            ),
            where=AreaWhere(tags=["zone"]),
            order_by="random",
            lock="scalable.zone",
            limit=1,
        )

    def zone_city_maze(randomizer: random.Random, factor: float) -> ChildrenAction:
        outer_tag = _rand_tag("scalable.city", randomizer)
        inner_tag = f"{outer_tag}.maze"
        outer_width = min(_scale_int(70, factor, 28, 180), max_feature_width)
        outer_height = min(_scale_int(70, factor, 28, 180), max_feature_height)
        inner_size = max(
            12, min(min(outer_width, outer_height) - 6, _scale_int(40, factor, 18, 150))
        )
        algorithm = randomizer.choice(["dfs", "kruskal"])
        base_room = 3 if algorithm == "dfs" else 4
        room_size = _scale_int(base_room, factor, 2, 8)
        wall_size = min(
            room_size, _scale_int(2 if algorithm == "kruskal" else 1, factor, 1, 5)
        )
        inner_scene = _wrap_layout(
            inner_size,
            inner_size,
            inner_tag,
            [
                ChildrenAction(
                    scene=Maze.factory(
                        MazeParams(
                            algorithm=algorithm,
                            room_size=IntConstantDistribution(value=room_size),
                            wall_size=IntConstantDistribution(value=wall_size),
                        )
                    ),
                    where=AreaWhere(tags=[inner_tag]),
                    limit=1,
                )
            ],
        )
        outer_children = [
            ChildrenAction(
                scene=BiomeCity.factory(
                    BiomeCityParams(
                        pitch=_scale_int(10, factor, 6, 18),
                        road_width=_scale_int(2, factor, 1, 5),
                        jitter=_scale_int(2, factor, 0, 5),
                        place_prob=randomizer.uniform(0.75, 0.95),
                    )
                ),
                where=AreaWhere(tags=[outer_tag]),
            ),
            ChildrenAction(
                scene=inner_scene,
                where=AreaWhere(tags=[outer_tag]),
                limit=1,
            ),
        ]
        return ChildrenAction(
            scene=_wrap_layout(
                outer_width,
                outer_height,
                outer_tag,
                outer_children,
                background="empty",
            ),
            where=AreaWhere(tags=["zone"]),
            order_by="random",
            lock="scalable.zone",
            limit=1,
        )

    def zone_caves_cluster(randomizer: random.Random, factor: float) -> ChildrenAction:
        outer_tag = _rand_tag("scalable.caves", randomizer)
        area_width = min(_scale_int(70, factor, 28, 200), max_feature_width)
        area_height = min(_scale_int(60, factor, 28, 200), max_feature_height)
        outer_children = [
            ChildrenAction(
                scene=BiomeCaves.factory(
                    BiomeCavesParams(
                        fill_prob=randomizer.uniform(0.35, 0.55),
                        steps=_scale_int(4, factor, 2, 7),
                        birth_limit=_scale_int(5, factor, 3, 7),
                        death_limit=_scale_int(3, factor, 2, 6),
                    )
                ),
                where=AreaWhere(tags=[outer_tag]),
            )
        ]
        return ChildrenAction(
            scene=_wrap_layout(
                area_width,
                area_height,
                outer_tag,
                outer_children,
                background="empty",
            ),
            where=AreaWhere(tags=["zone"]),
            order_by="random",
            lock="scalable.zone",
            limit=1,
        )

    def zone_bsp_full(randomizer: random.Random, factor: float) -> ChildrenAction:
        tag = _rand_tag("scalable.feature.bsp", randomizer)
        width = min(_scale_int(70, factor, 28, 200), max_feature_width)
        height = min(_scale_int(60, factor, 28, 200), max_feature_height)
        params = BSPParams(
            rooms=_scale_int(14, factor, 6, 40),
            min_room_size=_scale_int(4, factor, 3, 12),
            min_room_size_ratio=min(0.6, max(0.25, 0.35 * factor)),
            max_room_size_ratio=min(0.9, max(0.5, 0.7 * factor)),
        )
        inner_children = [
            ChildrenAction(
                scene=BSP.factory(params),
                where=AreaWhere(tags=[tag]),
                limit=1,
            )
        ]
        return ChildrenAction(
            scene=_wrap_layout(
                width,
                height,
                tag,
                inner_children,
                background="empty",
            ),
            where=AreaWhere(tags=["zone"]),
            order_by="random",
            lock="scalable.zone",
            limit=1,
        )

    def zone_radial_full(randomizer: random.Random, factor: float) -> ChildrenAction:
        tag = _rand_tag("scalable.feature.radial", randomizer)
        width = min(_scale_int(72, factor, 28, 200), max_feature_width)
        height = min(_scale_int(72, factor, 28, 200), max_feature_height)
        params = RadialMazeParams(
            arms=_scale_int(12, factor, 5, 22),
            arm_width=_scale_int(3, factor, 1, 6),
            arm_length=_scale_int(36, factor, 14, 96),
            fill_background=False,
        )
        inner_children = [
            ChildrenAction(
                scene=RadialMaze.factory(params),
                where=AreaWhere(tags=[tag]),
                limit=1,
            )
        ]
        return ChildrenAction(
            scene=_wrap_layout(
                width,
                height,
                tag,
                inner_children,
                background="empty",
            ),
            where=AreaWhere(tags=["zone"]),
            order_by="random",
            lock="scalable.zone",
            limit=1,
        )

    def zone_maze_dense(randomizer: random.Random, factor: float) -> ChildrenAction:
        tag = _rand_tag("scalable.feature.maze", randomizer)
        width = min(_scale_int(60, factor, 24, 160), max_feature_width)
        height = min(_scale_int(60, factor, 24, 160), max_feature_height)
        room_size = _scale_int(2, factor, 2, 5)
        wall_size = min(room_size, _scale_int(1, factor, 1, 3))
        params = MazeParams(
            algorithm="dfs",
            room_size=IntConstantDistribution(value=room_size),
            wall_size=IntConstantDistribution(value=wall_size),
        )
        inner_children = [
            ChildrenAction(
                scene=Maze.factory(params),
                where=AreaWhere(tags=[tag]),
                limit=1,
            )
        ]
        return ChildrenAction(
            scene=_wrap_layout(
                width,
                height,
                tag,
                inner_children,
                background="empty",
            ),
            where=AreaWhere(tags=["zone"]),
            order_by="random",
            lock="scalable.zone",
            limit=1,
        )

    def zone_city_dense(randomizer: random.Random, factor: float) -> ChildrenAction:
        tag = _rand_tag("scalable.feature.city", randomizer)
        width = min(_scale_int(60, factor, 24, 180), max_feature_width)
        height = min(_scale_int(60, factor, 24, 180), max_feature_height)
        params = BiomeCityParams(
            pitch=_scale_int(12, factor, 6, 22),
            road_width=_scale_int(3, factor, 1, 6),
            jitter=_scale_int(3, factor, 0, 6),
            place_prob=randomizer.uniform(0.7, 0.98),
        )
        inner_children = [
            ChildrenAction(
                scene=BiomeCity.factory(params),
                where=AreaWhere(tags=[tag]),
            )
        ]
        return ChildrenAction(
            scene=_wrap_layout(
                width,
                height,
                tag,
                inner_children,
                background="empty",
            ),
            where=AreaWhere(tags=["zone"]),
            order_by="random",
            lock="scalable.zone",
            limit=1,
        )

    builders: list[Callable[[random.Random, float], ChildrenAction]] = [
        zone_desert_radial,
        zone_forest_bsp,
        zone_city_maze,
        zone_caves_cluster,
        zone_bsp_full,
        zone_radial_full,
        zone_maze_dense,
        zone_city_dense,
    ]

    return [rng.choice(builders)(rng, scale) for _ in range(count)]


def _build_dungeon_actions(
    rng: random.Random,
    scale: float,
    count: int,
    max_width: int,
    max_height: int,
) -> list[ChildrenAction]:
    max_dungeon_width = max(16, max_width // 4)
    max_dungeon_height = max(16, max_height // 4)

    def _wrap_dungeon(
        width: int,
        height: int,
        tag: str,
        inner_child: ChildrenAction,
        background: str = "empty",
    ):
        clamped_width = max(10, min(width, max_dungeon_width))
        clamped_height = max(10, min(height, max_dungeon_height))
        children: list[ChildrenAction] = []
        if background:
            children.append(
                ChildrenAction(
                    scene=FillArea.factory(FillAreaParams(value=background)),
                    where=AreaWhere(tags=[tag]),
                )
            )
        children.append(inner_child)
        return Layout.factory(
            LayoutParams(
                areas=[
                    LayoutArea(
                        width=clamped_width,
                        height=clamped_height,
                        placement="center",
                        tag=tag,
                    )
                ]
            ),
            children_actions=children,
        )

    def dungeon_bsp(randomizer: random.Random, factor: float) -> ChildrenAction:
        tag = _rand_tag("scalable.dungeon.bsp", randomizer)
        width = _scale_int(48, factor, 20, 160)
        height = _scale_int(36, factor, 18, 140)
        params = BSPParams(
            rooms=_scale_int(8, factor, 6, 20),
            min_room_size=_scale_int(3, factor, 2, 8),
            min_room_size_ratio=0.35,
            max_room_size_ratio=0.7,
        )
        inner_child = ChildrenAction(
            scene=BSP.factory(params),
            where=AreaWhere(tags=[tag]),
            limit=1,
        )
        return ChildrenAction(
            scene=_wrap_dungeon(width, height, tag, inner_child, background="empty"),
            where=AreaWhere(tags=["zone"]),
            order_by="random",
            lock="scalable.zone.dungeon",
            limit=1,
        )

    def dungeon_dfs(randomizer: random.Random, factor: float) -> ChildrenAction:
        tag = _rand_tag("scalable.dungeon.dfs", randomizer)
        width = _scale_int(40, factor, 18, 140)
        height = _scale_int(40, factor, 18, 140)
        room_size = _scale_int(3, factor, 2, 5)
        wall_size = max(1, min(room_size - 1, _scale_int(1, factor, 1, 3)))
        params = MazeParams(
            algorithm="dfs",
            room_size=IntConstantDistribution(value=room_size),
            wall_size=IntConstantDistribution(value=wall_size),
        )
        inner_child = ChildrenAction(
            scene=Maze.factory(params),
            where=AreaWhere(tags=[tag]),
            limit=1,
        )
        return ChildrenAction(
            scene=_wrap_dungeon(width, height, tag, inner_child, background="empty"),
            where=AreaWhere(tags=["zone"]),
            order_by="random",
            lock="scalable.zone.dungeon",
            limit=1,
        )

    def dungeon_kruskal(randomizer: random.Random, factor: float) -> ChildrenAction:
        tag = _rand_tag("scalable.dungeon.kruskal", randomizer)
        width = _scale_int(42, factor, 18, 150)
        height = _scale_int(42, factor, 18, 150)
        room_size = _scale_int(4, factor, 2, 6)
        wall_size = max(1, min(room_size - 1, _scale_int(2, factor, 1, 4)))
        params = MazeParams(
            algorithm="kruskal",
            room_size=IntConstantDistribution(value=room_size),
            wall_size=IntConstantDistribution(value=wall_size),
        )
        inner_child = ChildrenAction(
            scene=Maze.factory(params),
            where=AreaWhere(tags=[tag]),
            limit=1,
        )
        return ChildrenAction(
            scene=_wrap_dungeon(width, height, tag, inner_child, background="empty"),
            where=AreaWhere(tags=["zone"]),
            order_by="random",
            lock="scalable.zone.dungeon",
            limit=1,
        )

    def dungeon_radial(randomizer: random.Random, factor: float) -> ChildrenAction:
        tag = _rand_tag("scalable.dungeon.radial", randomizer)
        width = _scale_int(44, factor, 18, 150)
        height = _scale_int(44, factor, 18, 150)
        params = RadialMazeParams(
            arms=_scale_int(8, factor, 4, 16),
            arm_width=_scale_int(2, factor, 1, 4),
            arm_length=_scale_int(20, factor, 10, 40),
            fill_background=False,
        )
        inner_child = ChildrenAction(
            scene=RadialMaze.factory(params),
            where=AreaWhere(tags=[tag]),
            limit=1,
        )
        return ChildrenAction(
            scene=_wrap_dungeon(width, height, tag, inner_child, background="empty"),
            where=AreaWhere(tags=["zone"]),
            order_by="random",
            lock="scalable.zone.dungeon",
            limit=1,
        )

    builders: list[Callable[[random.Random, float], ChildrenAction]] = [
        dungeon_bsp,
        dungeon_dfs,
        dungeon_kruskal,
        dungeon_radial,
    ]

    return [rng.choice(builders)(rng, scale) for _ in range(count)]


def _uniform_extractor_params(scale: float) -> UniformExtractorParams:
    target = min(0.2, max(0.003, 0.006 * scale))
    return UniformExtractorParams(
        target_coverage=target,
        jitter=max(0, _scale_int(0, scale, 0, 2)),
        extractor_names=[
            "carbon_extractor",
            "oxygen_extractor",
            "germanium_extractor",
            "silicon_extractor",
            "charger",
        ],
    )


def make_scalable_astroid(
    width: int,
    height: int,
    seed: Optional[int] = None,
) -> MettaGridConfig:
    rng = random.Random(seed)
    scale = _compute_scale(width, height)

    env = make_navigation(num_agents=4)
    _add_extractor_objects(env)

    resources = set(env.game.resource_names)
    resources.update({"energy", "carbon", "oxygen", "germanium", "silicon"})
    env.game.resource_names = sorted(resources)

    base_caves = BiomeCavesParams(fill_prob=0.45, steps=4, birth_limit=5, death_limit=3)

    zone_count = max(6, _scale_int(12, scale, 6, 40))

    sanctum_size = max(10, _scale_int(64, scale, 18, min(width, height) - 20))
    sanctum_tag = "sanctum.outpost"

    sanctum_outpost_action = ChildrenAction(
        scene=Layout.factory(
            LayoutParams(
                areas=[
                    LayoutArea(
                        width=sanctum_size,
                        height=sanctum_size,
                        placement="center",
                        tag=sanctum_tag,
                    )
                ]
            ),
            children_actions=[
                ChildrenAction(
                    scene=FillArea.factory(FillAreaParams(value="empty")),
                    where=AreaWhere(tags=[sanctum_tag]),
                ),
                ChildrenAction(
                    scene=BiomeCity.factory(
                        BiomeCityParams(
                            pitch=_scale_int(8, scale, 5, 16),
                            road_width=_scale_int(2, scale, 1, 4),
                            jitter=_scale_int(1, scale, 0, 3),
                            place_prob=0.7,
                        )
                    ),
                    where=AreaWhere(tags=[sanctum_tag]),
                    order_by="first",
                    limit=1,
                ),
                ChildrenAction(
                    scene=BaseHub.factory(
                        BaseHubParams(
                            assembler_object="altar",
                            include_inner_wall=True,
                            corner_objects=[
                                "carbon_ex_dep",
                                "oxygen_ex_dep",
                                "germanium_ex_dep",
                                "silicon_ex_dep",
                            ],
                        )
                    ),
                    where=AreaWhere(tags=[sanctum_tag]),
                    order_by="last",
                    limit=1,
                ),
            ],
        ),
        where="full",
        order_by="first",
        lock="sanctum.outpost",
        limit=1,
    )

    sanctum_center_action = ChildrenAction(
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
                    scene=UniformExtractorScene.factory(
                        UniformExtractorParams(
                            target_coverage=min(0.05, max(0.003, 0.003 * scale)),
                            jitter=0,
                            clear_existing=True,
                            extractor_names=[
                                "carbon_extractor",
                                "oxygen_extractor",
                                "germanium_extractor",
                                "silicon_extractor",
                                "charger",
                            ],
                        )
                    ),
                    where=AreaWhere(tags=["sanctum.center"]),
                    order_by="first",
                    lock="sanctum_extractors",
                    limit=1,
                ),
                ChildrenAction(
                    scene=BaseHub.factory(
                        BaseHubParams(
                            assembler_object="altar",
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
                ),
            ],
        ),
        where="full",
        order_by="last",
        lock="sanctum",
        limit=1,
    )

    zone_actions_primary = _build_zone_actions(
        rng,
        scale,
        max(8, int(zone_count * 1.5)),
        width,
        height,
    )
    zone_actions_secondary = _build_zone_actions(
        rng,
        scale * 0.85,
        max(6, zone_count),
        width,
        height,
    )
    zone_actions_tertiary = _build_zone_actions(
        rng,
        scale * 0.7,
        max(4, zone_count // 2),
        width,
        height,
    )
    dungeon_actions = _build_dungeon_actions(
        rng,
        scale,
        max(6, zone_count),
        width,
        height,
    )

    map_children: list[ChildrenAction] = [
        ChildrenAction(
            scene=BSPLayout.factory(
                BSPLayoutParams(area_count=max(10, zone_count * 2)),
                children_actions=zone_actions_primary,
            ),
            where="full",
            order_by="first",
            limit=1,
        ),
        ChildrenAction(
            scene=BSPLayout.factory(
                BSPLayoutParams(area_count=max(12, zone_count + zone_count // 2)),
                children_actions=zone_actions_secondary,
            ),
            where="full",
            order_by="first",
            limit=1,
        ),
        ChildrenAction(
            scene=BSPLayout.factory(
                BSPLayoutParams(area_count=max(8, zone_count)),
                children_actions=zone_actions_tertiary,
            ),
            where="full",
            order_by="first",
            limit=1,
        ),
        ChildrenAction(
            scene=BSPLayout.factory(
                BSPLayoutParams(area_count=max(12, zone_count + zone_count // 2)),
                children_actions=dungeon_actions,
            ),
            where="full",
            order_by="first",
            limit=1,
        ),
        ChildrenAction(
            scene=UniformExtractorScene.factory(_uniform_extractor_params(scale)),
            where="full",
            order_by="last",
            lock="resources",
            limit=1,
        ),
    ]

    map_children.extend([sanctum_outpost_action, sanctum_center_action])

    if max(width, height) <= 400:
        map_children.append(
            ChildrenAction(
                scene=MakeConnected.factory(MakeConnectedParams()),
                where="full",
                order_by="last",
                lock="connect",
                limit=1,
            )
        )

    env.game.map_builder = MapGen.Config(
        width=width,
        height=height,
        root=BiomeCaves.factory(base_caves, children_actions=map_children),
    )

    return env


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
        width=11,
        height=11,
        root=Layout.factory(
            LayoutParams(
                areas=[
                    LayoutArea(
                        width=11,
                        height=11,
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
    width: int = 100, height: int = 100
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
                ChildrenAction(
                    scene=Layout.factory(
                        LayoutParams(
                            areas=[
                                LayoutArea(
                                    width=64,
                                    height=64,
                                    placement="center",
                                    tag="sanctum.outpost",
                                )
                            ]
                        ),
                        children_actions=[
                            ChildrenAction(
                                scene=BiomeCity.factory(
                                    BiomeCityParams(
                                        pitch=8,
                                        road_width=2,
                                        jitter=1,
                                        place_prob=0.7,
                                    )
                                ),
                                where=AreaWhere(tags=["sanctum.outpost"]),
                                order_by="first",
                                limit=1,
                            ),
                            ChildrenAction(
                                scene=BaseHub.factory(
                                    BaseHubParams(
                                        assembler_object="altar",
                                        include_inner_wall=True,
                                        corner_objects=[
                                            "carbon_ex_dep",  # TL
                                            "oxygen_ex_dep",  # TR
                                            "germanium_ex_dep",  # BL
                                            "silicon_ex_dep",  # BR
                                        ],
                                    )
                                ),
                                where=AreaWhere(tags=["sanctum.outpost"]),
                                order_by="first",
                                limit=1,
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
                ChildrenAction(
                    scene=UniformExtractorScene.factory(
                        UniformExtractorParams(
                            target_coverage=0.006,
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
                    lock="resources",
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
                                scene=UniformExtractorScene.factory(
                                    UniformExtractorParams(
                                        target_coverage=0.003,
                                        jitter=0,
                                        clear_existing=True,
                                        extractor_names=[
                                            "carbon_extractor",
                                            "oxygen_extractor",
                                            "germanium_extractor",
                                            "silicon_extractor",
                                            "charger",
                                        ],
                                    )
                                ),
                                where=AreaWhere(tags=["sanctum.center"]),
                                order_by="first",
                                lock="sanctum_extractors",
                                limit=1,
                            ),
                            ChildrenAction(
                                scene=BaseHub.factory(
                                    BaseHubParams(
                                        assembler_object="altar",
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
                            ),
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
                                        assembler_object="assembler",
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
                                        assembler_object="altar",
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

    scalable_astroid = make_scalable_astroid(width=width, height=height)

    extractor_showcase = make_extractor_showcase()

    return (
        env,
        desert_noise,
        city,
        caves,
        forest,
        bsp_dungeon,
        radial_maze,
        scalable_astroid,
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
                                        assembler_object="altar",
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
        scalable_astroid,
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
        SimulationConfig(
            suite="biomes",
            name="scalable_astroid",
            env=scalable_astroid,
        ),
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
