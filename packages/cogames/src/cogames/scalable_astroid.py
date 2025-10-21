from __future__ import annotations

import math
import random
from typing import Callable, Literal, Optional

from cogames.cogs_vs_clips import stations as cvc_stations
from mettagrid.builder.envs import make_navigation
from mettagrid.config.config import Config
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.random.int import IntConstantDistribution
from mettagrid.mapgen.scene import ChildrenAction, SceneConfig
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
from mettagrid.mapgen.scenes.radial_maze import RadialMaze, RadialMazeParams
from mettagrid.mapgen.scenes.uniform_extractors import (
    UniformExtractorParams,
    UniformExtractorScene,
)
from mettagrid.mapgen.types import AreaWhere

DEFAULT_NUM_AGENTS = 4

MazeAlgorithm = Literal["dfs", "kruskal"]

MAZE_ALGORITHMS: tuple[MazeAlgorithm, MazeAlgorithm] = ("dfs", "kruskal")


class ScalableAstroidParams(Config):
    extractor_coverage: float | None = None
    extractor_names: list[str] | None = None
    extractor_weights: dict[str, float] | None = None
    extractor_padding: int | None = None
    extractor_jitter: int | None = None
    primary_zone_weights: dict[str, float] | None = None
    secondary_zone_weights: dict[str, float] | None = None
    tertiary_zone_weights: dict[str, float] | None = None
    dungeon_zone_weights: dict[str, float] | None = None


def _merge_params(
    default: ScalableAstroidParams,
    override: ScalableAstroidParams | None,
) -> ScalableAstroidParams:
    if override is None:
        return default

    update_data = {key: value for key, value in override.model_dump(exclude_unset=True).items() if value is not None}
    if not update_data:
        return default

    return default.model_copy(update=update_data)


def _compute_scale(width: int, height: int) -> float:
    base_area = 500 * 500
    return math.sqrt((width * height) / base_area)


def _scale_int(base: int, scale: float, minimum: int, maximum: Optional[int] = None) -> int:
    value = max(minimum, int(round(base * scale)))
    if maximum is not None:
        value = min(value, maximum)
    return value


def _rand_tag(prefix: str, rng: random.Random) -> str:
    return f"{prefix}.{rng.randrange(10_000_000)}"


def _add_station_objects(env: MettaGridConfig) -> None:
    objects = env.game.objects
    objects.setdefault("charger", cvc_stations.charger())
    objects.setdefault("carbon_extractor", cvc_stations.carbon_extractor())
    objects.setdefault("oxygen_extractor", cvc_stations.oxygen_extractor())
    objects.setdefault("germanium_extractor", cvc_stations.germanium_extractor())
    objects.setdefault("silicon_extractor", cvc_stations.silicon_extractor())
    objects.setdefault("chest", cvc_stations.chest())
    objects.setdefault("assembler", cvc_stations.assembler())
    objects.setdefault("carbon_ex_dep", cvc_stations.carbon_ex_dep())
    objects.setdefault("oxygen_ex_dep", cvc_stations.oxygen_ex_dep())
    objects.setdefault("germanium_ex_dep", cvc_stations.germanium_ex_dep())
    objects.setdefault("silicon_ex_dep", cvc_stations.silicon_ex_dep())


def _uniform_extractor_params(
    scale: float,
    params: ScalableAstroidParams,
) -> UniformExtractorParams:
    coverage = params.extractor_coverage
    if coverage is None:
        coverage = min(0.15, max(0.003, 0.006 * scale))

    extractor_names = params.extractor_names or [
        "carbon_extractor",
        "oxygen_extractor",
        "germanium_extractor",
        "silicon_extractor",
        "charger",
    ]

    defaults = UniformExtractorParams()

    return UniformExtractorParams(
        target_coverage=coverage,
        rows=defaults.rows,
        cols=defaults.cols,
        jitter=params.extractor_jitter if params.extractor_jitter is not None else defaults.jitter,
        padding=params.extractor_padding if params.extractor_padding is not None else defaults.padding,
        extractor_names=extractor_names,
        extractor_weights=params.extractor_weights,
    )


def _build_zone_actions(
    rng: random.Random,
    scale: float,
    count: int,
    max_width: int,
    max_height: int,
    weights: dict[str, float] | None = None,
) -> list[ChildrenAction]:
    max_feature_width = max(32, max_width // 3)
    max_feature_height = max(32, max_height // 3)

    def _wrap_layout(
        width: int,
        height: int,
        tag: str,
        children: list[ChildrenAction],
    ) -> SceneConfig:
        width = max(12, min(width, max_feature_width))
        height = max(12, min(height, max_feature_height))

        return Layout.factory(
            LayoutParams(
                areas=[
                    LayoutArea(
                        width=width,
                        height=height,
                        placement="center",
                        tag=tag,
                    )
                ]
            ),
            children_actions=[
                ChildrenAction(
                    scene=FillArea.factory(FillAreaParams(value="empty")),
                    where=AreaWhere(tags=[tag]),
                ),
                *children,
            ],
        )

    def zone_desert(randomizer: random.Random, factor: float) -> ChildrenAction:
        outer_tag = _rand_tag("arena.desert", randomizer)
        inner_tag = f"{outer_tag}.radial"
        outer_size = min(_scale_int(90, factor, 30, 220), max_feature_width)
        inner_size = max(14, min(outer_size - 6, _scale_int(30, factor, 14, 160)))

        inner = _wrap_layout(
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
                scene=inner,
                where=AreaWhere(tags=[outer_tag]),
                limit=1,
            ),
        ]

        return ChildrenAction(
            scene=_wrap_layout(outer_size, outer_size, outer_tag, outer_children),
            where=AreaWhere(tags=["zone"]),
            order_by="random",
            lock="arena.zone",
            limit=1,
        )

    def zone_forest(randomizer: random.Random, factor: float) -> ChildrenAction:
        outer_tag = _rand_tag("arena.forest", randomizer)
        inner_tag = f"{outer_tag}.bsp"
        outer_width = min(_scale_int(80, factor, 28, 200), max_feature_width)
        outer_height = min(_scale_int(70, factor, 26, 200), max_feature_height)
        inner_width = max(16, min(outer_width - 6, _scale_int(40, factor, 18, 160)))
        inner_height = max(16, min(outer_height - 6, _scale_int(34, factor, 18, 160)))

        inner = _wrap_layout(
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
                scene=inner,
                where=AreaWhere(tags=[outer_tag]),
                limit=1,
            ),
        ]

        return ChildrenAction(
            scene=_wrap_layout(outer_width, outer_height, outer_tag, outer_children),
            where=AreaWhere(tags=["zone"]),
            order_by="random",
            lock="arena.zone",
            limit=1,
        )

    def zone_city(randomizer: random.Random, factor: float) -> ChildrenAction:
        outer_tag = _rand_tag("arena.city", randomizer)
        inner_tag = f"{outer_tag}.maze"
        outer_width = min(_scale_int(72, factor, 28, 200), max_feature_width)
        outer_height = min(_scale_int(72, factor, 28, 200), max_feature_height)
        inner_size = max(16, min(min(outer_width, outer_height) - 6, _scale_int(40, factor, 18, 160)))

        algorithm = randomizer.choice(MAZE_ALGORITHMS)
        base_room = 3 if algorithm == "dfs" else 4
        room_size = _scale_int(base_room, factor, 2, 8)
        wall_size = min(room_size, _scale_int(2 if algorithm == "kruskal" else 1, factor, 1, 5))

        inner = _wrap_layout(
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
                scene=inner,
                where=AreaWhere(tags=[outer_tag]),
                limit=1,
            ),
        ]

        return ChildrenAction(
            scene=_wrap_layout(outer_width, outer_height, outer_tag, outer_children),
            where=AreaWhere(tags=["zone"]),
            order_by="random",
            lock="arena.zone",
            limit=1,
        )

    def zone_caves(randomizer: random.Random, factor: float) -> ChildrenAction:
        tag = _rand_tag("arena.caves", randomizer)
        width = min(_scale_int(70, factor, 28, 220), max_feature_width)
        height = min(_scale_int(60, factor, 28, 220), max_feature_height)

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
                where=AreaWhere(tags=[tag]),
            )
        ]

        return ChildrenAction(
            scene=_wrap_layout(width, height, tag, outer_children),
            where=AreaWhere(tags=["zone"]),
            order_by="random",
            lock="arena.zone",
            limit=1,
        )

    def zone_bsp(randomizer: random.Random, factor: float) -> ChildrenAction:
        tag = _rand_tag("arena.bsp", randomizer)
        width = min(_scale_int(70, factor, 28, 220), max_feature_width)
        height = min(_scale_int(60, factor, 28, 220), max_feature_height)
        params = BSPParams(
            rooms=_scale_int(14, factor, 6, 40),
            min_room_size=_scale_int(4, factor, 3, 12),
            min_room_size_ratio=min(0.6, max(0.25, 0.35 * factor)),
            max_room_size_ratio=min(0.9, max(0.5, 0.7 * factor)),
        )

        return ChildrenAction(
            scene=_wrap_layout(
                width,
                height,
                tag,
                [
                    ChildrenAction(
                        scene=BSP.factory(params),
                        where=AreaWhere(tags=[tag]),
                        limit=1,
                    )
                ],
            ),
            where=AreaWhere(tags=["zone"]),
            order_by="random",
            lock="arena.zone",
            limit=1,
        )

    def zone_radial(randomizer: random.Random, factor: float) -> ChildrenAction:
        tag = _rand_tag("arena.radial", randomizer)
        width = min(_scale_int(72, factor, 28, 220), max_feature_width)
        height = min(_scale_int(72, factor, 28, 220), max_feature_height)
        params = RadialMazeParams(
            arms=_scale_int(12, factor, 5, 22),
            arm_width=_scale_int(3, factor, 1, 6),
            arm_length=_scale_int(36, factor, 14, 96),
            fill_background=False,
        )

        return ChildrenAction(
            scene=_wrap_layout(
                width,
                height,
                tag,
                [
                    ChildrenAction(
                        scene=RadialMaze.factory(params),
                        where=AreaWhere(tags=[tag]),
                        limit=1,
                    )
                ],
            ),
            where=AreaWhere(tags=["zone"]),
            order_by="random",
            lock="arena.zone",
            limit=1,
        )

    def zone_maze(randomizer: random.Random, factor: float) -> ChildrenAction:
        tag = _rand_tag("arena.maze", randomizer)
        width = min(_scale_int(60, factor, 24, 180), max_feature_width)
        height = min(_scale_int(60, factor, 24, 180), max_feature_height)
        room_size = _scale_int(3, factor, 2, 6)
        wall_size = min(room_size, _scale_int(2, factor, 1, 4))

        return ChildrenAction(
            scene=_wrap_layout(
                width,
                height,
                tag,
                [
                    ChildrenAction(
                        scene=Maze.factory(
                            MazeParams(
                                algorithm=randomizer.choice(MAZE_ALGORITHMS),
                                room_size=IntConstantDistribution(value=room_size),
                                wall_size=IntConstantDistribution(value=wall_size),
                            )
                        ),
                        where=AreaWhere(tags=[tag]),
                        limit=1,
                    )
                ],
            ),
            where=AreaWhere(tags=["zone"]),
            order_by="random",
            lock="arena.zone",
            limit=1,
        )

    def zone_city_dense(randomizer: random.Random, factor: float) -> ChildrenAction:
        tag = _rand_tag("arena.city_dense", randomizer)
        width = min(_scale_int(60, factor, 24, 200), max_feature_width)
        height = min(_scale_int(60, factor, 24, 200), max_feature_height)
        params = BiomeCityParams(
            pitch=_scale_int(12, factor, 6, 22),
            road_width=_scale_int(3, factor, 1, 6),
            jitter=_scale_int(3, factor, 0, 6),
            place_prob=randomizer.uniform(0.7, 0.98),
        )

        return ChildrenAction(
            scene=_wrap_layout(
                width,
                height,
                tag,
                [
                    ChildrenAction(
                        scene=BiomeCity.factory(params),
                        where=AreaWhere(tags=[tag]),
                    )
                ],
            ),
            where=AreaWhere(tags=["zone"]),
            order_by="random",
            lock="arena.zone",
            limit=1,
        )

    builder_map: dict[str, Callable[[random.Random, float], ChildrenAction]] = {
        "desert": zone_desert,
        "forest": zone_forest,
        "city": zone_city,
        "caves": zone_caves,
        "bsp": zone_bsp,
        "radial": zone_radial,
        "maze": zone_maze,
        "city_dense": zone_city_dense,
    }

    names: list[str]
    probabilities: list[float]
    if weights:
        filtered = [(name, weight) for name, weight in weights.items() if name in builder_map and weight > 0]
        if not filtered:
            filtered = list(builder_map.items())
            names = [name for name, _ in filtered]
            probabilities = [1.0 for _ in filtered]
        else:
            names = [name for name, _ in filtered]
            probabilities = [weight for _, weight in filtered]
    else:
        names = list(builder_map.keys())
        probabilities = [1.0 for _ in names]

    total = sum(probabilities)
    if total <= 0:
        probabilities = [1.0 for _ in names]
        total = float(len(probabilities))

    probabilities = [weight / total for weight in probabilities]

    return [builder_map[rng.choices(names, probabilities)[0]](rng, scale) for _ in range(count)]


def _build_dungeon_actions(
    rng: random.Random,
    scale: float,
    count: int,
    max_width: int,
    max_height: int,
    weights: dict[str, float] | None = None,
) -> list[ChildrenAction]:
    max_dungeon_width = max(20, max_width // 4)
    max_dungeon_height = max(20, max_height // 4)

    def _wrap_dungeon(
        width: int,
        height: int,
        tag: str,
        inner: ChildrenAction,
    ) -> SceneConfig:
        width = max(10, min(width, max_dungeon_width))
        height = max(10, min(height, max_dungeon_height))

        return Layout.factory(
            LayoutParams(areas=[LayoutArea(width=width, height=height, placement="center", tag=tag)]),
            children_actions=[
                ChildrenAction(
                    scene=FillArea.factory(FillAreaParams(value="empty")),
                    where=AreaWhere(tags=[tag]),
                ),
                inner,
            ],
        )

    def dungeon_bsp(randomizer: random.Random, factor: float) -> ChildrenAction:
        tag = _rand_tag("arena.dungeon.bsp", randomizer)
        params = BSPParams(
            rooms=_scale_int(10, factor, 6, 26),
            min_room_size=_scale_int(4, factor, 3, 10),
            min_room_size_ratio=0.35,
            max_room_size_ratio=0.7,
        )
        inner = ChildrenAction(
            scene=BSP.factory(params),
            where=AreaWhere(tags=[tag]),
            limit=1,
        )
        return ChildrenAction(
            scene=_wrap_dungeon(
                _scale_int(48, factor, 20, 160),
                _scale_int(36, factor, 20, 140),
                tag,
                inner,
            ),
            where=AreaWhere(tags=["zone"]),
            order_by="random",
            lock="arena.zone.dungeon",
            limit=1,
        )

    def dungeon_maze(randomizer: random.Random, factor: float) -> ChildrenAction:
        tag = _rand_tag("arena.dungeon.maze", randomizer)
        room_size = _scale_int(3, factor, 2, 6)
        wall_size = max(1, min(room_size - 1, _scale_int(2, factor, 1, 4)))
        params = MazeParams(
            algorithm=randomizer.choice(MAZE_ALGORITHMS),
            room_size=IntConstantDistribution(value=room_size),
            wall_size=IntConstantDistribution(value=wall_size),
        )
        inner = ChildrenAction(
            scene=Maze.factory(params),
            where=AreaWhere(tags=[tag]),
            limit=1,
        )
        return ChildrenAction(
            scene=_wrap_dungeon(
                _scale_int(44, factor, 18, 150),
                _scale_int(44, factor, 18, 150),
                tag,
                inner,
            ),
            where=AreaWhere(tags=["zone"]),
            order_by="random",
            lock="arena.zone.dungeon",
            limit=1,
        )

    def dungeon_radial(randomizer: random.Random, factor: float) -> ChildrenAction:
        tag = _rand_tag("arena.dungeon.radial", randomizer)
        params = RadialMazeParams(
            arms=_scale_int(8, factor, 4, 16),
            arm_width=_scale_int(2, factor, 1, 4),
            arm_length=_scale_int(24, factor, 10, 48),
            fill_background=False,
        )
        inner = ChildrenAction(
            scene=RadialMaze.factory(params),
            where=AreaWhere(tags=[tag]),
            limit=1,
        )
        return ChildrenAction(
            scene=_wrap_dungeon(
                _scale_int(52, factor, 20, 160),
                _scale_int(52, factor, 20, 160),
                tag,
                inner,
            ),
            where=AreaWhere(tags=["zone"]),
            order_by="random",
            lock="arena.zone.dungeon",
            limit=1,
        )

    def dungeon_dense(randomizer: random.Random, factor: float) -> ChildrenAction:
        tag = _rand_tag("arena.dungeon.dense", randomizer)
        room_size = _scale_int(2, factor, 2, 5)
        inner = ChildrenAction(
            scene=Maze.factory(
                MazeParams(
                    algorithm="dfs",
                    room_size=IntConstantDistribution(value=room_size),
                    wall_size=IntConstantDistribution(value=max(1, room_size - 1)),
                )
            ),
            where=AreaWhere(tags=[tag]),
            limit=1,
        )
        return ChildrenAction(
            scene=_wrap_dungeon(
                _scale_int(36, factor, 18, 120),
                _scale_int(36, factor, 18, 120),
                tag,
                inner,
            ),
            where=AreaWhere(tags=["zone"]),
            order_by="random",
            lock="arena.zone.dungeon",
            limit=1,
        )

    builder_map: dict[str, Callable[[random.Random, float], ChildrenAction]] = {
        "bsp": dungeon_bsp,
        "maze": dungeon_maze,
        "radial": dungeon_radial,
        "dense": dungeon_dense,
    }

    names: list[str]
    probabilities: list[float]
    if weights:
        filtered = [(name, weight) for name, weight in weights.items() if name in builder_map and weight > 0]
        if not filtered:
            filtered = list(builder_map.items())
            names = [name for name, _ in filtered]
            probabilities = [1.0 for _ in filtered]
        else:
            names = [name for name, _ in filtered]
            probabilities = [weight for _, weight in filtered]
    else:
        names = list(builder_map.keys())
        probabilities = [1.0 for _ in names]

    total = sum(probabilities)
    if total <= 0:
        probabilities = [1.0 for _ in names]
        total = float(len(probabilities))

    probabilities = [weight / total for weight in probabilities]

    return [builder_map[rng.choices(names, probabilities)[0]](rng, scale) for _ in range(count)]


def make_scalable_arena(
    width: int = 300,
    height: int = 300,
    num_agents: int = DEFAULT_NUM_AGENTS,
    seed: Optional[int] = None,
    params: ScalableAstroidParams | None = None,
) -> MettaGridConfig:
    """Generate a scalable mixed-biome arena with extractor stations."""

    rng = random.Random(seed)
    scale = _compute_scale(width, height)

    env = make_navigation(num_agents=num_agents)
    _add_station_objects(env)

    resources = set(env.game.resource_names)
    resources.update(cvc_stations.resources)
    env.game.resource_names = sorted(resources)

    params = _merge_params(ScalableAstroidParams(), params)

    base_caves = BiomeCavesParams(fill_prob=0.45, steps=4, birth_limit=5, death_limit=3)

    zone_count = max(6, _scale_int(12, scale, 6, 40))

    sanctum_size = max(15, _scale_int(64, scale, 20, min(width, height) - 20))
    sanctum_tag = "arena.sanctum"

    sanctum_outpost = ChildrenAction(
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
                    limit=1,
                ),
                ChildrenAction(
                    scene=BaseHub.factory(
                        BaseHubParams(
                            assembler_object="assembler",
                            include_inner_wall=True,
                            corner_objects=[
                                "carbon_extractor",
                                "oxygen_extractor",
                                "germanium_extractor",
                                "silicon_extractor",
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
        lock="arena.sanctum",
        limit=1,
    )

    sanctum_core = ChildrenAction(
        scene=Layout.factory(
            LayoutParams(areas=[LayoutArea(width=21, height=21, placement="center", tag="arena.sanctum.core")]),
            children_actions=[
                ChildrenAction(
                    scene=FillArea.factory(FillAreaParams(value="empty")),
                    where=AreaWhere(tags=["arena.sanctum.core"]),
                ),
                ChildrenAction(
                    scene=UniformExtractorScene.factory(
                        UniformExtractorParams(
                            target_coverage=min(0.05, 0.02 * scale + 0.01),
                            jitter=0,
                            clear_existing=True,
                            extractor_names=[
                                "carbon_extractor",
                                "oxygen_extractor",
                                "germanium_extractor",
                                "silicon_extractor",
                                "charger",
                                "carbon_ex_dep",
                                "oxygen_ex_dep",
                                "germanium_ex_dep",
                                "silicon_ex_dep",
                            ],
                        )
                    ),
                    where=AreaWhere(tags=["arena.sanctum.core"]),
                    limit=1,
                ),
            ],
        ),
        where="full",
        order_by="last",
        lock="arena.sanctum",
        limit=1,
    )

    zone_actions_primary = _build_zone_actions(
        rng,
        scale,
        max(8, int(zone_count * 1.5)),
        width,
        height,
        weights=params.primary_zone_weights,
    )
    zone_actions_secondary = _build_zone_actions(
        rng,
        scale * 0.85,
        max(6, zone_count),
        width,
        height,
        weights=params.secondary_zone_weights,
    )
    zone_actions_tertiary = _build_zone_actions(
        rng,
        scale * 0.7,
        max(4, zone_count // 2),
        width,
        height,
        weights=params.tertiary_zone_weights,
    )
    dungeon_actions = _build_dungeon_actions(
        rng,
        scale,
        max(6, zone_count),
        width,
        height,
        weights=params.dungeon_zone_weights,
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
            scene=UniformExtractorScene.factory(_uniform_extractor_params(scale, params)),
            where="full",
            order_by="last",
            lock="arena.resources",
            limit=1,
        ),
        sanctum_outpost,
        sanctum_core,
    ]

    if max(width, height) <= 400:
        map_children.append(
            ChildrenAction(
                scene=MakeConnected.factory(MakeConnectedParams()),
                where="full",
                order_by="last",
                lock="arena.connect",
                limit=1,
            )
        )

    env.game.map_builder = MapGen.Config(
        width=width,
        height=height,
        root=BiomeCaves.factory(
            base_caves,
            children_actions=map_children,
        ),
    )

    return env


def scalable_astroid_config(
    width: int = 300,
    height: int = 300,
    num_agents: int = DEFAULT_NUM_AGENTS,
    seed: Optional[int] = None,
) -> MettaGridConfig:
    """Alias for :func:`make_scalable_arena` for backwards compatibility."""

    return make_scalable_arena(width=width, height=height, num_agents=num_agents, seed=seed)
