from typing import Any, Literal, cast

import numpy as np

from mettagrid.map_builder.map_builder import MapBuilderConfig
from mettagrid.mapgen.area import AreaWhere
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.random.int import IntConstantDistribution
from mettagrid.mapgen.scene import ChildrenAction, SceneConfig
from mettagrid.mapgen.scenes.base_hub import BaseHub
from mettagrid.mapgen.scenes.biome_caves import BiomeCaves, BiomeCavesConfig
from mettagrid.mapgen.scenes.biome_city import BiomeCity, BiomeCityConfig
from mettagrid.mapgen.scenes.biome_desert import BiomeDesert, BiomeDesertConfig
from mettagrid.mapgen.scenes.biome_forest import BiomeForest, BiomeForestConfig
from mettagrid.mapgen.scenes.bounded_layout import BoundedLayout
from mettagrid.mapgen.scenes.bsp import BSP, BSPLayout
from mettagrid.mapgen.scenes.make_connected import MakeConnected
from mettagrid.mapgen.scenes.maze import Maze
from mettagrid.mapgen.scenes.radial_maze import RadialMaze
from mettagrid.mapgen.scenes.random_scene import RandomScene, RandomSceneCandidate
from mettagrid.mapgen.scenes.uniform_extractors import UniformExtractorScene

HubBundle = Literal["chests", "extractors", "none", "custom"]


def _normalize_bundle(value: str | None, default: HubBundle) -> HubBundle:
    if value in {"chests", "extractors", "none", "custom"}:
        return cast(HubBundle, value)
    return default


def make_machina_procedural_map_builder(
    num_cogs: int,
    *,
    width: int = 100,
    height: int = 100,
    seed: int | None = None,
    base_biome: str = "caves",
    base_biome_config: dict[str, Any] | None = None,
    extractor_coverage: float = 0.01,
    extractors: dict[str, float] | None = None,
    hub_variant: str | None = None,
    hub_corner_bundle: str | None = None,
    hub_cross_bundle: str | None = None,
    hub_cross_distance: int | None = None,
    biome_weights: dict[str, float] | None = None,
    dungeon_weights: dict[str, float] | None = None,
    biome_count: int | None = None,
    dungeon_count: int | None = None,
    density_scale: float = 1.0,
    max_biome_zone_fraction: float = 0.35,
    max_dungeon_zone_fraction: float = 0.25,
) -> MapBuilderConfig:
    def _autoscale_zone_counts(
        w: int,
        h: int,
        *,
        biome_density: float = 1.0,
        dungeon_density: float = 1.0,
    ) -> tuple[int, int]:
        """Compute reasonable zone counts from available area.

        Heuristic: target approx 6 biome zones and 6–8 dungeon zones on a 100x100 map,
        scaled linearly by area and density.
        """
        area = max(1, w * h)
        # Tunable divisors chosen to produce ~6 zones on 10_000 area by default
        biome_divisor = max(800, int(1600 / max(0.1, biome_density)))
        dungeon_divisor = max(800, int(1500 / max(0.1, dungeon_density)))
        biomes = max(3, min(48, area // biome_divisor))
        dungeons = max(3, min(48, area // dungeon_divisor))
        return int(biomes), int(dungeons)

    biome_map: dict[str, tuple[type, type]] = {
        "caves": (BiomeCaves, BiomeCavesConfig),
        "forest": (BiomeForest, BiomeForestConfig),
        "desert": (BiomeDesert, BiomeDesertConfig),
        "city": (BiomeCity, BiomeCityConfig),
    }

    if base_biome not in biome_map:
        raise ValueError(f"Unknown base_biome '{base_biome}'. Valid: {sorted(biome_map.keys())}")

    _, ConfigModel = biome_map[base_biome]
    base_cfg = ConfigModel.model_validate(base_biome_config or {})

    default_extractors = {
        "chest": 0.0,
        "charger": 0.6,
        "germanium_extractor": 0.6,
        "silicon_extractor": 0.3,
        "oxygen_extractor": 0.3,
        "carbon_extractor": 0.3,
    }

    extractor_config = extractors or default_extractors
    extractor_names_final = list(extractor_config.keys())
    extractor_weights_final = extractor_config

    # Optional layered biomes via BSPLayout
    # Autoscale counts based on available area if not explicitly provided
    if biome_count is None or dungeon_count is None:
        auto_biomes, auto_dungeons = _autoscale_zone_counts(
            width, height, biome_density=density_scale, dungeon_density=density_scale
        )
        biome_count = auto_biomes if biome_count is None else biome_count
        dungeon_count = auto_dungeons if dungeon_count is None else dungeon_count

    # Enforce upper bounds on per-zone footprint by increasing counts if needed
    def _min_count_for_fraction(frac: float) -> int:
        if frac <= 0:
            return 1
        return int(np.ceil(1.0 / min(0.9, max(0.02, float(frac)))))

    biome_count = max(int(biome_count), _min_count_for_fraction(max_biome_zone_fraction))
    dungeon_count = max(int(dungeon_count), _min_count_for_fraction(max_dungeon_zone_fraction))

    def _make_biome_candidates(weights: dict[str, float] | None) -> list[RandomSceneCandidate]:
        defaults = {"caves": 1.0, "forest": 1.0, "desert": 1.0, "city": 1.0}
        w = {**defaults, **(weights or {})}
        cands: list[RandomSceneCandidate] = []
        if w.get("caves", 0) > 0:
            cands.append(RandomSceneCandidate(scene=BiomeCaves.Config(), weight=float(w["caves"])))
        if w.get("forest", 0) > 0:
            cands.append(RandomSceneCandidate(scene=BiomeForest.Config(), weight=float(w["forest"])))
        if w.get("desert", 0) > 0:
            cands.append(RandomSceneCandidate(scene=BiomeDesert.Config(), weight=float(w["desert"])))
        if w.get("city", 0) > 0:
            cands.append(RandomSceneCandidate(scene=BiomeCity.Config(), weight=float(w["city"])))
        return cands

    def _make_dungeon_candidates(weights: dict[str, float] | None) -> list[RandomSceneCandidate]:
        defaults = {"bsp": 1.0, "maze": 1.0, "radial": 1.0}
        w = {**defaults, **(weights or {})}
        cands: list[RandomSceneCandidate] = []
        if w.get("bsp", 0) > 0:
            cands.append(
                RandomSceneCandidate(
                    scene=BSP.Config(rooms=4, min_room_size=6, min_room_size_ratio=0.35, max_room_size_ratio=0.75),
                    weight=float(w["bsp"]),
                )
            )
        if w.get("maze", 0) > 0:
            # Prefer thinner corridors for clarity; include both DFS (winding) and Kruskal (grid-like)
            maze_weight = float(w["maze"]) if isinstance(w.get("maze", 0), (int, float)) else 1.0
            cands.append(
                RandomSceneCandidate(
                    scene=Maze.Config(
                        algorithm="dfs",
                        room_size=IntConstantDistribution(value=2),
                        wall_size=IntConstantDistribution(value=1),
                    ),
                    weight=maze_weight * 0.6,
                )
            )
            cands.append(
                RandomSceneCandidate(
                    scene=Maze.Config(
                        algorithm="kruskal",
                        room_size=IntConstantDistribution(value=2),
                        wall_size=IntConstantDistribution(value=1),
                    ),
                    weight=maze_weight * 0.4,
                )
            )
        if w.get("radial", 0) > 0:
            cands.append(
                RandomSceneCandidate(
                    scene=RadialMaze.Config(arms=8, arm_width=2),
                    weight=float(w["radial"]),
                )
            )
        return cands

    # Compute max footprint per feature (size clamp inside zones)
    biome_max_w = max(10, int(min(width * max_biome_zone_fraction, width // 2)))
    biome_max_h = max(10, int(min(height * max_biome_zone_fraction, height // 2)))
    dungeon_max_w = max(10, int(min(width * max_dungeon_zone_fraction, width // 2)))
    dungeon_max_h = max(10, int(min(height * max_dungeon_zone_fraction, height // 2)))

    def _wrap_in_layout(scene_cfg: SceneConfig, tag: str, max_w: int, max_h: int) -> SceneConfig:
        # Use BoundedLayout to clamp to both max_* and current zone size
        return BoundedLayout.Config(
            max_width=max_w,
            max_height=max_h,
            tag=tag,
            children=[
                ChildrenAction(
                    scene=scene_cfg,
                    where=AreaWhere(tags=[tag]),
                    limit=1,
                    order_by="first",
                )
            ],
        )

    biome_layer: ChildrenAction | None = None
    biome_cands = _make_biome_candidates(biome_weights)
    if biome_cands:
        # Fill only a subset of zones to preserve base shell background
        biome_fill_count = max(1, int(biome_count * 0.6))
        # Wrap RandomScene in a clamped Layout so no single biome fills its entire zone
        biome_layer = ChildrenAction(
            scene=BSPLayout.Config(
                area_count=biome_count,
                children=[
                    ChildrenAction(
                        scene=_wrap_in_layout(
                            RandomScene.Config(candidates=biome_cands),
                            tag="biome.zone",
                            max_w=biome_max_w,
                            max_h=biome_max_h,
                        ),
                        where=AreaWhere(tags=["zone"]),
                        order_by="random",
                        limit=biome_fill_count,
                    )
                ],
            ),
            where="full",
            order_by="first",
            limit=1,
        )

    dungeon_layer: ChildrenAction | None = None
    dungeon_cands = _make_dungeon_candidates(dungeon_weights)
    if dungeon_cands:
        # Fill only a subset of zones to preserve base shell background
        dungeon_fill_count = max(1, int(dungeon_count * 0.5))
        # Wrap RandomScene in a clamped Layout so no single dungeon fills its entire zone
        dungeon_layer = ChildrenAction(
            scene=BSPLayout.Config(
                area_count=dungeon_count,
                children=[
                    ChildrenAction(
                        scene=_wrap_in_layout(
                            RandomScene.Config(candidates=dungeon_cands),
                            tag="dungeon.zone",
                            max_w=dungeon_max_w,
                            max_h=dungeon_max_h,
                        ),
                        where=AreaWhere(tags=["zone"]),
                        order_by="random",
                        limit=dungeon_fill_count,
                    )
                ],
            ),
            where="full",
            order_by="first",
            limit=1,
        )

    corner_bundle = _normalize_bundle(hub_corner_bundle, "chests")
    cross_bundle = _normalize_bundle(hub_cross_bundle, "none")

    match (hub_variant or "").lower():
        case "store":
            corner_bundle = "chests"
            cross_bundle = "none"
        case "extractor":
            corner_bundle = "extractors"
            cross_bundle = "none"
        case "both":
            corner_bundle = "chests"
            cross_bundle = "extractors"

    cross_distance = hub_cross_distance if hub_cross_distance is not None else 7

    base_cfg.children = [
        # Optional biome/dungeon layers over the base shell
        *([biome_layer] if biome_layer is not None else []),
        *([dungeon_layer] if dungeon_layer is not None else []),
        # Resources first so connectors will tunnel between them
        ChildrenAction(
            scene=UniformExtractorScene.Config(
                target_coverage=extractor_coverage,
                extractor_names=extractor_names_final,
                extractor_weights=extractor_weights_final,
                clear_existing=False,
            ),
            where="full",
            order_by="last",
            lock="arena.resources",
            limit=1,
        ),
        # Ensure connectivity after resources to connect resource pockets
        ChildrenAction(
            scene=MakeConnected.Config(),
            where="full",
            order_by="last",
            lock="arena.connect",
            limit=1,
        ),
        # Place hub last to keep spawns intact; it self-centers and sizes internally
        ChildrenAction(
            scene=BaseHub.Config(
                spawn_count=num_cogs,
                hub_width=21,
                hub_height=21,
                corner_bundle=corner_bundle or "chests",
                cross_bundle=cross_bundle or "none",
                cross_distance=cross_distance,
            ),
            where="full",
            order_by="last",
            limit=1,
        ),
    ]

    return MapGen.Config(width=width, height=height, instance=base_cfg, seed=seed)
