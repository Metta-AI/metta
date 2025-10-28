from typing import Any, Literal, cast

import numpy as np
from pydantic import BaseModel

from mettagrid.map_builder.map_builder import MapBuilderConfig
from mettagrid.mapgen.area import AreaWhere
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.random.int import IntConstantDistribution
from mettagrid.mapgen.scene import ChildrenAction, GridTransform, SceneConfig
from mettagrid.mapgen.scenes.base_hub import BaseHub
from mettagrid.mapgen.scenes.biome_caves import BiomeCaves, BiomeCavesConfig
from mettagrid.mapgen.scenes.biome_city import BiomeCity, BiomeCityConfig
from mettagrid.mapgen.scenes.biome_desert import BiomeDesert, BiomeDesertConfig
from mettagrid.mapgen.scenes.biome_forest import BiomeForest, BiomeForestConfig
from mettagrid.mapgen.scenes.bounded_layout import BoundedLayout
from mettagrid.mapgen.scenes.bsp import BSP, BSPLayout
from mettagrid.mapgen.scenes.building_distributions import DistributionConfig, UniformExtractorScene
from mettagrid.mapgen.scenes.make_connected import MakeConnected
from mettagrid.mapgen.scenes.maze import Maze
from mettagrid.mapgen.scenes.radial_maze import RadialMaze
from mettagrid.mapgen.scenes.random_scene import RandomScene, RandomSceneCandidate

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
    extractor_names: list[str] | None = None,
    extractor_weights: dict[str, float] | None = None,
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
    distribution: dict[str, Any] | DistributionConfig | None = None,
    building_distributions: dict[str, dict[str, Any] | DistributionConfig] | None = None,
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

    biome_map: dict[str, tuple[type[Any], type[BaseModel]]] = {
        "caves": (BiomeCaves, BiomeCavesConfig),
        "forest": (BiomeForest, BiomeForestConfig),
        "desert": (BiomeDesert, BiomeDesertConfig),
        "city": (BiomeCity, BiomeCityConfig),
    }

    if base_biome not in biome_map:
        raise ValueError(f"Unknown base_biome '{base_biome}'. Valid: {sorted(biome_map.keys())}")

    ConfigModelType: type[BaseModel] = biome_map[base_biome][1]
    base_cfg = cast(SceneConfig, ConfigModelType.model_validate(base_biome_config or {}))

    default_extractors = {
        "chest": 0.0,
        "charger": 0.6,
        "germanium_extractor": 0.6,
        "silicon_extractor": 0.3,
        "oxygen_extractor": 0.3,
        "carbon_extractor": 0.3,
        # TO DO: these are the default weights, can add distribution preset here
    }

    # Legacy "extractors" dict takes precedence if provided
    extractor_config = extractors or default_extractors
    names = extractor_names or list(extractor_config.keys())
    weights = extractor_weights or extractor_config

    extractor_names_final = list(dict.fromkeys(names))
    extractor_weights_final = {
        name: weights.get(name, default_extractors.get(name, 1.0)) for name in extractor_names_final
    }

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
            cands.append(RandomSceneCandidate(scene=BiomeCavesConfig(), weight=float(w["caves"])))
        if w.get("forest", 0) > 0:
            cands.append(RandomSceneCandidate(scene=BiomeForestConfig(), weight=float(w["forest"])))
        if w.get("desert", 0) > 0:
            cands.append(RandomSceneCandidate(scene=BiomeDesertConfig(), weight=float(w["desert"])))
        if w.get("city", 0) > 0:
            cands.append(RandomSceneCandidate(scene=BiomeCityConfig(), weight=float(w["city"])))
        return cands

    def _make_dungeon_candidates(weights: dict[str, float] | None) -> list[RandomSceneCandidate]:
        defaults = {"bsp": 1.0, "maze": 1.0, "radial": 1.0}
        w = {**defaults, **(weights or {})}
        cands: list[RandomSceneCandidate] = []
        if w.get("bsp", 0) > 0:
            cands.append(
                RandomSceneCandidate(
                    scene=cast(Any, BSP).Config(
                        rooms=4,
                        min_room_size=6,
                        min_room_size_ratio=0.35,
                        max_room_size_ratio=0.75,
                    ),
                    weight=float(w["bsp"]),
                )
            )
        if w.get("maze", 0) > 0:
            # Prefer thinner corridors for clarity; include both DFS (winding) and Kruskal (grid-like)
            maze_weight = float(w["maze"]) if isinstance(w.get("maze", 0), (int, float)) else 1.0
            cands.append(
                RandomSceneCandidate(
                    scene=cast(Any, Maze).Config(
                        algorithm="dfs",
                        room_size=IntConstantDistribution(value=2),
                        wall_size=IntConstantDistribution(value=1),
                    ),
                    weight=maze_weight * 0.6,
                )
            )
            cands.append(
                RandomSceneCandidate(
                    scene=cast(Any, Maze).Config(
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
                    scene=cast(Any, RadialMaze).Config(arms=8, arm_width=2),
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
        return cast(Any, BoundedLayout).Config(
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
            scene=cast(Any, BSPLayout).Config(
                area_count=biome_count,
                children=[
                    ChildrenAction(
                        scene=_wrap_in_layout(
                            cast(Any, RandomScene).Config(candidates=biome_cands),
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
            scene=cast(Any, BSPLayout).Config(
                area_count=dungeon_count,
                children=[
                    ChildrenAction(
                        scene=_wrap_in_layout(
                            cast(Any, RandomScene).Config(candidates=dungeon_cands),
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

    cross_distance = hub_cross_distance if hub_cross_distance is not None else 7

    # Process distribution configurations
    dist_config = (
        distribution if isinstance(distribution, DistributionConfig) else DistributionConfig(**(distribution or {}))
    )

    building_dist_configs: dict[str, DistributionConfig] | None = None
    if building_distributions:
        building_dist_configs = {}
        for name, config in building_distributions.items():
            if isinstance(config, DistributionConfig):
                building_dist_configs[name] = config
            else:
                building_dist_configs[name] = DistributionConfig(**config)

    base_cfg.children = [
        # Optional biome/dungeon layers over the base shell
        *([biome_layer] if biome_layer is not None else []),
        *([dungeon_layer] if dungeon_layer is not None else []),
        # Resources first so connectors will tunnel between them
        ChildrenAction(
            scene=cast(Any, UniformExtractorScene).Config(
                target_coverage=extractor_coverage,
                extractor_names=extractor_names_final,
                extractor_weights=extractor_weights_final,
                clear_existing=False,
                distribution=dist_config,
                building_distributions=building_dist_configs,
            ),
            where="full",
            order_by="last",
            lock="arena.resources",
            limit=1,
        ),
        # Ensure connectivity after resources to connect resource pockets
        ChildrenAction(
            scene=cast(Any, MakeConnected).Config(),
            where="full",
            order_by="last",
            lock="arena.connect",
            limit=1,
        ),
        # Place hub last to keep spawns intact; it self-centers and sizes internally
        ChildrenAction(
            scene=cast(Any, BaseHub).Config(
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
        # Final connectivity sweep to ensure all empty regions are connected end-to-end
        ChildrenAction(
            scene=cast(Any, MakeConnected).Config(),
            where="full",
            order_by="last",
            lock="arena.connect.final",
            limit=1,
        ),
    ]

    return MapGen.Config(width=width, height=height, instance=base_cfg, seed=seed)


def make_hub_only_map_builder(
    num_cogs: int,
    *,
    width: int = 21,
    height: int = 21,
    seed: int | None = None,
    layout: Literal["default", "tight"] = "default",
    corner_bundle: HubBundle = "none",
    cross_bundle: HubBundle = "none",
    cross_distance: int = 7,
    corner_objects: list[str] | None = None,
    cross_objects: list[str] | None = None,
    transforms: list[GridTransform] | None = None,
) -> MapBuilderConfig:
    """Build a hub-only map using RandomScene over BaseHub with random transforms.

    Notes:
    - If corner_objects is provided (len==4), BaseHub will use that set directly.
    - corner_bundle/cross_bundle can be "none" | "chests" | "extractors".
    - When both objects and bundle are provided, objects win (per BaseHub logic).
    """

    # Default transform set to randomize orientation/reflection
    transform_set = transforms or [
        GridTransform.IDENTITY,
        GridTransform.ROT_90,
        GridTransform.ROT_180,
        GridTransform.ROT_270,
        GridTransform.FLIP_H,
        GridTransform.FLIP_V,
        GridTransform.TRANSPOSE,
        GridTransform.TRANSPOSE_ALT,
    ]

    base_kwargs: dict[str, Any] = {
        "spawn_count": num_cogs,
        "hub_width": width,
        "hub_height": height,
        "include_inner_wall": True,
        "layout": layout,
        "corner_bundle": corner_bundle,
        "cross_bundle": cross_bundle,
        "cross_distance": cross_distance,
    }
    if corner_objects is not None:
        base_kwargs["corner_objects"] = list(corner_objects)
    if cross_objects is not None:
        base_kwargs["cross_objects"] = list(cross_objects)

    candidates = [
        RandomSceneCandidate(scene=cast(Any, BaseHub).Config(**base_kwargs, transform=t), weight=1.0)
        for t in transform_set
    ]

    return cast(Any, MapGen).Config(
        width=width,
        height=height,
        seed=seed,
        instance=cast(Any, RandomScene).Config(candidates=candidates),
    )


def apply_hub_overrides_to_builder(
    builder: MapBuilderConfig,
    *,
    num_cogs: int,
    overrides: dict[str, Any] | None = None,
) -> MapBuilderConfig:
    """If builder is a hub-only MapGen with BaseHub scenes, apply corner/cross overrides.

    Best-effort: if structure is not recognized, return builder unchanged.
    """
    try:
        if not isinstance(builder, MapGen.Config):
            return builder

        width = getattr(builder, "width", None) or 21
        height = getattr(builder, "height", None) or 21

        inst = getattr(builder, "instance", None)
        if isinstance(inst, cast(Any, RandomScene).Config):
            # Verify candidates are BaseHub configs; extract transforms
            transforms: list[GridTransform] = []
            basehub_seen = False
            existing_corner_bundle: HubBundle | None = None
            existing_cross_bundle: HubBundle | None = None
            existing_cross_distance: int | None = None
            for cand in cast(Any, inst).candidates:
                scn = cand.scene
                if isinstance(scn, cast(Any, BaseHub).Config):
                    basehub_seen = True
                    transforms.append(getattr(scn, "transform", GridTransform.IDENTITY))
                    # Capture existing hub settings from the first BaseHub config we see
                    if existing_corner_bundle is None:
                        existing_corner_bundle = _normalize_bundle(getattr(scn, "corner_bundle", None), "chests")
                    if existing_cross_bundle is None:
                        existing_cross_bundle = _normalize_bundle(getattr(scn, "cross_bundle", None), "none")
                    if existing_cross_distance is None:
                        existing_cross_distance = int(getattr(scn, "cross_distance", 7) or 7)
            if basehub_seen:
                ov_corner = (overrides or {}).get("hub_corner_bundle")
                ov_cross = (overrides or {}).get("hub_cross_bundle")
                ov_dist = (overrides or {}).get("hub_cross_distance")

                # If override not provided, preserve existing scene values
                corner_bundle = (
                    _normalize_bundle(ov_corner, existing_corner_bundle or "chests")
                    if ov_corner is not None
                    else (existing_corner_bundle or "chests")
                )
                cross_bundle = (
                    _normalize_bundle(ov_cross, existing_cross_bundle or "none")
                    if ov_cross is not None
                    else (existing_cross_bundle or "none")
                )
                cross_distance = int(ov_dist) if ov_dist is not None else int(existing_cross_distance or 7)
                return make_hub_only_map_builder(
                    num_cogs=num_cogs,
                    width=width,
                    height=height,
                    corner_bundle=corner_bundle,
                    cross_bundle=cross_bundle,
                    cross_distance=cross_distance,
                    transforms=transforms or None,
                )
        return builder
    except Exception:
        return builder


def apply_procedural_overrides_to_builder(
    builder: MapBuilderConfig,
    *,
    num_cogs: int,
    overrides: dict[str, Any] | None = None,
) -> MapBuilderConfig:
    """Apply mission-level procedural_overrides to a MapGen builder when possible.

    Supports:
    - Hub-only builders produced by make_hub_only_map_builder (RandomScene[BaseHub]).
    - Machina builders produced by make_machina_procedural_map_builder (Biome* base scenes).
    Falls back to the original builder if structure is unrecognized.
    """
    ov = overrides or {}

    # 1) Try hub-only first
    hub_applied = apply_hub_overrides_to_builder(builder, num_cogs=num_cogs, overrides=ov)
    if hub_applied is not builder:
        return hub_applied

    # 2) Try machina-style (biome-based) procedural
    try:
        if not isinstance(builder, MapGen.Config):
            return builder

        base_inst = getattr(builder, "instance", None)
        biome_bases = (BiomeCavesConfig, BiomeForestConfig, BiomeDesertConfig, BiomeCityConfig)
        if isinstance(base_inst, biome_bases) or isinstance(base_inst, cast(Any, biome_bases)):
            width = int(ov.get("width", getattr(builder, "width", 100) or 100))
            height = int(ov.get("height", getattr(builder, "height", 100) or 100))
            seed = ov.get("seed", getattr(builder, "seed", None))

            allowed_keys = {
                "base_biome",
                "base_biome_config",
                "extractor_coverage",
                "extractors",
                "extractor_names",
                "extractor_weights",
                "hub_corner_bundle",
                "hub_cross_bundle",
                "hub_cross_distance",
                "biome_weights",
                "dungeon_weights",
                "biome_count",
                "dungeon_count",
                "density_scale",
                "max_biome_zone_fraction",
                "max_dungeon_zone_fraction",
                "distribution",
                "building_distributions",
            }
            kwargs = {k: v for k, v in ov.items() if k in allowed_keys}
            return make_machina_procedural_map_builder(
                num_cogs=num_cogs,
                width=width,
                height=height,
                seed=seed,
                **kwargs,
            )
        return builder
    except Exception:
        return builder
