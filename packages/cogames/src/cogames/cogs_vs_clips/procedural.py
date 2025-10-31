from typing import Any, Literal, cast

import numpy as np

from mettagrid.map_builder.map_builder import MapBuilderConfig
from mettagrid.mapgen.area import AreaWhere
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.random.int import IntConstantDistribution
from mettagrid.mapgen.scene import ChildrenAction, GridTransform, Scene, SceneConfig
from mettagrid.mapgen.scenes.base_hub import BaseHubConfig
from mettagrid.mapgen.scenes.biome_caves import BiomeCavesConfig
from mettagrid.mapgen.scenes.biome_city import BiomeCityConfig
from mettagrid.mapgen.scenes.biome_desert import BiomeDesertConfig
from mettagrid.mapgen.scenes.biome_forest import BiomeForestConfig
from mettagrid.mapgen.scenes.bounded_layout import BoundedLayoutConfig
from mettagrid.mapgen.scenes.bsp import BSPConfig, BSPLayoutConfig
from mettagrid.mapgen.scenes.building_distributions import (
    DistributionConfig,
    UniformExtractorParams,
)
from mettagrid.mapgen.scenes.make_connected import MakeConnectedConfig
from mettagrid.mapgen.scenes.maze import MazeConfig
from mettagrid.mapgen.scenes.radial_maze import RadialMazeConfig
from mettagrid.mapgen.scenes.random_scene import RandomSceneCandidate, RandomSceneConfig

HubBundle = Literal["chests", "extractors", "none", "custom"]


def _normalize_bundle(value: str | None, default: HubBundle) -> HubBundle:
    if value in {"chests", "extractors", "none", "custom"}:
        return cast(HubBundle, value)
    return default


class MachinaArenaConfig(SceneConfig):
    # Core composition
    spawn_count: int
    base_biome: str = "caves"
    base_biome_config: dict[str, Any] = {}

    # Building placement
    building_coverage: float = 0.01
    building_weights: dict[str, float] | None = None
    building_names: list[str] | None = None

    # Hub bundles
    hub_corner_bundle: HubBundle = "chests"
    hub_cross_bundle: HubBundle = "none"
    hub_cross_distance: int = 7

    # Layers
    biome_weights: dict[str, float] | None = None
    dungeon_weights: dict[str, float] | None = None
    biome_count: int | None = None
    dungeon_count: int | None = None
    density_scale: float = 1.0
    max_biome_zone_fraction: float = 0.35
    max_dungeon_zone_fraction: float = 0.25

    # Distributions
    distribution: DistributionConfig = DistributionConfig()
    building_distributions: dict[str, DistributionConfig] | None = None


class MachinaArena(Scene[MachinaArenaConfig]):
    def render(self) -> None:
        # No direct drawing; composition is done via children actions
        return

    def get_children(self) -> list[ChildrenAction]:
        cfg = self.config

        # Base biome map
        biome_map: dict[str, type[SceneConfig]] = {
            "caves": BiomeCavesConfig,
            "forest": BiomeForestConfig,
            "desert": BiomeDesertConfig,
            "city": BiomeCityConfig,
        }
        if cfg.base_biome not in biome_map:
            raise ValueError(f"Unknown base_biome '{cfg.base_biome}'. Valid: {sorted(biome_map.keys())}")
        BaseCfgModel: type[SceneConfig] = biome_map[cfg.base_biome]
        base_cfg: SceneConfig = BaseCfgModel.model_validate(cfg.base_biome_config or {})

        # Building weights
        default_building_weights = {
            "chest": 0.0,
            "charger": 0.6,
            "germanium_extractor": 0.6,
            "silicon_extractor": 0.3,
            "oxygen_extractor": 0.3,
            "carbon_extractor": 0.3,
        }

        weights_dict: dict[str, float] = (
            {str(k): v for k, v in cfg.building_weights.items()} if cfg.building_weights is not None else {}
        )
        if not weights_dict:
            if cfg.building_names is not None:
                weights_dict = {name: default_building_weights.get(name, 1.0) for name in cfg.building_names}
            else:
                weights_dict = {k: v for k, v in default_building_weights.items()}

        building_names_final = list(dict.fromkeys(list((cfg.building_names or list(weights_dict.keys())))))
        building_weights_final = {
            name: weights_dict.get(name, default_building_weights.get(name, 1.0)) for name in building_names_final
        }

        # Autoscale counts
        def _autoscale_zone_counts(
            w: int, h: int, *, biome_density: float = 1.0, dungeon_density: float = 1.0
        ) -> tuple[int, int]:
            area = max(1, w * h)
            biome_divisor = max(800, int(1600 / max(0.1, biome_density)))
            dungeon_divisor = max(800, int(1500 / max(0.1, dungeon_density)))
            biomes = max(3, min(48, area // biome_divisor))
            dungeons = max(3, min(48, area // dungeon_divisor))
            return int(biomes), int(dungeons)

        biome_count = cfg.biome_count
        dungeon_count = cfg.dungeon_count
        if biome_count is None or dungeon_count is None:
            auto_biomes, auto_dungeons = _autoscale_zone_counts(
                self.width, self.height, biome_density=cfg.density_scale, dungeon_density=cfg.density_scale
            )
            biome_count = auto_biomes if biome_count is None else biome_count
            dungeon_count = auto_dungeons if dungeon_count is None else dungeon_count

        def _min_count_for_fraction(frac: float) -> int:
            if frac <= 0:
                return 1
            return int(np.ceil(1.0 / min(0.9, max(0.02, float(frac)))))

        biome_count = max(int(biome_count), _min_count_for_fraction(cfg.max_biome_zone_fraction))
        dungeon_count = max(int(dungeon_count), _min_count_for_fraction(cfg.max_dungeon_zone_fraction))

        # Candidates
        def _make_biome_candidates(weights: dict[str, float] | None) -> list[RandomSceneCandidate]:
            defaults = {"caves": 1.0, "forest": 1.0, "desert": 1.0, "city": 1.0}
            w = {**defaults, **(weights or {})}
            cands: list[RandomSceneCandidate] = []
            if w.get("caves", 0) > 0:
                cands.append(RandomSceneCandidate(scene=BiomeCavesConfig(), weight=w["caves"]))
            if w.get("forest", 0) > 0:
                cands.append(RandomSceneCandidate(scene=BiomeForestConfig(), weight=w["forest"]))
            if w.get("desert", 0) > 0:
                cands.append(RandomSceneCandidate(scene=BiomeDesertConfig(), weight=w["desert"]))
            if w.get("city", 0) > 0:
                cands.append(RandomSceneCandidate(scene=BiomeCityConfig(), weight=w["city"]))
            return cands

        def _make_dungeon_candidates(weights: dict[str, float] | None) -> list[RandomSceneCandidate]:
            defaults = {"bsp": 1.0, "maze": 1.0, "radial": 1.0}
            w = {**defaults, **(weights or {})}
            cands: list[RandomSceneCandidate] = []
            if w.get("bsp", 0) > 0:
                cands.append(
                    RandomSceneCandidate(
                        scene=BSPConfig(
                            rooms=4,
                            min_room_size=6,
                            min_room_size_ratio=0.35,
                            max_room_size_ratio=0.75,
                        ),
                        weight=w["bsp"],
                    )
                )
            if w.get("maze", 0) > 0:
                maze_weight = w["maze"]
                cands.append(
                    RandomSceneCandidate(
                        scene=MazeConfig(
                            algorithm="dfs",
                            room_size=IntConstantDistribution(value=2),
                            wall_size=IntConstantDistribution(value=1),
                        ),
                        weight=maze_weight * 0.6,
                    )
                )
                cands.append(
                    RandomSceneCandidate(
                        scene=MazeConfig(
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
                        scene=RadialMazeConfig(arms=8, arm_width=2, clear_background=False, outline_walls=True),
                        weight=w["radial"],
                    )
                )
            return cands

        biome_max_w = max(10, int(min(self.width * cfg.max_biome_zone_fraction, self.width // 2)))
        biome_max_h = max(10, int(min(self.height * cfg.max_biome_zone_fraction, self.height // 2)))
        dungeon_max_w = max(10, int(min(self.width * cfg.max_dungeon_zone_fraction, self.width // 2)))
        dungeon_max_h = max(10, int(min(self.height * cfg.max_dungeon_zone_fraction, self.height // 2)))

        def _wrap_in_layout(scene_cfg: SceneConfig, tag: str, max_w: int, max_h: int) -> SceneConfig:
            return BoundedLayoutConfig(
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
        biome_cands = _make_biome_candidates(cfg.biome_weights)
        if biome_cands:
            biome_fill_count = max(1, int(biome_count * 0.6))
            biome_layer = ChildrenAction(
                scene=BSPLayoutConfig(
                    area_count=biome_count,
                    children=[
                        ChildrenAction(
                            scene=_wrap_in_layout(
                                RandomSceneConfig(candidates=biome_cands),
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
        dungeon_cands = _make_dungeon_candidates(cfg.dungeon_weights)
        if dungeon_cands:
            dungeon_fill_count = max(1, int(dungeon_count * 0.5))
            dungeon_layer = ChildrenAction(
                scene=BSPLayoutConfig(
                    area_count=dungeon_count,
                    children=[
                        ChildrenAction(
                            scene=_wrap_in_layout(
                                RandomSceneConfig(candidates=dungeon_cands),
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

        children: list[ChildrenAction] = []

        # Base shell first
        children.append(ChildrenAction(scene=base_cfg, where="full", order_by="first", limit=1))

        if biome_layer is not None:
            children.append(biome_layer)
        if dungeon_layer is not None:
            children.append(dungeon_layer)

        # Resources
        children.append(
            ChildrenAction(
                scene=UniformExtractorParams(
                    target_coverage=float(cfg.building_coverage),
                    building_names=building_names_final,
                    building_weights=building_weights_final,
                    clear_existing=False,
                    distribution=cfg.distribution,
                    building_distributions=cfg.building_distributions,
                ),
                where="full",
                order_by="last",
                lock="arena.resources",
                limit=1,
            )
        )

        # Connectivity + hub
        children.append(
            ChildrenAction(scene=MakeConnectedConfig(), where="full", order_by="last", lock="arena.connect", limit=1)
        )

        children.append(
            ChildrenAction(
                scene=BaseHubConfig(
                    spawn_count=int(cfg.spawn_count),
                    hub_width=21,
                    hub_height=21,
                    corner_bundle=cfg.hub_corner_bundle,
                    cross_bundle=cfg.hub_cross_bundle,
                    cross_distance=int(cfg.hub_cross_distance),
                ),
                where="full",
                order_by="last",
                limit=1,
            )
        )

        children.append(
            ChildrenAction(
                scene=MakeConnectedConfig(),
                where="full",
                order_by="last",
                lock="arena.connect.final",
                limit=1,
            )
        )

        return children


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
        RandomSceneCandidate(scene=BaseHubConfig(**base_kwargs, transform=t), weight=1.0) for t in transform_set
    ]

    return MapGen.Config(
        width=width,
        height=height,
        seed=seed,
        instance=RandomSceneConfig(candidates=candidates),
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
    if not isinstance(builder, MapGen.Config):
        return builder

    # Preserve existing top-level seed unless explicitly overridden
    existing_seed = builder.seed
    override_seed = (overrides or {}).get("seed", None)
    final_seed = override_seed if override_seed is not None else existing_seed

    width = builder.width or 21
    height = builder.height or 21

    inst = builder.instance
    if isinstance(inst, RandomSceneConfig):
        # Verify candidates are BaseHub configs; extract transforms
        transforms: list[GridTransform] = []
        basehub_seen = False
        existing_corner_bundle: HubBundle | None = None
        existing_cross_bundle: HubBundle | None = None
        existing_cross_distance: int | None = None
        for cand in inst.candidates:
            scn = cand.scene
            if isinstance(scn, BaseHubConfig):
                basehub_seen = True
                transforms.append(scn.transform)
                if existing_corner_bundle is None:
                    existing_corner_bundle = scn.corner_bundle
                if existing_cross_bundle is None:
                    existing_cross_bundle = scn.cross_bundle
                if existing_cross_distance is None:
                    existing_cross_distance = int(scn.cross_distance)
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
                seed=final_seed,
                corner_bundle=corner_bundle,
                cross_bundle=cross_bundle,
                cross_distance=cross_distance,
                transforms=transforms or None,
            )
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
    if not isinstance(builder, MapGen.Config):
        return builder

    base_inst = builder.instance
    if isinstance(base_inst, MachinaArenaConfig):
        width = int(ov.pop("width", builder.width))
        height = int(ov.pop("height", builder.height))
        seed = ov.pop("seed", builder.seed)

        allowed_keys = {
            "base_biome",
            "base_biome_config",
            # Building-based keys
            "building_coverage",
            "building_weights",
            "building_names",
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
        unknown = set(ov.keys()) - allowed_keys - {"width", "height", "seed"}
        if unknown:
            raise ValueError("Unknown procedural override key(s): " + ", ".join(sorted(unknown)))

        kwargs = {k: v for k, v in ov.items() if k in allowed_keys}
        return MapGen.Config(
            width=width,
            height=height,
            seed=seed,
            instance=MachinaArenaConfig(spawn_count=num_cogs, **kwargs),
        )
    return builder
