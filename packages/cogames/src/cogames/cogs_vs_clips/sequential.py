from typing import Literal

import numpy as np

from cogames.cogs_vs_clips.mission import Mission
from cogames.cogs_vs_clips.procedural import (
    BaseHubVariant,
    EnvNodeVariant,
    MapGenVariant,
    MapSeedVariant,
    RandomTransform,
    RandomTransformConfig,
)
from cogames.cogs_vs_clips.procedural import (
    MachinaArenaConfig as BaseMachinaArenaConfig,
)
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.mapgen.area import AreaWhere
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.scene import ChildrenAction, Scene, SceneConfig
from mettagrid.mapgen.scenes.asteroid_mask import AsteroidMaskConfig
from mettagrid.mapgen.scenes.biome_caves import BiomeCavesConfig
from mettagrid.mapgen.scenes.biome_city import BiomeCityConfig
from mettagrid.mapgen.scenes.biome_desert import BiomeDesertConfig
from mettagrid.mapgen.scenes.biome_forest import BiomeForestConfig
from mettagrid.mapgen.scenes.biome_plains import BiomePlainsConfig
from mettagrid.mapgen.scenes.bounded_layout import BoundedLayout
from mettagrid.mapgen.scenes.bsp import BSPLayout
from mettagrid.mapgen.scenes.building_distributions import UniformExtractorParams
from mettagrid.mapgen.scenes.make_connected import MakeConnected
from mettagrid.mapgen.scenes.maze import MazeConfig
from mettagrid.mapgen.scenes.radial_maze import RadialMaze
from mettagrid.mapgen.scenes.random_scene import RandomScene, RandomSceneCandidate

HubBundle = Literal["extractors", "none", "custom"]

__all__ = [
    "BaseHubVariant",
    "BaseMachinaArenaConfig",
    "EnvNodeVariant",
    "MapGenVariant",
    "MapSeedVariant",
    "RandomTransform",
    "RandomTransformConfig",
    "SequentialMachinaArena",
    "SequentialMachinaArenaConfig",
    "SequentialMachinaArenaVariant",
]


class SequentialMachinaArenaConfig(BaseMachinaArenaConfig):
    _scene_cls = None


class SequentialMachinaArena(Scene[SequentialMachinaArenaConfig]):
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
            "plains": BiomePlainsConfig,
        }
        if cfg.base_biome not in biome_map:
            raise ValueError(f"Unknown base_biome '{cfg.base_biome}'. Valid: {sorted(biome_map.keys())}")
        BaseCfgModel: type[SceneConfig] = biome_map[cfg.base_biome]
        base_cfg: SceneConfig = BaseCfgModel.model_validate(cfg.base_biome_config or {})

        # Building weights
        default_building_weights = {
            "chest": 0.0,
            "charger": 0.6,
            "germanium_extractor": 0.5,
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
        def _make_biomes(weights: dict[str, float] | None) -> list[SceneConfig]:
            if weights is not None and "none" in weights:
                return []
            defaults = {"caves": 0.0, "forest": 1.0, "desert": 1.0, "city": 1.0, "plains": 1.0}
            w = {**defaults, **(weights or {})}
            biomes: list[SceneConfig] = []
            if float(w.get("caves", 0.0)) > 0:
                biomes.append(BiomeCavesConfig())
            if float(w.get("forest", 0.0)) > 0:
                biomes.append(BiomeForestConfig())
            if float(w.get("desert", 0.0)) > 0:
                biomes.append(BiomeDesertConfig())
            if float(w.get("city", 0.0)) > 0:
                biomes.append(BiomeCityConfig())
            if float(w.get("plains", 0.0)) > 0:
                biomes.append(BiomePlainsConfig())
            return biomes

        def _make_dungeons(weights: dict[str, float] | None) -> list[SceneConfig]:
            if weights is not None and "none" in weights:
                return []
            defaults = {"maze": 1.0, "radial": 1.0}
            w = {**defaults, **(weights or {})}
            dungeons: list[SceneConfig] = []
            if float(w.get("maze", 0.0)) > 0:
                dungeons.append(
                    RandomScene.Config(
                        candidates=[
                            RandomSceneCandidate(
                                scene=MazeConfig(
                                    algorithm="dfs",
                                ),
                                weight=0.6,
                            ),
                            RandomSceneCandidate(
                                scene=MazeConfig(
                                    algorithm="kruskal",
                                ),
                                weight=0.4,
                            ),
                        ]
                    )
                )
            if float(w.get("radial", 0.0)) > 0:
                dungeons.append(RadialMaze.Config())
            return dungeons

        biome_max_w = max(10, int(min(self.width * cfg.max_biome_zone_fraction, self.width // 2)))
        biome_max_h = max(10, int(min(self.height * cfg.max_biome_zone_fraction, self.height // 2)))
        dungeon_max_w = max(10, int(min(self.width * cfg.max_dungeon_zone_fraction, self.width // 2)))
        dungeon_max_h = max(10, int(min(self.height * cfg.max_dungeon_zone_fraction, self.height // 2)))

        def _wrap_in_layout(scene_cfg: SceneConfig, tag: str, max_w: int, max_h: int) -> SceneConfig:
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
        biomes = _make_biomes(cfg.biome_weights)
        if biomes:
            biome_count = len(biomes) if cfg.biome_count is None else max(int(biome_count), len(biomes))
            biome_children: list[ChildrenAction] = []
            for scene_cfg in biomes:
                biome_children.append(
                    ChildrenAction(
                        scene=_wrap_in_layout(scene_cfg, tag="biome.zone", max_w=biome_max_w, max_h=biome_max_h),
                        where=AreaWhere(tags=["zone"]),
                        order_by="random",
                        limit=1,
                        lock="biome.zone",
                    )
                )

            biome_layer = ChildrenAction(
                scene=BSPLayout.Config(
                    area_count=biome_count,
                    children=biome_children,
                ),
                where="full",
            )

        dungeon_layer: ChildrenAction | None = None
        dungeons = _make_dungeons(cfg.dungeon_weights)
        if dungeons:
            dungeon_count = len(dungeons) if cfg.dungeon_count is None else max(int(dungeon_count), len(dungeons))
            dungeon_children: list[ChildrenAction] = []
            for scene_cfg in dungeons:
                dungeon_children.append(
                    ChildrenAction(
                        scene=_wrap_in_layout(
                            scene_cfg,
                            tag="dungeon.zone",
                            max_w=dungeon_max_w,
                            max_h=dungeon_max_h,
                        ),
                        where=AreaWhere(tags=["zone"]),
                        order_by="random",
                        limit=1,
                        lock="dungeon.zone",
                    )
                )

            dungeon_layer = ChildrenAction(
                scene=BSPLayout.Config(
                    area_count=dungeon_count,
                    children=dungeon_children,
                ),
                where="full",
                order_by="first",
                limit=1,
            )

        children: list[ChildrenAction] = []

        # Base shell first
        children.append(ChildrenAction(scene=base_cfg, where="full"))

        if biome_layer is not None:
            children.append(biome_layer)
        if dungeon_layer is not None:
            children.append(dungeon_layer)

        asteroid_mask = cfg.asteroid_mask
        if asteroid_mask is None and min(self.width, self.height) >= 80:
            asteroid_mask = AsteroidMaskConfig()
        if asteroid_mask is not None:
            children.append(ChildrenAction(scene=asteroid_mask, where="full"))

        # Resources
        children.append(
            ChildrenAction(
                scene=UniformExtractorParams(
                    target_coverage=cfg.building_coverage,
                    building_names=building_names_final,
                    building_weights=building_weights_final,
                    clear_existing=False,
                    distribution=cfg.distribution,
                    building_distributions=cfg.building_distributions,
                ),
                where="full",
            )
        )

        # Connectivity + hub
        children.append(
            ChildrenAction(
                scene=cfg.hub.model_copy(deep=True, update={"spawn_count": cfg.spawn_count}),
                where="full",
            )
        )

        children.append(
            ChildrenAction(
                scene=MakeConnected.Config(
                    balance_corners=cfg.balance_corners,
                    balance_tolerance=cfg.balance_tolerance,
                    max_balance_shortcuts=cfg.max_balance_shortcuts,
                ),
                where="full",
            )
        )

        return children


class SequentialMachinaArenaVariant(EnvNodeVariant[SequentialMachinaArenaConfig]):
    def compat(self, mission: Mission) -> bool:
        env = mission.make_env()
        return isinstance(env.game.map_builder, MapGen.Config) and isinstance(
            env.game.map_builder.instance, SequentialMachinaArena.Config
        )

    @classmethod
    def extract_node(cls, env: MettaGridConfig) -> SequentialMachinaArenaConfig:
        assert isinstance(env.game.map_builder, MapGen.Config)
        assert isinstance(env.game.map_builder.instance, SequentialMachinaArena.Config)
        return env.game.map_builder.instance
