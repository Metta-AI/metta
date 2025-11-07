import abc
import typing

import numpy as np

import cogames.cogs_vs_clips.mission
import mettagrid.config.mettagrid_config
import mettagrid.mapgen.area
import mettagrid.mapgen.mapgen
import mettagrid.mapgen.random.int
import mettagrid.mapgen.scene
import mettagrid.mapgen.scenes.base_hub
import mettagrid.mapgen.scenes.biome_caves
import mettagrid.mapgen.scenes.biome_city
import mettagrid.mapgen.scenes.biome_desert
import mettagrid.mapgen.scenes.biome_forest
import mettagrid.mapgen.scenes.bounded_layout
import mettagrid.mapgen.scenes.bsp
import mettagrid.mapgen.scenes.building_distributions
import mettagrid.mapgen.scenes.make_connected
import mettagrid.mapgen.scenes.maze
import mettagrid.mapgen.scenes.radial_maze
import mettagrid.mapgen.scenes.random_scene

HubBundle = typing.Literal["extractors", "none", "custom"]


class MachinaArenaConfig(mettagrid.mapgen.scene.SceneConfig):
    # Core composition
    spawn_count: int

    # Biome / dungeon structure
    base_biome: str = "caves"
    base_biome_config: dict[str, typing.Any] = {}

    #### Building placement ####

    # How much of the map is covered by buildings
    building_coverage: float = 0.01
    # Resource placement (building-based API)
    # Defines the set of buildings that can be placed on the map
    building_names: list[str] | None = None
    # What proportion of buildings are of a type, falls back to default if not set
    # If building_names is not set, this is used to determine the buildings
    building_weights: dict[str, float] | None = None

    # Hub config. `spawn_count` will be set based on `spawn_count` in this config.
    hub: mettagrid.mapgen.scenes.base_hub.BaseHubConfig = mettagrid.mapgen.scenes.base_hub.BaseHubConfig(
        corner_bundle="extractors",
        cross_bundle="none",
        cross_distance=7,
    )

    #### Layers ####

    biome_weights: dict[str, float] | None = None
    dungeon_weights: dict[str, float] | None = None
    biome_count: int | None = None
    dungeon_count: int | None = None
    density_scale: float = 1.0
    max_biome_zone_fraction: float = 0.35
    max_dungeon_zone_fraction: float = 0.25

    #### Distributions ####

    # How buildings are distributed on the map
    distribution: mettagrid.mapgen.scenes.building_distributions.DistributionConfig = (
        mettagrid.mapgen.scenes.building_distributions.DistributionConfig()
    )

    # How buildings are distributed on the map per building type, falls back to global distribution if not set
    building_distributions: dict[str, mettagrid.mapgen.scenes.building_distributions.DistributionConfig] | None = None


class MachinaArena(mettagrid.mapgen.scene.Scene[MachinaArenaConfig]):
    def render(self) -> None:
        # No direct drawing; composition is done via children actions
        return

    def get_children(self) -> list[mettagrid.mapgen.scene.ChildrenAction]:
        cfg = self.config

        # Base biome map
        biome_map: dict[str, type[mettagrid.mapgen.scene.SceneConfig]] = {
            "caves": mettagrid.mapgen.scenes.biome_caves.BiomeCavesConfig,
            "forest": mettagrid.mapgen.scenes.biome_forest.BiomeForestConfig,
            "desert": mettagrid.mapgen.scenes.biome_desert.BiomeDesertConfig,
            "city": mettagrid.mapgen.scenes.biome_city.BiomeCityConfig,
        }
        if cfg.base_biome not in biome_map:
            raise ValueError(f"Unknown base_biome '{cfg.base_biome}'. Valid: {sorted(biome_map.keys())}")
        BaseCfgModel: type[mettagrid.mapgen.scene.SceneConfig] = biome_map[cfg.base_biome]
        base_cfg: mettagrid.mapgen.scene.SceneConfig = BaseCfgModel.model_validate(cfg.base_biome_config or {})

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
        def _make_biome_candidates(
            weights: dict[str, float] | None,
        ) -> list[mettagrid.mapgen.scenes.random_scene.RandomSceneCandidate]:
            defaults = {"caves": 1.0, "forest": 1.0, "desert": 1.0, "city": 1.0}
            w = {**defaults, **(weights or {})}
            cands: list[mettagrid.mapgen.scenes.random_scene.RandomSceneCandidate] = []
            if w.get("caves", 0) > 0:
                cands.append(
                    mettagrid.mapgen.scenes.random_scene.RandomSceneCandidate(
                        scene=mettagrid.mapgen.scenes.biome_caves.BiomeCavesConfig(), weight=w["caves"]
                    )
                )
            if w.get("forest", 0) > 0:
                cands.append(
                    mettagrid.mapgen.scenes.random_scene.RandomSceneCandidate(
                        scene=mettagrid.mapgen.scenes.biome_forest.BiomeForestConfig(), weight=w["forest"]
                    )
                )
            if w.get("desert", 0) > 0:
                cands.append(
                    mettagrid.mapgen.scenes.random_scene.RandomSceneCandidate(
                        scene=mettagrid.mapgen.scenes.biome_desert.BiomeDesertConfig(), weight=w["desert"]
                    )
                )
            if w.get("city", 0) > 0:
                cands.append(
                    mettagrid.mapgen.scenes.random_scene.RandomSceneCandidate(
                        scene=mettagrid.mapgen.scenes.biome_city.BiomeCityConfig(), weight=w["city"]
                    )
                )
            return cands

        def _make_dungeon_candidates(
            weights: dict[str, float] | None,
        ) -> list[mettagrid.mapgen.scenes.random_scene.RandomSceneCandidate]:
            defaults = {"bsp": 1.0, "maze": 1.0, "radial": 1.0}
            w = {**defaults, **(weights or {})}
            cands: list[mettagrid.mapgen.scenes.random_scene.RandomSceneCandidate] = []
            if w.get("bsp", 0) > 0:
                cands.append(
                    mettagrid.mapgen.scenes.random_scene.RandomSceneCandidate(
                        scene=mettagrid.mapgen.scenes.bsp.BSPConfig(
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
                    mettagrid.mapgen.scenes.random_scene.RandomSceneCandidate(
                        scene=mettagrid.mapgen.scenes.maze.MazeConfig(
                            algorithm="dfs",
                            room_size=mettagrid.mapgen.random.int.IntConstantDistribution(value=2),
                            wall_size=mettagrid.mapgen.random.int.IntConstantDistribution(value=1),
                        ),
                        weight=maze_weight * 0.6,
                    )
                )
                cands.append(
                    mettagrid.mapgen.scenes.random_scene.RandomSceneCandidate(
                        scene=mettagrid.mapgen.scenes.maze.MazeConfig(
                            algorithm="kruskal",
                            room_size=mettagrid.mapgen.random.int.IntConstantDistribution(value=2),
                            wall_size=mettagrid.mapgen.random.int.IntConstantDistribution(value=1),
                        ),
                        weight=maze_weight * 0.4,
                    )
                )
            if w.get("radial", 0) > 0:
                cands.append(
                    mettagrid.mapgen.scenes.random_scene.RandomSceneCandidate(
                        scene=mettagrid.mapgen.scenes.radial_maze.RadialMaze.Config(
                            arms=8, arm_width=2, clear_background=False, outline_walls=True
                        ),
                        weight=w["radial"],
                    )
                )
            return cands

        biome_max_w = max(10, int(min(self.width * cfg.max_biome_zone_fraction, self.width // 2)))
        biome_max_h = max(10, int(min(self.height * cfg.max_biome_zone_fraction, self.height // 2)))
        dungeon_max_w = max(10, int(min(self.width * cfg.max_dungeon_zone_fraction, self.width // 2)))
        dungeon_max_h = max(10, int(min(self.height * cfg.max_dungeon_zone_fraction, self.height // 2)))

        def _wrap_in_layout(
            scene_cfg: mettagrid.mapgen.scene.SceneConfig, tag: str, max_w: int, max_h: int
        ) -> mettagrid.mapgen.scene.SceneConfig:
            return mettagrid.mapgen.scenes.bounded_layout.BoundedLayout.Config(
                max_width=max_w,
                max_height=max_h,
                tag=tag,
                children=[
                    mettagrid.mapgen.scene.ChildrenAction(
                        scene=scene_cfg,
                        where=mettagrid.mapgen.area.AreaWhere(tags=[tag]),
                        limit=1,
                        order_by="first",
                    )
                ],
            )

        biome_layer: mettagrid.mapgen.scene.ChildrenAction | None = None
        biome_cands = _make_biome_candidates(cfg.biome_weights)
        if biome_cands:
            biome_fill_count = max(1, int(biome_count * 0.6))
            biome_layer = mettagrid.mapgen.scene.ChildrenAction(
                scene=mettagrid.mapgen.scenes.bsp.BSPLayout.Config(
                    area_count=biome_count,
                    children=[
                        mettagrid.mapgen.scene.ChildrenAction(
                            scene=_wrap_in_layout(
                                mettagrid.mapgen.scenes.random_scene.RandomScene.Config(candidates=biome_cands),
                                tag="biome.zone",
                                max_w=biome_max_w,
                                max_h=biome_max_h,
                            ),
                            where=mettagrid.mapgen.area.AreaWhere(tags=["zone"]),
                            order_by="random",
                            limit=biome_fill_count,
                        )
                    ],
                ),
                where="full",
            )

        dungeon_layer: mettagrid.mapgen.scene.ChildrenAction | None = None
        dungeon_cands = _make_dungeon_candidates(cfg.dungeon_weights)
        if dungeon_cands:
            dungeon_fill_count = max(1, int(dungeon_count * 0.5))
            dungeon_layer = mettagrid.mapgen.scene.ChildrenAction(
                scene=mettagrid.mapgen.scenes.bsp.BSPLayout.Config(
                    area_count=dungeon_count,
                    children=[
                        mettagrid.mapgen.scene.ChildrenAction(
                            scene=_wrap_in_layout(
                                mettagrid.mapgen.scenes.random_scene.RandomSceneConfig(candidates=dungeon_cands),
                                tag="dungeon.zone",
                                max_w=dungeon_max_w,
                                max_h=dungeon_max_h,
                            ),
                            where=mettagrid.mapgen.area.AreaWhere(tags=["zone"]),
                            order_by="random",
                            limit=dungeon_fill_count,
                        )
                    ],
                ),
                where="full",
                order_by="first",
                limit=1,
            )

        children: list[mettagrid.mapgen.scene.ChildrenAction] = []

        # Base shell first
        children.append(mettagrid.mapgen.scene.ChildrenAction(scene=base_cfg, where="full"))

        if biome_layer is not None:
            children.append(biome_layer)
        if dungeon_layer is not None:
            children.append(dungeon_layer)

        # Resources
        children.append(
            mettagrid.mapgen.scene.ChildrenAction(
                scene=mettagrid.mapgen.scenes.building_distributions.UniformExtractorParams(
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
            mettagrid.mapgen.scene.ChildrenAction(
                scene=mettagrid.mapgen.scenes.make_connected.MakeConnected.Config(),
                where="full",
            )
        )

        children.append(
            mettagrid.mapgen.scene.ChildrenAction(
                scene=cfg.hub.model_copy(deep=True, update={"spawn_count": cfg.spawn_count}),
                where="full",
            )
        )

        children.append(
            mettagrid.mapgen.scene.ChildrenAction(
                scene=mettagrid.mapgen.scenes.make_connected.MakeConnected.Config(),
                where="full",
            )
        )

        return children


class RandomTransformConfig(mettagrid.mapgen.scene.SceneConfig):
    scene: mettagrid.mapgen.scene.AnySceneConfig


class RandomTransform(mettagrid.mapgen.scene.Scene[RandomTransformConfig]):
    def render(self) -> None:
        return

    def get_children(self) -> list[mettagrid.mapgen.scene.ChildrenAction]:
        return [
            mettagrid.mapgen.scene.ChildrenAction(
                scene=self.config.scene.model_copy(
                    update={
                        "transform": mettagrid.mapgen.scene.GridTransform(
                            self.rng.choice(list(mettagrid.mapgen.scene.GridTransform))
                        )
                    }
                ),
                where="full",
            )
        ]


class EnvNodeVariant[T](cogames.cogs_vs_clips.mission.MissionVariant, abc.ABC):
    @abc.abstractmethod
    def extract_node(self, env: mettagrid.config.mettagrid_config.MettaGridConfig) -> T: ...

    @abc.abstractmethod
    def modify_node(self, node: T): ...

    @typing.override
    def modify_env(self, mission, env) -> None:
        node = self.extract_node(env)
        self.modify_node(node)


class MapGenVariant(EnvNodeVariant[mettagrid.mapgen.mapgen.MapGenConfig]):
    @classmethod
    def extract_node(
        cls, env: mettagrid.config.mettagrid_config.MettaGridConfig
    ) -> mettagrid.mapgen.mapgen.MapGenConfig:
        map_builder = env.game.map_builder
        if not isinstance(map_builder, mettagrid.mapgen.mapgen.MapGen.Config):
            raise TypeError("MapGenConfigVariant can only be applied to MapGen.Config builders")
        return map_builder


class BaseHubVariant(EnvNodeVariant[mettagrid.mapgen.scenes.base_hub.BaseHubConfig]):
    @classmethod
    def extract_node(
        cls, env: mettagrid.config.mettagrid_config.MettaGridConfig
    ) -> mettagrid.mapgen.scenes.base_hub.BaseHubConfig:
        map_builder = env.game.map_builder
        if not isinstance(map_builder, mettagrid.mapgen.mapgen.MapGen.Config):
            raise TypeError("BaseHubVariant can only be applied to MapGen.Config builders")
        instance = map_builder.instance
        if isinstance(instance, RandomTransform.Config) and isinstance(
            instance.scene, mettagrid.mapgen.scenes.base_hub.BaseHub.Config
        ):
            return instance.scene
        elif isinstance(instance, MachinaArena.Config):
            return instance.hub
        else:
            raise TypeError("BaseHubVariant can only be applied RandomTransform/BaseHub or MachinaArena scenes")


class MachinaArenaVariant(EnvNodeVariant[MachinaArenaConfig]):
    @classmethod
    def extract_node(cls, env: mettagrid.config.mettagrid_config.MettaGridConfig) -> MachinaArenaConfig:
        map_builder = env.game.map_builder
        if not isinstance(map_builder, mettagrid.mapgen.mapgen.MapGen.Config):
            raise TypeError("MachinaArenaVariant can only be applied to MapGen.Config builders")
        instance = map_builder.instance
        if isinstance(instance, MachinaArena.Config):
            return instance
        else:
            raise TypeError("MachinaArenaVariant can only be applied to MachinaArena.Config scenes")
