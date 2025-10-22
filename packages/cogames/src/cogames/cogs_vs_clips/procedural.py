from typing import Any

from mettagrid.map_builder.map_builder import MapBuilderConfig
from mettagrid.mapgen.area import AreaWhere
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.random.int import IntConstantDistribution
from mettagrid.mapgen.scene import ChildrenAction
from mettagrid.mapgen.scenes.base_hub import BaseHub
from mettagrid.mapgen.scenes.biome_caves import BiomeCaves, BiomeCavesConfig
from mettagrid.mapgen.scenes.biome_city import BiomeCity, BiomeCityConfig
from mettagrid.mapgen.scenes.biome_desert import BiomeDesert, BiomeDesertConfig
from mettagrid.mapgen.scenes.biome_forest import BiomeForest, BiomeForestConfig
from mettagrid.mapgen.scenes.bsp import BSP, BSPLayout
from mettagrid.mapgen.scenes.make_connected import MakeConnected
from mettagrid.mapgen.scenes.maze import Maze
from mettagrid.mapgen.scenes.radial_maze import RadialMaze
from mettagrid.mapgen.scenes.random_scene import RandomScene, RandomSceneCandidate
from mettagrid.mapgen.scenes.uniform_extractors import UniformExtractorScene


def make_machina_procedural_map_builder(
    num_cogs: int,
    *,
    width: int = 100,
    height: int = 100,
    base_biome: str = "caves",
    base_biome_config: dict[str, Any] | None = None,
    extractor_coverage: float = 0.02,
    extractor_names: list[str] | None = None,
    extractor_weights: dict[str, float] | None = None,
    biome_weights: dict[str, float] | None = None,
    dungeon_weights: dict[str, float] | None = None,
    biome_count: int | None = None,
    dungeon_count: int | None = None,
) -> MapBuilderConfig:
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

    chest_names = extractor_names or ["chest"]
    chest_weights = extractor_weights or {name: 1.0 for name in chest_names}

    # Optional layered biomes via BSPLayout
    # Defaults: fewer, larger zones for more dramatic/interesting layouts
    if biome_count is None:
        biome_count = max(3, (width * height) // 3000)
    if dungeon_count is None:
        dungeon_count = max(2, (width * height) // 4000)

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
            cands.append(
                RandomSceneCandidate(
                    scene=Maze.Config(
                        algorithm="dfs",
                        room_size=IntConstantDistribution(value=4),
                        wall_size=IntConstantDistribution(value=1),
                    ),
                    weight=float(w["maze"]),
                )
            )
        if w.get("radial", 0) > 0:
            cands.append(RandomSceneCandidate(scene=RadialMaze.Config(arms=8, arm_width=5), weight=float(w["radial"])))
        return cands

    biome_layer: ChildrenAction | None = None
    biome_cands = _make_biome_candidates(biome_weights) if biome_weights is not None else []
    if biome_cands:
        # Fill only a subset of zones to preserve base shell background
        biome_fill_count = max(1, int(biome_count * 0.6))
        biome_layer = ChildrenAction(
            scene=BSPLayout.Config(
                area_count=biome_count,
                children=[
                    ChildrenAction(
                        scene=RandomScene.Config(candidates=biome_cands),
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
    dungeon_cands = _make_dungeon_candidates(dungeon_weights) if dungeon_weights is not None else []
    if dungeon_cands:
        # Fill only a subset of zones to preserve base shell background
        dungeon_fill_count = max(1, int(dungeon_count * 0.5))
        dungeon_layer = ChildrenAction(
            scene=BSPLayout.Config(
                area_count=dungeon_count,
                children=[
                    ChildrenAction(
                        scene=RandomScene.Config(candidates=dungeon_cands),
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

    base_cfg.children = [
        # Optional biome/dungeon layers over the base shell
        *([biome_layer] if biome_layer is not None else []),
        *([dungeon_layer] if dungeon_layer is not None else []),
        # Resources first so connectors will tunnel between them
        ChildrenAction(
            scene=UniformExtractorScene.Config(
                target_coverage=extractor_coverage,
                extractor_names=chest_names,
                extractor_weights=chest_weights,
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
            scene=BaseHub.Config(spawn_count=num_cogs, hub_width=21, hub_height=21),
            where="full",
            order_by="last",
            limit=1,
        ),
    ]

    return MapGen.Config(width=width, height=height, instance=base_cfg)
