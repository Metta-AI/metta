from cogames.cogs_vs_clips.stations import (
    assembler,
    carbon_extractor,
    charger,
    chest,
    geranium_extractor,
    oxygen_extractor,
    resources,
    silicon_extractor,
)
from mettagrid.config.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    ChangeGlyphActionConfig,
    GameConfig,
    MettaGridConfig,
    RecipeConfig,
    WallConfig,
)
from mettagrid.map_builder.random import RandomMapBuilder
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.scenes.base_hub import BaseHub, BaseHubParams
from mettagrid.mapgen.scenes.bsp import BSP, BSPParams
from mettagrid.mapgen.scenes.make_connected import MakeConnected, MakeConnectedParams
from mettagrid.mapgen.scenes.quadrant_resources import QuadrantResources, QuadrantResourcesParams
from mettagrid.mapgen.scenes.quadrants import Quadrants, QuadrantsParams
from mettagrid.mapgen.scenes.random_scene import RandomScene, RandomSceneCandidate, RandomSceneParams
from mettagrid.mapgen.types import AreaWhere


def make_game(
    num_cogs: int = 4,
    width: int = 10,
    height: int = 10,
    num_assemblers: int = 1,
    num_chargers: int = 0,
    num_carbon_extractors: int = 0,
    num_oxygen_extractors: int = 0,
    num_geranium_extractors: int = 0,
    num_silicon_extractors: int = 0,
    num_chests: int = 0,
) -> MettaGridConfig:
    return MettaGridConfig(
        game=GameConfig(
            resource_names=resources,
            num_agents=num_cogs,
            actions=ActionsConfig(
                move=ActionConfig(consumed_resources={"energy": 1}),
                noop=ActionConfig(),
                change_glyph=ChangeGlyphActionConfig(number_of_glyphs=16),
                put_items=ActionConfig(enabled=False),
                get_items=ActionConfig(enabled=False),
            ),
            objects={
                "wall": WallConfig(type_id=1),
                "charger": charger(),
                "carbon_extractor": carbon_extractor(),
                "oxygen_extractor": oxygen_extractor(),
                "geranium_extractor": geranium_extractor(),
                "silicon_extractor": silicon_extractor(),
                "chest": chest(),
                "assembler": assembler(),
            },
            map_builder=RandomMapBuilder.Config(
                width=width,
                height=height,
                agents=num_cogs,
                objects={
                    "assembler": num_assemblers,
                    "charger": num_chargers,
                    "carbon_extractor": num_carbon_extractors,
                    "oxygen_extractor": num_oxygen_extractors,
                    "geranium_extractor": num_geranium_extractors,
                    "silicon_extractor": num_silicon_extractors,
                    "chest": num_chests,
                },
                seed=42,
            ),
            agent=AgentConfig(
                default_resource_limit=10,
                resource_limits={
                    "heart": 1,
                    "energy": 100,
                },
                rewards=AgentRewards(
                    inventory={
                        "heart": 1,
                    },
                ),
                initial_inventory={
                    "energy": 100,
                },
            ),
        )
    )


def tutorial_assembler_simple(num_cogs: int = 1) -> MettaGridConfig:
    cfg = make_game(num_cogs=num_cogs, num_assemblers=1)
    cfg.game.objects["assembler"] = assembler()
    cfg.game.objects["assembler"].recipes = [
        (["Any"], RecipeConfig(input_resources={"battery_red": 3}, output_resources={"heart": 1}, cooldown=10))
    ]
    return cfg


def tutorial_assembler_complex(num_cogs: int = 1) -> MettaGridConfig:
    cfg = make_game(num_cogs=num_cogs, num_assemblers=1)
    cfg.game.objects["assembler"] = assembler()
    cfg.game.objects["assembler"].recipes = [
        (["Any"], RecipeConfig(input_resources={"battery_red": 3}, output_resources={"heart": 1}, cooldown=10))
    ]
    return cfg


def games() -> dict[str, MettaGridConfig]:
    return {
        "assembler_1_simple": tutorial_assembler_complex(num_cogs=1),
        "assembler_1_complex": tutorial_assembler_simple(num_cogs=1),
        "assembler_2_simple": tutorial_assembler_simple(num_cogs=4),
        "assembler_2_complex": tutorial_assembler_complex(num_cogs=4),
        # "extractor_1cog_1resource": tutorial_extractor(num_cogs=1),
        # "extractor_1cog_4resource": tutorial_extractor(num_cogs=1),
        # "harvest_1": tutorial_harvest(num_cogs=1),
        # "harvest_4": tutorial_harvest(num_cogs=4),
        # "base_1": tutorial_base(num_cogs=1),
        # "base_4": tutorial_base(num_cogs=4),
        # "forage_1": tutorial_forage(num_cogs=1),
        # "forage_4": tutorial_forage(num_cogs=4),
        # "chest_1": tutorial_chest(num_cogs=1),
        # "chest_4": tutorial_chest(num_cogs=4),
        "machina_1": make_game(num_cogs=1),
        "machina_2": make_game(
            num_cogs=4,
            width=20,
            height=20,
            num_assemblers=1,
            num_chests=1,
            num_chargers=5,
            num_carbon_extractors=1,
            num_oxygen_extractors=1,
            num_geranium_extractors=1,
            num_silicon_extractors=1,
        ),
        "machina_sanctum": machina_sanctum(num_cogs=4),
    }


def machina_sanctum(num_cogs: int = 4) -> MettaGridConfig:
    """Quadrants map with a protected central base.

    Base: 11x11 area with assembler at center, 4 corner generators, L-shaped exits, and spawn pads.
    Outer: Four quadrants populated with BSP scenes.
    """

    cfg = make_game(num_cogs=num_cogs)

    # Swap to scene-based MapGen
    cfg.game.map_builder = MapGen.Config(
        width=55,
        height=55,
        seed=None,
        root=Quadrants.factory(
            params=QuadrantsParams(base_size=11),
            children_actions=[
                # Central base
                dict(
                    scene=BaseHub.factory(
                        BaseHubParams(
                            altar_object="altar",
                            corner_generator="generator_red",
                        )
                    ),
                    where=AreaWhere(tags=["base"]),
                    limit=1,
                    lock="keep",
                    order_by="first",
                ),
                # Quadrant population (randomly choose layout style per quadrant)
                dict(
                    scene=RandomScene.factory(
                        RandomSceneParams(
                            candidates=[
                                RandomSceneCandidate(
                                    scene=BSP.factory(
                                        BSPParams(
                                            rooms=10,
                                            min_room_size=4,
                                            min_room_size_ratio=0.2,
                                            max_room_size_ratio=0.55,
                                        )
                                    ),
                                    weight=2.0,
                                ),
                                RandomSceneCandidate(
                                    scene=BSP.factory(
                                        BSPParams(
                                            rooms=8,
                                            min_room_size=5,
                                            min_room_size_ratio=0.25,
                                            max_room_size_ratio=0.5,
                                        )
                                    ),
                                    weight=1.0,
                                ),
                            ]
                        )
                    ),
                    where=AreaWhere(tags=["quadrant.0"]),
                    limit=1,
                    lock="quad",
                    order_by="first",
                ),
                dict(
                    scene=RandomScene.factory(
                        RandomSceneParams(
                            candidates=[
                                RandomSceneCandidate(
                                    scene=BSP.factory(
                                        BSPParams(
                                            rooms=9,
                                            min_room_size=4,
                                            min_room_size_ratio=0.2,
                                            max_room_size_ratio=0.5,
                                        )
                                    ),
                                    weight=1.0,
                                ),
                                RandomSceneCandidate(
                                    scene=BSP.factory(
                                        BSPParams(
                                            rooms=12,
                                            min_room_size=3,
                                            min_room_size_ratio=0.15,
                                            max_room_size_ratio=0.6,
                                        )
                                    ),
                                    weight=1.5,
                                ),
                            ]
                        )
                    ),
                    where=AreaWhere(tags=["quadrant.1"]),
                    limit=1,
                    lock="quad",
                    order_by="first",
                ),
                dict(
                    scene=RandomScene.factory(
                        RandomSceneParams(
                            candidates=[
                                RandomSceneCandidate(
                                    scene=BSP.factory(
                                        BSPParams(
                                            rooms=8, min_room_size=4, min_room_size_ratio=0.2, max_room_size_ratio=0.5
                                        )
                                    ),
                                    weight=1.0,
                                ),
                                RandomSceneCandidate(
                                    scene=BSP.factory(
                                        BSPParams(
                                            rooms=6, min_room_size=6, min_room_size_ratio=0.3, max_room_size_ratio=0.5
                                        )
                                    ),
                                    weight=1.0,
                                ),
                            ]
                        )
                    ),
                    where=AreaWhere(tags=["quadrant.2"]),
                    limit=1,
                    lock="quad",
                    order_by="first",
                ),
                dict(
                    scene=RandomScene.factory(
                        RandomSceneParams(
                            candidates=[
                                RandomSceneCandidate(
                                    scene=BSP.factory(
                                        BSPParams(
                                            rooms=8, min_room_size=4, min_room_size_ratio=0.2, max_room_size_ratio=0.5
                                        )
                                    ),
                                    weight=1.0,
                                ),
                                RandomSceneCandidate(
                                    scene=BSP.factory(
                                        BSPParams(
                                            rooms=10,
                                            min_room_size=5,
                                            min_room_size_ratio=0.25,
                                            max_room_size_ratio=0.55,
                                        )
                                    ),
                                    weight=1.0,
                                ),
                            ]
                        )
                    ),
                    where=AreaWhere(tags=["quadrant.3"]),
                    limit=1,
                    lock="quad",
                    order_by="first",
                ),
                # Randomly assign each quadrant a distinct resource type and place them
                dict(
                    scene=QuadrantResources.factory(
                        QuadrantResourcesParams(
                            resource_types=["generator_red", "generator_blue", "generator_green", "lab"],
                            count_per_quadrant=6,
                            k=2.5,
                            min_radius=4,
                            clearance=1,
                        )
                    ),
                    where=AreaWhere(tags=["quadrant"]),
                    lock="resources",
                    order_by="first",
                ),
                # Ensure connectivity across the whole inner map area
                dict(
                    scene=MakeConnected.factory(MakeConnectedParams()),
                    where="full",
                    lock="finalize",
                    order_by="first",
                ),
            ],
        ),
    )

    return cfg
