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
from mettagrid.mapgen.scenes.distance_balance import DistanceBalance, DistanceBalanceParams
from mettagrid.mapgen.scenes.enforce_symmetry import EnforceSymmetry, EnforceSymmetryParams
from mettagrid.mapgen.scenes.make_connected import MakeConnected, MakeConnectedParams
from mettagrid.mapgen.scenes.quadrant_layout import QuadrantLayout, QuadrantLayoutParams
from mettagrid.mapgen.scenes.quadrant_resources import QuadrantResources, QuadrantResourcesParams
from mettagrid.mapgen.scenes.quadrants import Quadrants, QuadrantsParams
from mettagrid.mapgen.scenes.relabel_converters import RelabelConverters, RelabelConvertersParams
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
        "machina_symmetry_sanctum": machina_symmetry_sanctum(num_cogs=4),
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
                # Quadrant population from a library (BSP/Maze/VariedTerrain)
                dict(
                    scene=QuadrantLayout.factory(
                        QuadrantLayoutParams(
                            weight_bsp10=0.8,
                            weight_bsp8=0.6,
                            weight_maze=1.6,
                            weight_terrain_balanced=0.0,
                            weight_terrain_maze=0.0,
                        )
                    ),
                    where=AreaWhere(tags=["quadrant"]),
                    lock="quad",
                    order_by="first",
                ),
                # Enforce symmetry over terrain only (before resources)
                dict(
                    scene=EnforceSymmetry.factory(EnforceSymmetryParams(horizontal=False, vertical=False)),
                    where="full",
                    lock="symmetry",
                    order_by="first",
                ),
                # Place a single placeholder converter type everywhere for symmetry
                dict(
                    scene=QuadrantResources.factory(
                        QuadrantResourcesParams(
                            resource_types=["generator_green"],
                            forced_type="generator_green",
                            count_per_quadrant=6,
                            k=2.5,
                            min_radius=6,
                            clearance=1,
                        )
                    ),
                    where=AreaWhere(tags=["quadrant"]),
                    lock="resources",
                    order_by="first",
                ),
                # Mirror resource placements so converter positions are symmetric
                dict(
                    scene=EnforceSymmetry.factory(EnforceSymmetryParams(horizontal=True, vertical=True)),
                    where="full",
                    lock="symmetry_resources",
                    order_by="first",
                ),
                # Ensure connectivity across the whole inner map area
                dict(
                    scene=MakeConnected.factory(MakeConnectedParams()),
                    where="full",
                    lock="finalize",
                    order_by="first",
                ),
                # Stamp the central sanctum/base last so it overrides terrain
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
                # Relabel converters to target mix while keeping symmetric placements (run after base)
                dict(
                    scene=RelabelConverters.factory(
                        RelabelConvertersParams(
                            target_counts={
                                "generator_red": 4,
                                "generator_blue": 4,
                                "generator_green": 4,
                                "lab": 4,
                            }
                        )
                    ),
                    where="full",
                    lock="relabel",
                    order_by="first",
                ),
                # Analyze and optionally balance converter distances from the altar
                dict(
                    scene=DistanceBalance.factory(
                        DistanceBalanceParams(
                            converter_types=["generator_red", "generator_blue", "generator_green", "lab"],
                            tolerance=8.0,
                            balance=True,
                            carves_per_type=1,
                            carve_width=1,
                        )
                    ),
                    where="full",
                    lock="post",
                    order_by="first",
                ),
            ],
        ),
    )

    return cfg


def machina_symmetry_sanctum(num_cogs: int = 4) -> MettaGridConfig:
    """Same as machina_sanctum but enforces both horizontal and vertical symmetry."""
    cfg = machina_sanctum(num_cogs=num_cogs)

    # Flip the EnforceSymmetry node to both axes
    children = cfg.game.map_builder.root.children or []
    for child in children:
        try:
            if getattr(child.scene.type, "__name__", "") == "EnforceSymmetry":
                child.scene.params.horizontal = True
                child.scene.params.vertical = True
                break
        except Exception:
            pass
    cfg.game.map_builder.root.children = children
    return cfg
