from pathlib import Path

from cogames.cogs_vs_clips.stations import (
    assembler,
    carbon_extractor,
    charger,
    chest,
    germanium_extractor,
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
from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.map_builder.random import RandomMapBuilder


def _base_game_config(num_agents: int, map_builder) -> MettaGridConfig:
    """Shared base configuration for all game types."""
    return MettaGridConfig(
        game=GameConfig(
            resource_names=resources,
            num_agents=num_agents,
            actions=ActionsConfig(
                move=ActionConfig(consumed_resources={"energy": 1}),
                noop=ActionConfig(),
                change_glyph=ChangeGlyphActionConfig(number_of_glyphs=16),
            ),
            objects={
                "wall": WallConfig(type_id=1),
                "charger": charger(),
                "carbon_extractor": carbon_extractor(),
                "oxygen_extractor": oxygen_extractor(),
                "germanium_extractor": germanium_extractor(),
                "silicon_extractor": silicon_extractor(),
                "chest": chest(),
                "assembler": assembler(),
            },
            map_builder=map_builder,
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


def make_game(
    num_cogs: int = 4,
    width: int = 10,
    height: int = 10,
    num_assemblers: int = 1,
    num_chargers: int = 0,
    num_carbon_extractors: int = 0,
    num_oxygen_extractors: int = 0,
    num_germanium_extractors: int = 0,
    num_silicon_extractors: int = 0,
    num_chests: int = 0,
) -> MettaGridConfig:
    map_builder = RandomMapBuilder.Config(
        width=width,
        height=height,
        agents=num_cogs,
        border_width=5,
        objects={
            "assembler": num_assemblers,
            "charger": num_chargers,
            "carbon_extractor": num_carbon_extractors,
            "oxygen_extractor": num_oxygen_extractors,
            "germanium_extractor": num_germanium_extractors,
            "silicon_extractor": num_silicon_extractors,
            "chest": num_chests,
        },
        seed=42,
    )
    return _base_game_config(num_cogs, map_builder)


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


def make_game_from_map(map_name: str, num_agents: int = 4) -> MettaGridConfig:
    """Create a game configuration from a map file."""
    maps_dir = Path(__file__).parent.parent / "maps"
    map_path = maps_dir / map_name
    map_builder = AsciiMapBuilder.Config.from_uri(str(map_path))
    return _base_game_config(num_agents, map_builder)


def games() -> dict[str, MettaGridConfig]:
    return {
        "assembler_1_simple": tutorial_assembler_complex(num_cogs=1),
        "assembler_1_complex": tutorial_assembler_simple(num_cogs=1),
        "assembler_2_simple": tutorial_assembler_simple(num_cogs=4),
        "assembler_2_complex": tutorial_assembler_complex(num_cogs=4),
        # "extractor_1cog_1resource": tutorial_extractor(num_cogs=1),""
        # "extractor_1cog_4resource": tutorial_extractor(num_cogs=1),
        # "harvest_1": tutorial_harvest(num_cogs=1),
        # "harvest_4": tutorial_harvest(num_cogs=4),
        # "base_1": tutorial_base(num_cogs=1),
        # "base_4": tutorial_base(num_cogs=4),
        # "forage_1": tutorial_forage(num_cogs=1),
        # "forage_4": tutorial_forage(num_cogs=4),
        # "chest_1": tutorial_chest(num_cogs=1),
        # "chest_4": tutorial_chest(num_cogs=4),
        # Biomes dungeon maps with stations
        "machina_1": make_game_from_map("canidate1_500_stations.map"),
        "machina_2": make_game_from_map("canidate1_1000_stations.map"),
        "machina_3": make_game_from_map("canidate2_500_stations.map"),
        "machina_4": make_game_from_map("canidate2_1000_stations.map"),
        "machina_5": make_game_from_map("canidate3_500_stations.map"),
        "machina_6": make_game_from_map("canidate3_1000_stations.map"),
        "machina_7": make_game_from_map("canidate4_500_stations.map"),
    }
