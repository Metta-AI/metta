from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel, ConfigDict

from cogames.cogs_vs_clips.missions import get_all_missions
from cogames.cogs_vs_clips.stations import assembler
from mettagrid.config.mettagrid_config import (
    MettaGridConfig,
    RecipeConfig,
)
from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.map_builder.random import RandomMapBuilder


def _get_base_cfg_from_mission(mission_name: str) -> MettaGridConfig:
    return MettaGridConfig(game=get_all_missions()[mission_name].game)


def make_game(
    mission_name: str,
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
    cfg = _get_base_cfg_from_mission(mission_name)
    cfg.game.num_agents = num_cogs
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
    cfg.game.map_builder = map_builder
    return cfg


def tutorial_assembler_simple(num_cogs: int = 1) -> Callable[[str], MettaGridConfig]:
    def generate(mission_name: str) -> MettaGridConfig:
        cfg = make_game(mission_name=mission_name, num_cogs=num_cogs, num_assemblers=1)
        cfg.game.objects["assembler"] = assembler()
        cfg.game.objects["assembler"].recipes = [
            (["Any"], RecipeConfig(input_resources={"battery_red": 3}, output_resources={"heart": 1}, cooldown=10))
        ]
        return cfg

    return generate


def tutorial_assembler_complex(num_cogs: int = 1) -> Callable[[str], MettaGridConfig]:
    def generate(mission_name: str) -> MettaGridConfig:
        cfg = make_game(
            mission_name=mission_name,
            num_cogs=num_cogs,
            num_assemblers=1,
            num_chests=1,
            num_carbon_extractors=1,
            num_oxygen_extractors=1,
            num_germanium_extractors=1,
            num_silicon_extractors=1,
        )
        cfg.game.objects["assembler"] = assembler()
        cfg.game.objects["assembler"].recipes = [
            (["Any"], RecipeConfig(input_resources={"battery_red": 3}, output_resources={"heart": 1}, cooldown=10))
        ]
        return cfg

    return generate


def make_game_from_map(map_name: str, num_agents: int = 4) -> Callable[[str], MettaGridConfig]:
    """Create a game configuration from a map file."""

    # Build the full config first to get the objects
    def generate(mission_name: str) -> MettaGridConfig:
        config = _get_base_cfg_from_mission(mission_name or "default")
        config.game.num_agents = num_agents

        maps_dir = Path(__file__).parent.parent / "maps"
        map_path = maps_dir / map_name
        map_builder = AsciiMapBuilder.Config.from_uri(
            str(map_path), {o.map_char: o.name for o in config.game.objects.values()}
        )
        config.game.map_builder = map_builder
        return config

    return generate


class GameCatalogEntry(BaseModel):
    map_name: str

    # Takes in a mission name and returns a game configuration
    generate: Callable[[str], MettaGridConfig]

    default_mission: str = "default"

    model_config = ConfigDict(arbitrary_types_allowed=True)


GAMES_CATALOG: list[GameCatalogEntry] = [
    GameCatalogEntry(map_name="assembler_1_simple", generate=tutorial_assembler_complex(num_cogs=1)),
    GameCatalogEntry(map_name="assembler_1_complex", generate=tutorial_assembler_simple(num_cogs=1)),
    GameCatalogEntry(map_name="assembler_2_simple", generate=tutorial_assembler_simple(num_cogs=4)),
    GameCatalogEntry(map_name="assembler_2_complex", generate=tutorial_assembler_complex(num_cogs=4)),
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
    GameCatalogEntry(map_name="machina_1", generate=make_game_from_map("cave_base_50.map")),
    GameCatalogEntry(map_name="machina_2", generate=make_game_from_map("machina_100_stations.map")),
    GameCatalogEntry(map_name="machina_3", generate=make_game_from_map("machina_200_stations.map")),
    GameCatalogEntry(map_name="machina_1_big", generate=make_game_from_map("canidate1_500_stations.map")),
    GameCatalogEntry(map_name="machina_2_bigger", generate=make_game_from_map("canidate1_1000_stations.map")),
    GameCatalogEntry(map_name="machina_3_big", generate=make_game_from_map("canidate2_500_stations.map")),
    GameCatalogEntry(map_name="machina_4_bigger", generate=make_game_from_map("canidate2_1000_stations.map")),
    GameCatalogEntry(map_name="machina_5_big", generate=make_game_from_map("canidate3_500_stations.map")),
    GameCatalogEntry(map_name="machina_6_bigger", generate=make_game_from_map("canidate3_1000_stations.map")),
    GameCatalogEntry(map_name="machina_7_big", generate=make_game_from_map("canidate4_500_stations.map")),
    GameCatalogEntry(map_name="training_facility_1", generate=make_game_from_map("training_facility_open_1.map")),
    GameCatalogEntry(map_name="training_facility_2", generate=make_game_from_map("training_facility_open_2.map")),
    GameCatalogEntry(map_name="training_facility_3", generate=make_game_from_map("training_facility_open_3.map")),
    GameCatalogEntry(map_name="training_facility_4", generate=make_game_from_map("training_facility_tight_4.map")),
    GameCatalogEntry(map_name="training_facility_5", generate=make_game_from_map("training_facility_tight_5.map")),
]


def games(mission_name: Optional[str], *args: Any, **kwargs: Any) -> dict[str, MettaGridConfig]:
    return {
        game.map_name: game.generate(mission_name or game.default_mission, *args, **kwargs) for game in GAMES_CATALOG
    }
