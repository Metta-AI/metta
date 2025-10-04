from pathlib import Path

from cogames.cogs_vs_clips.map_utils import DynamicAgentAsciiMapBuilder
from cogames.cogs_vs_clips import glyphs
from cogames.cogs_vs_clips.stations import (
    assembler,
    carbon_ex_dep,
    carbon_extractor,
    charger,
    chest,
    chest_carbon,
    chest_germanium,
    chest_oxygen,
    chest_silicon,
    clipped_carbon_extractor,
    clipped_germanium_extractor,
    clipped_oxygen_extractor,
    clipped_silicon_extractor,
    germanium_ex_dep,
    germanium_extractor,
    oxygen_ex_dep,
    oxygen_extractor,
    resources,
    silicon_ex_dep,
    silicon_extractor,
)
from mettagrid.config.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    ChangeGlyphActionConfig,
    ClipperConfig,
    GameConfig,
    MettaGridConfig,
    RecipeConfig,
    WallConfig,
)
from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.map_builder.random import RandomMapBuilder

MAP_GAME_FILES: dict[str, str] = {
    "machina_1": "cave_base_50.map",
    "machina_2": "machina_100_stations.map",
    "machina_3": "machina_200_stations.map",
    "machina_1_big": "canidate1_500_stations.map",
    "machina_2_bigger": "canidate1_1000_stations.map",
    "machina_3_big": "canidate2_500_stations.map",
    "machina_4_bigger": "canidate2_1000_stations.map",
    "machina_5_big": "canidate3_500_stations.map",
    "machina_6_bigger": "canidate3_1000_stations.map",
    "machina_7_big": "canidate4_500_stations.map",
    "training_facility_1": "training_facility_open_1.map",
    "training_facility_2": "training_facility_open_2.map",
    "training_facility_3": "training_facility_open_3.map",
    "training_facility_4": "training_facility_tight_4.map",
    "training_facility_5": "training_facility_tight_5.map",
}

MAPS_DIR = Path(__file__).resolve().parent.parent / "maps"


def supports_dynamic_spawn(game_key: str) -> bool:
    """Return whether the provided game key supports dynamic agent spawning."""

    return game_key in MAP_GAME_FILES


def _base_game_config(num_agents: int) -> MettaGridConfig:
    """Shared base configuration for all game types."""
    return MettaGridConfig(
        game=GameConfig(
            resource_names=resources,
            num_agents=num_agents,
            actions=ActionsConfig(
                move=ActionConfig(consumed_resources={"energy": 2}),
                noop=ActionConfig(),
                change_glyph=ChangeGlyphActionConfig(number_of_glyphs=len(glyphs.GLYPHS)),
            ),
            objects={
                "wall": WallConfig(name="wall", type_id=1, map_char="#", render_symbol="⬛"),
                "charger": charger(),
                "carbon_extractor": carbon_extractor(),
                "oxygen_extractor": oxygen_extractor(),
                "germanium_extractor": germanium_extractor(),
                "silicon_extractor": silicon_extractor(),
                # depleted variants
                "silicon_ex_dep": silicon_ex_dep(),
                "oxygen_ex_dep": oxygen_ex_dep(),
                "carbon_ex_dep": carbon_ex_dep(),
                "germanium_ex_dep": germanium_ex_dep(),
                "clipped_carbon_extractor": clipped_carbon_extractor(),
                "clipped_oxygen_extractor": clipped_oxygen_extractor(),
                "clipped_germanium_extractor": clipped_germanium_extractor(),
                "clipped_silicon_extractor": clipped_silicon_extractor(),
                "chest": chest(),
                "chest_carbon": chest_carbon(),
                "chest_oxygen": chest_oxygen(),
                "chest_germanium": chest_germanium(),
                "chest_silicon": chest_silicon(),
                "assembler": assembler(),
            },
            agent=AgentConfig(
                resource_limits={
                    "heart": 1,
                    "energy": 100,
                    ("carbon", "oxygen", "germanium", "silicon"): 100,
                    ("scrambler", "modulator", "decoder", "resonator"): 5,
                },
                rewards=AgentRewards(
                    stats={"chest.heart.amount": 1},
                    # inventory={
                    #     "heart": 1,
                    # },
                ),
                initial_inventory={
                    "energy": 100,
                },
                shareable_resources=["energy"],
                inventory_regen_amounts={"energy": 1},
            ),
            inventory_regen_interval=1,
            # Enable clipper system to allow start_clipped assemblers to work
            clipper=ClipperConfig(
                unclipping_recipes=[
                    RecipeConfig(
                        input_resources={"decoder": 1},
                        cooldown=1,
                    ),
                    RecipeConfig(
                        input_resources={"modulator": 1},
                        cooldown=1,
                    ),
                    RecipeConfig(
                        input_resources={"scrambler": 1},
                        cooldown=1,
                    ),
                    RecipeConfig(
                        input_resources={"resonator": 1},
                        cooldown=1,
                    ),
                ],
                length_scale=10.0,
                cutoff_distance=0.0,
                clip_rate=0.0,  # Don't clip during gameplay, only use start_clipped
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
    cfg = _base_game_config(num_cogs)
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


def tutorial_assembler_simple(num_cogs: int = 1) -> MettaGridConfig:
    cfg = make_game(num_cogs=num_cogs, num_assemblers=1)
    cfg.game.objects["assembler"] = assembler()
    cfg.game.objects["assembler"].recipes = [
        (["Any"], RecipeConfig(input_resources={"battery_red": 3}, output_resources={"heart": 1}, cooldown=10))
    ]
    return cfg


def tutorial_assembler_complex(num_cogs: int = 1) -> MettaGridConfig:
    cfg = make_game(
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


def make_game_from_map(map_name: str, num_agents: int = 4, dynamic_spawn: bool = False) -> MettaGridConfig:
    """Create a game configuration from a map file.

    Args:
        map_name: Name of the map file to load
        num_agents: Number of agents to spawn
        dynamic_spawn: If True, spawn agents dynamically in center area instead of using map's @ markers
    """

    # Build the full config first to get the objects
    config = _base_game_config(num_agents)

    map_path = MAPS_DIR / map_name

    if dynamic_spawn:
        # Create a custom map builder that handles dynamic agent spawning
        map_builder = DynamicAgentAsciiMapBuilder.Config.from_uri(
            str(map_path), {o.map_char: o.name for o in config.game.objects.values()}, num_agents=num_agents
        )
    else:
        # Use standard ASCII map builder
        map_builder = AsciiMapBuilder.Config.from_uri(
            str(map_path), {o.map_char: o.name for o in config.game.objects.values()}
        )

    config.game.map_builder = map_builder

    return config


def make_map_game(game_key: str, num_agents: int = 4, dynamic_spawn: bool = False) -> MettaGridConfig:
    """Create a map-based game configuration by key."""

    map_file = MAP_GAME_FILES.get(game_key)
    if map_file is None:
        raise ValueError(f"Unknown map-based game: {game_key}")

    return make_game_from_map(map_file, num_agents=num_agents, dynamic_spawn=dynamic_spawn)


def make_game_from_map_with_agents(map_name: str, num_agents: int) -> MettaGridConfig:
    """Convenience function to create a game from a map with dynamic agent spawning.

    This function automatically enables dynamic_spawn=True and places the specified
    number of agents in the center area of the map, ignoring the original @ markers.

    Args:
        map_name: Name of the map file to load
        num_agents: Number of agents to spawn dynamically in the center
    """
    return make_game_from_map(map_name, num_agents=num_agents, dynamic_spawn=True)


def games() -> dict[str, MettaGridConfig]:
    games_dict: dict[str, MettaGridConfig] = {
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
        "machina_1": make_game_from_map("cave_base_50.map"),
        "machina_2": make_game_from_map("machina_100_stations.map"),
        "machina_3": make_game_from_map("machina_200_stations.map"),
        "machina_1_big": make_game_from_map("canidate1_500_stations.map"),
        "machina_2_bigger": make_game_from_map("canidate1_1000_stations.map"),
        "machina_3_big": make_game_from_map("canidate2_500_stations.map"),
        "machina_4_bigger": make_game_from_map("canidate2_1000_stations.map"),
        "machina_5_big": make_game_from_map("canidate3_500_stations.map"),
        "machina_6_bigger": make_game_from_map("canidate3_1000_stations.map"),
        "machina_7_big": make_game_from_map("canidate4_500_stations.map"),
        "training_facility_1": make_game_from_map("training_facility_open_1.map"),
        "training_facility_2": make_game_from_map("training_facility_open_2.map"),
        "training_facility_3": make_game_from_map("training_facility_open_3.map"),
        "training_facility_4": make_game_from_map("training_facility_tight_4.map"),
        "training_facility_5": make_game_from_map("training_facility_tight_5.map"),
        "training_facility_6": make_game_from_map("training_facility_clipped.map"),
    }

    # Add map-based games from shared mapping
    for game_key, map_filename in MAP_GAME_FILES.items():
        games_dict[game_key] = make_game_from_map(map_filename)

    # Dynamic agent spawning examples (agents spawn in center area)
    games_dict["machina_1_8agents"] = make_game_from_map(MAP_GAME_FILES["machina_1"], num_agents=8, dynamic_spawn=True)
    games_dict["machina_1_16agents"] = make_game_from_map(
        MAP_GAME_FILES["machina_1"], num_agents=16, dynamic_spawn=True
    )
    games_dict["training_facility_1_8agents"] = make_game_from_map(
        MAP_GAME_FILES["training_facility_1"], num_agents=8, dynamic_spawn=True
    )
    games_dict["training_facility_1_16agents"] = make_game_from_map(
        MAP_GAME_FILES["training_facility_1"], num_agents=16, dynamic_spawn=True
    )

    return games_dict
