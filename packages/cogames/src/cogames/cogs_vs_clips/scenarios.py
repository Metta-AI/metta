from pathlib import Path

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
    GameConfig,
    MettaGridConfig,
    RecipeConfig,
    WallConfig,
)
from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.map_builder.random import RandomMapBuilder

MAPS_DIR = Path(__file__).resolve().parent.parent / "maps"


def _base_game_config(num_agents: int) -> MettaGridConfig:
    """Shared base configuration for all game types."""
    return MettaGridConfig(
        game=GameConfig(
            resource_names=resources,
            num_agents=num_agents,
            actions=ActionsConfig(
                move=ActionConfig(consumed_resources={"energy": 2}),
                noop=ActionConfig(),
                change_glyph=ChangeGlyphActionConfig(number_of_glyphs=16),
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


def _char_to_name_for(base_config: MettaGridConfig) -> dict[str, str]:
    """Build a char->name map from object configs plus agent and spawn defaults."""
    char_to_name = {obj.map_char: obj.name for obj in base_config.game.objects.values()}
    char_to_name.setdefault("@", "agent.agent")
    char_to_name.setdefault("%", "empty")
    return char_to_name


def _map_name_to_file() -> dict[str, str]:
    """Single source of truth for map-based scenarios."""
    return {
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


def make_game_from_map(map_name: str, num_agents: int = 4, dynamic_spawn: bool = False) -> MettaGridConfig:
    map_path = MAPS_DIR / map_name
    raw_lines = [line.rstrip("\n") for line in map_path.read_text(encoding="utf-8").splitlines() if line]

    if not raw_lines:
        raise ValueError(f"Map '{map_name}' is empty")

    # Count existing agents on the map
    agent_count = sum(line.count("@") for line in raw_lines)

    # When dynamic_spawn is requested, pass a target so the builder promotes '%' to '@'.
    # Otherwise, leave the map as-is and use the pre-placed '@' agents in the file.
    builder_target = num_agents if dynamic_spawn else None
    effective_agents = num_agents if dynamic_spawn else (agent_count or num_agents)

    base_config = _base_game_config(effective_agents)
    char_to_name = _char_to_name_for(base_config)

    map_builder = AsciiMapBuilder.Config.from_uri(
        str(map_path),
        char_to_name_map=char_to_name,
        target_agents=builder_target,
    )

    base_config.game.map_builder = map_builder
    base_config.game.num_agents = effective_agents

    return base_config


def make_game_from_map_with_agents(map_name: str, num_agents: int) -> MettaGridConfig:
    return make_game_from_map(map_name, num_agents=num_agents, dynamic_spawn=True)


def games() -> dict[str, MettaGridConfig]:
    games_dict: dict[str, MettaGridConfig] = {}

    # Tutorials
    games_dict["assembler_1_simple"] = tutorial_assembler_complex(num_cogs=1)
    games_dict["assembler_1_complex"] = tutorial_assembler_simple(num_cogs=1)
    games_dict["assembler_2_simple"] = tutorial_assembler_simple(num_cogs=4)
    games_dict["assembler_2_complex"] = tutorial_assembler_complex(num_cogs=4)

    # Map-based scenarios (default to 4 dynamic agents)
    for name, filename in _map_name_to_file().items():
        games_dict[name] = make_game_from_map(filename, num_agents=4, dynamic_spawn=True)

    # Predefined variants
    variant_agents_by_name: dict[str, list[int]] = {
        "machina_1": [8, 16],
        "training_facility_1": [8, 16],
    }
    for base_name, agent_counts in variant_agents_by_name.items():
        filename = _map_name_to_file()[base_name]
        for count in agent_counts:
            games_dict[f"{base_name}_{count}agents"] = make_game_from_map(
                filename, num_agents=count, dynamic_spawn=True
            )

    return games_dict


def supports_dynamic_spawn(game_name: str) -> bool:
    """Return True if the named game is a map-based scenario that supports dynamic spawning."""
    return game_name in _map_name_to_file()


def make_map_game(game_name: str, num_agents: int, dynamic_spawn: bool = True) -> MettaGridConfig:
    """Create a map-based game by name, optionally enabling dynamic spawn."""
    name_to_map = _map_name_to_file()
    if game_name not in name_to_map:
        raise ValueError(f"Game '{game_name}' is not a recognized map-based scenario")
    return make_game_from_map(name_to_map[game_name], num_agents=num_agents, dynamic_spawn=dynamic_spawn)
