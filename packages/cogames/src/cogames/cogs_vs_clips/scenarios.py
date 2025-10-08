from pathlib import Path

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

MAPS_DIR = Path(__file__).resolve().parent.parent / "maps"


def _base_game_config(num_cogs: int, clipping_rate: float) -> MettaGridConfig:
    """Shared base configuration for all game types."""
    return MettaGridConfig(
        game=GameConfig(
            resource_names=resources,
            num_agents=num_cogs,
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
                # clipped variants
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
                rewards=AgentRewards(stats={"chest.heart.amount": 1 / num_cogs}),
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
                    RecipeConfig(input_resources={"decoder": 1}, cooldown=1),
                    RecipeConfig(input_resources={"modulator": 1}, cooldown=1),
                    RecipeConfig(input_resources={"scrambler": 1}, cooldown=1),
                    RecipeConfig(input_resources={"resonator": 1}, cooldown=1),
                ],
                clip_rate=clipping_rate,
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
    clipping_rate: float = 0.0,
) -> MettaGridConfig:
    cfg = _base_game_config(num_cogs, clipping_rate)
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


def _map_name_to_file() -> dict[str, tuple[str, float]]:
    """Single source of truth for map-based scenarios."""
    return {
        # name: (filename, default_clip_rate)
        "machina_1": ("cave_base_50.map", 0.0),
        "machina_1_clipped": ("cave_base_50.map", 0.02),
        "machina_2": ("machina_100_stations.map", 0.0),
        "machina_3": ("machina_200_stations.map", 0.0),
        "machina_1_big": ("canidate1_500_stations.map", 0.0),
        "machina_2_bigger": ("canidate1_1000_stations.map", 0.0),
        "machina_3_big": ("canidate2_500_stations.map", 0.0),
        "machina_4_bigger": ("canidate2_1000_stations.map", 0.0),
        "machina_5_big": ("canidate3_500_stations.map", 0.0),
        "machina_6_bigger": ("canidate3_1000_stations.map", 0.0),
        "machina_7_big": ("canidate4_500_stations.map", 0.0),
        "training_facility_1": ("training_facility_open_1.map", 0.0),
        "training_facility_2": ("training_facility_open_2.map", 0.0),
        "training_facility_3": ("training_facility_open_3.map", 0.0),
        "training_facility_4": ("training_facility_tight_4.map", 0.0),
        "training_facility_5": ("training_facility_tight_5.map", 0.0),
        "training_facility_6": ("training_facility_clipped.map", 0.0),
    }


def make_game_from_map(
    map_name: str,
    num_cogs: int = 4,
    clipping_rate: float = 0.0,
    dynamic_spawn: bool = False,
) -> MettaGridConfig:
    """Create a game configuration from a map file, optionally enabling dynamic spawn via '%'."""
    map_path = MAPS_DIR / map_name
    if not map_path.exists():
        raise ValueError(f"Map '{map_name}' not found: {map_path}")

    base_config = _base_game_config(num_cogs, clipping_rate)
    char_to_name = _char_to_name_for(base_config)

    map_builder = AsciiMapBuilder.Config.from_uri(
        str(map_path),
        char_to_name_map=char_to_name,
        target_agents=(num_cogs if dynamic_spawn else None),
    )

    base_config.game.map_builder = map_builder
    base_config.game.num_agents = num_cogs
    return base_config


def make_game_from_map_with_agents(map_name: str, num_agents: int) -> MettaGridConfig:
    return make_game_from_map(map_name, num_cogs=num_agents, dynamic_spawn=True)


def games() -> dict[str, MettaGridConfig]:
    # Build from catalog to avoid duplication
    g: dict[str, MettaGridConfig] = {}

    # Tutorials (unchanged)
    g["assembler_1_simple"] = tutorial_assembler_complex(num_cogs=1)
    g["assembler_1_complex"] = tutorial_assembler_simple(num_cogs=1)
    g["assembler_2_simple"] = tutorial_assembler_simple(num_cogs=4)
    g["assembler_2_complex"] = tutorial_assembler_complex(num_cogs=4)

    # Map-based from catalog; default to non-dynamic maps, clip rate from catalog
    for name, (filename, clip_rate) in _map_name_to_file().items():
        # Only auto-generate canonical names; variants like *_clipped covered in catalog
        g[name] = make_game_from_map(filename, clipping_rate=clip_rate)

    return g


def supports_dynamic_spawn(game_name: str) -> bool:
    """Return True if the named game is a map-based scenario that supports dynamic spawning."""
    return game_name in _map_name_to_file()


def make_map_game(game_name: str, num_agents: int, dynamic_spawn: bool = True) -> MettaGridConfig:
    """Create a map-based game by name, optionally enabling dynamic spawn."""
    catalog = _map_name_to_file()
    if game_name not in catalog:
        raise ValueError(f"Game '{game_name}' is not a recognized map-based scenario")
    filename, clip_rate = catalog[game_name]
    return make_game_from_map(filename, num_cogs=num_agents, clipping_rate=clip_rate, dynamic_spawn=dynamic_spawn)
