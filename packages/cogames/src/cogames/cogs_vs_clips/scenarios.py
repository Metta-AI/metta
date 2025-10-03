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


def _base_game_config(num_agents: int) -> MettaGridConfig:
    """Shared base configuration for all game types."""

    heart_gain_reward = 5.0
    heart_deposit_reward = 7.5
    chest_reward = 2.5
    stats_rewards: dict[str, float] = {
        "heart.gained": heart_gain_reward,
        "heart.put": heart_deposit_reward,
        "chest.heart.amount": chest_reward,
    }

    return MettaGridConfig(
        game=GameConfig(
            resource_names=resources,
            num_agents=num_agents,
            actions=ActionsConfig(
                move=ActionConfig(consumed_resources={"energy": 2}),
                noop=ActionConfig(),
                change_glyph=ChangeGlyphActionConfig(number_of_glyphs=len(glyphs.GLYPHS)),
            ),
            resource_loss_prob=0.01,
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
                rewards=AgentRewards(stats=stats_rewards),
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
    width: int = 6,
    height: int = 6,
    num_assemblers: int = 1,
    num_chargers: int = 1,
    num_carbon_extractors: int = 0,
    num_oxygen_extractors: int = 0,
    num_germanium_extractors: int = 0,
    num_silicon_extractors: int = 0,
    num_chests: int = 1,
) -> MettaGridConfig:
    cfg = _base_game_config(num_cogs)
    max_border = max(0, min(width, height) // 2 - 1)
    border_width = min(1, max_border)

    object_counts = {
        "assembler": num_assemblers,
        "chest": num_chests,
        "charger": num_chargers,
        "carbon_extractor": num_carbon_extractors,
        "oxygen_extractor": num_oxygen_extractors,
        "germanium_extractor": num_germanium_extractors,
        "silicon_extractor": num_silicon_extractors,
    }

    filtered_objects = {name: count for name, count in object_counts.items() if count > 0}

    map_builder = RandomMapBuilder.Config(
        width=width,
        height=height,
        agents=num_cogs,
        border_width=border_width,
        objects=filtered_objects,
        seed=42,
    )
    cfg.game.map_builder = map_builder
    cfg.game.max_steps *= 20
    return cfg


def tutorial_assembler_simple(num_cogs: int = 1) -> MettaGridConfig:
    cfg = make_game(num_cogs=num_cogs)
    assembler_cfg = assembler()
    assembler_cfg.recipes = [
        (["Any"], RecipeConfig(input_resources={"energy": 1}, output_resources={"heart": 1}, cooldown=1))
    ]
    cfg.game.objects["assembler"] = assembler_cfg
    cfg.game.objects["chest"] = chest()
    return cfg


def tutorial_assembler_complex(num_cogs: int = 1) -> MettaGridConfig:
    cfg = make_game(num_cogs=num_cogs)
    assembler_cfg = assembler()
    assembler_cfg.recipes = [
        (["Any"], RecipeConfig(input_resources={"energy": 1}, output_resources={"heart": 1}, cooldown=1))
    ]
    cfg.game.objects["assembler"] = assembler_cfg
    cfg.game.objects["chest"] = chest()
    return cfg


def make_game_from_map(map_name: str, num_agents: int = 4) -> MettaGridConfig:
    """Create a game configuration from a map file."""

    # Build the full config first to get the objects
    config = _base_game_config(num_agents)

    maps_dir = Path(__file__).parent.parent / "maps"
    map_path = maps_dir / map_name
    map_builder = AsciiMapBuilder.Config.from_uri(
        str(map_path), {o.map_char: o.name for o in config.game.objects.values()}
    )
    config.game.map_builder = map_builder
    config.game.max_steps *= 20

    return config


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
