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


def add_easy_heart_recipe(cfg: MettaGridConfig) -> None:
    assembler_cfg = cfg.game.objects.get("assembler")
    if assembler_cfg is None:
        return

    input_options = (
        {"carbon": 1},
        {"oxygen": 1},
        {"germanium": 1},
        {"silicon": 1},
        {"energy": 1},
    )

    for _, recipe in assembler_cfg.recipes:
        if recipe.output_resources.get("heart") and recipe.input_resources in input_options:
            return

    for inputs in input_options:
        assembler_cfg.recipes.insert(
            0,
            (["Any"], RecipeConfig(input_resources=inputs, output_resources={"heart": 1}, cooldown=1)),
        )


def add_shaped_rewards(cfg: MettaGridConfig) -> None:
    agent_cfg = cfg.game.agent
    stats = dict(agent_cfg.rewards.stats or {})

    stats["heart.gained"] = 1.0
    stats["heart.put"] = 0.0
    stats["chest.heart.amount"] = 1 / cfg.game.num_agents

    shaped_reward = 0.25
    stats.update(
        {
            "carbon.gained": shaped_reward,
            "oxygen.gained": shaped_reward,
            "germanium.gained": shaped_reward,
            "silicon.gained": shaped_reward,
            "energy.gained": shaped_reward / 5,
        }
    )

    agent_cfg.rewards.stats = stats


def _base_game_config(num_cogs: int, clipping_rate: float) -> MettaGridConfig:
    """Shared base configuration for all game types."""

    heart_gain_reward = 5.0
    heart_deposit_reward = 7.5
    chest_reward = 1 / num_cogs
    stats_rewards: dict[str, float] = {
        "heart.gained": heart_gain_reward,
        "heart.put": heart_deposit_reward,
        "chest.heart.amount": chest_reward,
    }
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
                clip_rate=clipping_rate,
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
    clipping_rate: float = 0.0,
) -> MettaGridConfig:
    cfg = _base_game_config(num_cogs, clipping_rate=clipping_rate)
    max_border = max(0, min(width, height) // 2 - 1)
    border_width = min(1, max_border)

    map_builder = RandomMapBuilder.Config(
        width=width,
        height=height,
        agents=num_cogs,
        border_width=border_width,
        objects={
            name: count
            for name, count in {
                "assembler": num_assemblers,
                "chest": num_chests,
                "charger": num_chargers,
                "carbon_extractor": num_carbon_extractors,
                "oxygen_extractor": num_oxygen_extractors,
                "germanium_extractor": num_germanium_extractors,
                "silicon_extractor": num_silicon_extractors,
            }.items()
            if count > 0
        },
        seed=42,
    )
    cfg.game.map_builder = map_builder
    return cfg


def tutorial_assembler_simple(num_cogs: int = 1) -> MettaGridConfig:
    cfg = make_game(num_cogs=num_cogs, num_assemblers=1)
    cfg.game.objects["assembler"] = assembler()
    cfg.game.objects["assembler"].recipes = [
        (["Any"], RecipeConfig(input_resources={"energy": 1}, output_resources={"heart": 1}, cooldown=1))
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
        (["Any"], RecipeConfig(input_resources={"energy": 1}, output_resources={"heart": 1}, cooldown=1))
    ]
    return cfg


def make_game_from_map(map_name: str, num_cogs: int = 4, clipping_rate: float = 0.0) -> MettaGridConfig:
    """Create a game configuration from a map file."""

    # Build the full config first to get the objects
    config = _base_game_config(num_cogs, clipping_rate)

    maps_dir = Path(__file__).parent.parent / "maps"
    map_path = maps_dir / map_name
    map_builder = AsciiMapBuilder.Config.from_uri(
        str(map_path), {o.map_char: o.name for o in config.game.objects.values()}
    )
    config.game.map_builder = map_builder

    return config


def games() -> dict[str, MettaGridConfig]:
    return {
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
        "training_facility_1": make_game_from_map("training_facility_open_1.map"),
        "training_facility_2": make_game_from_map("training_facility_open_2.map"),
        "training_facility_3": make_game_from_map("training_facility_open_3.map"),
        "training_facility_4": make_game_from_map("training_facility_tight_4.map"),
        "training_facility_5": make_game_from_map("training_facility_tight_5.map"),
        "training_facility_6": make_game_from_map("training_facility_clipped.map"),
        # Biomes dungeon maps with stations
        "machina_1_clipped": make_game_from_map("cave_base_50.map", clipping_rate=0.02),
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
    }
