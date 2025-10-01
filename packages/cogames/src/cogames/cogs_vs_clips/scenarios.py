from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple

from cogames.cogs_vs_clips.stations import (
    assembler,
    carbon_ex_dep,
    carbon_extractor,
    charger,
    chest,
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
from mettagrid.util.char_encoder import grid_object_to_char


_RANDOM_MAP_CACHE: Dict[Tuple[int, int, int, Tuple[Tuple[str, int], ...]], Deque[AsciiMapBuilder.Config]] = {}
_NUM_PREGENERATED_RANDOM_MAPS = 8
_MAX_BIOME_DIMENSION = 100


def _game_map_to_ascii(game_map) -> list[list[str]]:
    return [[grid_object_to_char(cell) for cell in row] for row in game_map.grid]


def _materialize_random_map(
    num_agents: int,
    width: int,
    height: int,
    objects: dict[str, int],
) -> AsciiMapBuilder.Config:
    key = (num_agents, width, height, tuple(sorted(objects.items())))
    cache = _RANDOM_MAP_CACHE.get(key)
    if cache is None or not cache:
        cache = deque()
        base_seed = 42
        for offset in range(_NUM_PREGENERATED_RANDOM_MAPS):
            cfg = RandomMapBuilder.Config(
                width=width,
                height=height,
                agents=num_agents,
                objects=objects,
                seed=base_seed + offset,
            )
            builder = cfg.create()
            ascii_map = _game_map_to_ascii(builder.build())
            cache.append(AsciiMapBuilder.Config(map_data=ascii_map))
        _RANDOM_MAP_CACHE[key] = cache
    ascii_cfg = cache[0]
    cache.rotate(-1)
    return ascii_cfg


def _crop_ascii_map(map_data: List[List[str]], max_dim: int) -> List[List[str]]:
    if not map_data:
        return map_data
    height = len(map_data)
    width = len(map_data[0])
    if height <= max_dim and width <= max_dim:
        return map_data

    crop_h = min(max_dim, height)
    crop_w = min(max_dim, width)
    top = max(0, (height - crop_h) // 2)
    left = max(0, (width - crop_w) // 2)
    cropped = [row[left:left + crop_w] for row in map_data[top:top + crop_h]]
    return cropped


RESOURCE_REWARD_WEIGHTS: dict[str, float] = {
    "carbon": 0.05,
    "oxygen": 0.05,
    "germanium": 0.08,
    "silicon": 0.08,
    "decoder": 0.1,
    "modulator": 0.1,
    "resonator": 0.1,
    "scrambler": 0.1,
}


def _base_game_config(num_agents: int, map_builder) -> MettaGridConfig:
    """Shared base configuration for all game types."""

    heart_reward = 1.0
    stats_rewards: dict[str, float] = {
        "heart.gained": heart_reward,
        "heart.put": heart_reward * 0.5,
    }

    for resource, weight in RESOURCE_REWARD_WEIGHTS.items():
        stats_rewards[f"{resource}.gained"] = weight
        stats_rewards[f"{resource}.put"] = weight * 0.25

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
                # depleted variants
                "silicon_ex_dep": silicon_ex_dep(),
                "oxygen_ex_dep": oxygen_ex_dep(),
                "carbon_ex_dep": carbon_ex_dep(),
                "germanium_ex_dep": germanium_ex_dep(),
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
                        "heart": heart_reward,
                    },
                    stats=stats_rewards,
                ),
                initial_inventory={
                    "energy": 100,
                },
            ),
        )
    )


def make_game(
    num_cogs: int = 96,
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
    objects = {
        "assembler": num_assemblers,
        "charger": num_chargers,
        "carbon_extractor": num_carbon_extractors,
        "oxygen_extractor": num_oxygen_extractors,
        "germanium_extractor": num_germanium_extractors,
        "silicon_extractor": num_silicon_extractors,
        "chest": num_chests,
    }
    map_builder = _materialize_random_map(num_cogs, width, height, objects)
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


def make_game_from_map(map_name: str, num_agents: int = 96) -> MettaGridConfig:
    """Create a game configuration from a map file."""
    maps_dir = Path(__file__).parent.parent / "maps"
    map_path = maps_dir / map_name
    map_cfg = AsciiMapBuilder.Config.from_uri(str(map_path))
    cropped_map = _crop_ascii_map(map_cfg.map_data, _MAX_BIOME_DIMENSION)
    map_builder = AsciiMapBuilder.Config(map_data=cropped_map)
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
        "machina_1": make_game_from_map("cave_base_50.map"),
        "machina_1_big": make_game_from_map("canidate1_500_stations.map"),
        "machina_2_bigger": make_game_from_map("canidate1_1000_stations.map"),
        "machina_3_big": make_game_from_map("canidate2_500_stations.map"),
        "machina_4_bigger": make_game_from_map("canidate2_1000_stations.map"),
        "machina_5_big": make_game_from_map("canidate3_500_stations.map"),
        "machina_6_bigger": make_game_from_map("canidate3_1000_stations.map"),
        "machina_7_big": make_game_from_map("canidate4_500_stations.map"),
    }
