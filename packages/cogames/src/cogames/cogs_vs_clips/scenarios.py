from pathlib import Path
from typing import Any, Mapping

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
from cogames.scalable_astroid import ScalableAstroidParams, make_scalable_arena
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


def _merge_scalable_params(
    base: ScalableAstroidParams,
    override: ScalableAstroidParams | None,
) -> ScalableAstroidParams:
    if override is None:
        return base

    update_data = override.model_dump(exclude_unset=True)
    filtered = {key: value for key, value in update_data.items() if value is not None}

    if not filtered:
        return base

    return base.model_copy(update=filtered)


def _make_scalable_with_overrides(
    overrides: Mapping[str, Any] | None,
    *,
    width: int,
    height: int,
    num_agents: int,
    seed: int | None,
    params: ScalableAstroidParams | None,
) -> MettaGridConfig:
    override_dict = dict(overrides or {})

    base_params = params or ScalableAstroidParams()
    param_updates = {
        key: value for key, value in override_dict.items() if key in base_params.model_fields and value is not None
    }
    merged_params = base_params.model_copy(update=param_updates)

    return make_scalable_arena(
        width=int(override_dict.get("width", width)) if "width" in override_dict else width,
        height=int(override_dict.get("height", height)) if "height" in override_dict else height,
        num_agents=int(override_dict.get("num_agents", num_agents)) if "num_agents" in override_dict else num_agents,
        seed=override_dict.get("seed", seed),
        params=merged_params,
    )


def make_energy_rich_arena(
    width: int = 300,
    height: int = 300,
    num_agents: int = 4,
    seed: int | None = None,
    params: ScalableAstroidParams | None = None,
) -> MettaGridConfig:
    """Arena variant with abundant chargers and dense urban zones."""

    overrides = {
        "width": width,
        "height": height,
        "num_agents": num_agents,
        "seed": seed,
        "extractor_coverage": 0.12,
        "extractor_names": [
            "charger",
            "carbon_extractor",
            "oxygen_extractor",
            "germanium_extractor",
            "silicon_extractor",
        ],
        "extractor_weights": {
            "charger": 5.0,
            "carbon_extractor": 1.0,
            "oxygen_extractor": 1.0,
            "germanium_extractor": 0.8,
            "silicon_extractor": 0.8,
        },
        "primary_zone_weights": {
            "city": 3.0,
            "city_dense": 2.0,
            "desert": 1.0,
            "bsp": 1.0,
        },
        "secondary_zone_weights": {
            "city": 2.0,
            "city_dense": 1.5,
            "forest": 1.0,
        },
    }

    return _make_scalable_with_overrides(
        overrides,
        width=width,
        height=height,
        num_agents=num_agents,
        seed=seed,
        params=params,
    )


def make_resource_depleted_arena(
    width: int = 300,
    height: int = 300,
    num_agents: int = 4,
    seed: int | None = None,
    params: ScalableAstroidParams | None = None,
) -> MettaGridConfig:
    """Arena variant with mostly depleted extractor deposits and sparse chargers."""

    overrides = {
        "width": width,
        "height": height,
        "num_agents": num_agents,
        "seed": seed,
        "extractor_coverage": 0.1,
        "extractor_names": [
            "carbon_ex_dep",
            "oxygen_ex_dep",
            "germanium_ex_dep",
            "silicon_ex_dep",
            "charger",
        ],
        "extractor_weights": {
            "carbon_ex_dep": 1.5,
            "oxygen_ex_dep": 1.5,
            "germanium_ex_dep": 1.0,
            "silicon_ex_dep": 1.0,
            "charger": 0.6,
        },
        "extractor_jitter": 0,
        "primary_zone_weights": {
            "caves": 2.5,
            "bsp": 1.5,
            "desert": 1.0,
        },
        "secondary_zone_weights": {
            "caves": 2.0,
            "radial": 1.0,
            "maze": 1.0,
        },
        "tertiary_zone_weights": {
            "caves": 1.5,
            "bsp": 1.0,
        },
    }

    return _make_scalable_with_overrides(
        overrides,
        width=width,
        height=height,
        num_agents=num_agents,
        seed=seed,
        params=params,
    )


def make_energy_poor_arena(
    width: int = 300,
    height: int = 300,
    num_agents: int = 4,
    seed: int | None = None,
    params: ScalableAstroidParams | None = None,
) -> MettaGridConfig:
    """Arena variant with scarce chargers and emphasis on natural biomes."""

    overrides = {
        "width": width,
        "height": height,
        "num_agents": num_agents,
        "seed": seed,
        "extractor_coverage": 0.045,
        "extractor_names": [
            "charger",
            "carbon_extractor",
            "oxygen_extractor",
            "germanium_extractor",
            "silicon_extractor",
        ],
        "extractor_weights": {
            "charger": 0.4,
            "carbon_extractor": 1.4,
            "oxygen_extractor": 1.2,
            "germanium_extractor": 1.0,
            "silicon_extractor": 1.0,
        },
        "primary_zone_weights": {
            "forest": 2.0,
            "caves": 1.8,
            "radial": 1.2,
        },
        "secondary_zone_weights": {
            "forest": 1.8,
            "maze": 1.2,
            "city": 0.6,
        },
        "tertiary_zone_weights": {
            "forest": 1.6,
            "caves": 1.4,
        },
        "dungeon_zone_weights": {
            "maze": 1.5,
            "radial": 1.5,
            "dense": 1.0,
        },
    }

    return _make_scalable_with_overrides(
        overrides,
        width=width,
        height=height,
        num_agents=num_agents,
        seed=seed,
        params=params,
    )


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
