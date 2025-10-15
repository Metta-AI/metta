from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Iterable

from cogames.cogs_vs_clips import glyphs, mission_modifications
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
    AnyGridObjectConfig,
    ChangeGlyphActionConfig,
    ClipperConfig,
    GameConfig,
    MettaGridConfig,
    RecipeConfig,
    WallConfig,
)
from mettagrid.map_builder.map_builder import MapBuilderConfig
from mettagrid.map_builder.random import RandomMapBuilder


def _get_default_map_objects() -> dict[str, AnyGridObjectConfig]:
    return {
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
    }


def _default_mission(*, num_cogs: int = 4, clip_rate: float = 0.0, **kwargs: dict[str, Any]) -> GameConfig:
    game = GameConfig(
        resource_names=resources,
        num_agents=num_cogs,
        actions=ActionsConfig(
            move=ActionConfig(consumed_resources={"energy": 2}),
            noop=ActionConfig(),
            change_glyph=ChangeGlyphActionConfig(number_of_glyphs=len(glyphs.GLYPHS)),
        ),
        objects=_get_default_map_objects(),
        agent=AgentConfig(
            resource_limits={
                "heart": 1,
                "energy": 100,
                ("carbon", "oxygen", "germanium", "silicon"): 100,
                ("scrambler", "modulator", "decoder", "resonator"): 5,
            },
            rewards=AgentRewards(
                stats={"chest.heart.amount": 1 / num_cogs},
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
            clip_rate=clip_rate,
        ),
    )

    return game


def energy_intensive(**kwargs: dict[str, Any]) -> GameConfig:
    game = _default_mission(**kwargs)  # type: ignore
    game.actions.move.consumed_resources = {"energy": 5}
    game.agent.resource_limits.update(
        {
            "heart": 1,
            ("carbon", "oxygen", "germanium", "silicon"): 2,
            ("scrambler", "modulator", "decoder", "resonator"): 5,
        }
    )
    game.agent.inventory_regen_amounts = {"energy": 3}
    return game


def get_mission_generator(mission: str = "default") -> Callable[..., GameConfig]:
    index = {
        "default": _default_mission,
        "energy_intensive": energy_intensive,
    }
    if mission not in index:
        raise ValueError(f"Mission {mission} not found")
    return index[mission]


def get_map_builder_for_site(site: str) -> MapBuilderConfig:
    """Get the map builder for a site. Site is the name of the site file in the maps directory."""
    maps_dir = Path(__file__).parent.parent / "maps"
    map_path = maps_dir / site

    return MapBuilderConfig.from_uri(str(map_path))


def get_random_map_builder(
    num_cogs: int = 4,
    width: int = 100,
    height: int = 100,
    num_assemblers: int = 1,
    num_chargers: int = 0,
    num_carbon_extractors: int = 0,
    num_oxygen_extractors: int = 0,
    num_germanium_extractors: int = 0,
    num_silicon_extractors: int = 0,
    num_chests: int = 0,
    **kwargs: dict[str, Any],
) -> MapBuilderConfig[RandomMapBuilder]:
    return RandomMapBuilder.Config(
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


class UserMap(ABC):
    name: str

    @property
    def default_mission(self) -> str:
        if not self.available_missions:
            raise ValueError(f"Map {self.name} has no missions")
        return self.available_missions[0]

    def generate_env(self, mission_name: str) -> MettaGridConfig:
        if mission_name not in self.available_missions:
            raise ValueError(f"Mission {mission_name} not found")
        return self._generate_env(mission_name)

    @property
    @abstractmethod
    def available_missions(self) -> list[str]:
        pass

    @abstractmethod
    def _generate_env(self, mission_name: str) -> MettaGridConfig:
        pass


class RandomUserMap(UserMap):
    def __init__(
        self,
        name: str,
        mission_args: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.name = name
        self._mission_args = mission_args or dict(default={})

    @property
    def available_missions(self) -> list[str]:
        return list(self._mission_args.keys())

    def _generate_env(self, mission_name: str) -> MettaGridConfig:
        args = dict(self._mission_args.get(mission_name, {}))
        easy = bool(args.pop("easy", False))
        shaped = bool(args.pop("shaped", False))
        base_mission = args.pop("base_mission", self.default_mission)
        game = get_mission_generator(base_mission)(**args)
        game.map_builder = get_random_map_builder(**args)
        cfg = MettaGridConfig(game=game)
        mission_modifications.apply_modifications(
            cfg, {"easy": easy, "shaped": shaped, "extend_max_steps": easy or shaped}
        )
        return cfg


class SiteUserMap(UserMap):
    def __init__(
        self,
        name: str,
        site: str,
        mission_args: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.name = name
        self._site = site
        self._mission_args = mission_args or dict(default={})

    @property
    def available_missions(self) -> list[str]:
        return list(self._mission_args.keys())

    def _generate_env(self, mission_name: str) -> MettaGridConfig:
        args = dict(self._mission_args.get(mission_name, {}))
        easy = bool(args.pop("easy", False))
        shaped = bool(args.pop("shaped", False))
        base_mission = args.pop("base_mission", self.default_mission)
        game = get_mission_generator(base_mission)(**args)
        game.map_builder = get_map_builder_for_site(self._site)
        cfg = MettaGridConfig(game=game)
        mission_modifications.apply_modifications(
            cfg, {"easy": easy, "shaped": shaped, "extend_max_steps": easy or shaped}
        )
        return cfg


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
    clip_rate: float = 0.0,
) -> MettaGridConfig:
    mission_args = dict(
        default=dict(
            num_cogs=num_cogs,
            width=width,
            height=height,
            num_assemblers=num_assemblers,
            num_chargers=num_chargers,
            num_carbon_extractors=num_carbon_extractors,
            num_oxygen_extractors=num_oxygen_extractors,
            num_germanium_extractors=num_germanium_extractors,
            num_silicon_extractors=num_silicon_extractors,
            num_chests=num_chests,
            clip_rate=clip_rate,
        )
    )
    return RandomUserMap(name="random", mission_args=mission_args).generate_env("default")


def _with_easy_shaped_variants(
    extra: Iterable[tuple[str, dict[str, Any]]] | None = None,
) -> dict[str, dict[str, Any]]:
    missions: list[tuple[str, dict[str, Any]]] = [("default", {})]
    if extra is not None:
        missions.extend((name, dict(values)) for name, values in extra)
    missions.append(("easy", {"easy": True}))
    missions.append(("shaped", {"shaped": True}))
    missions.append(("easy_shaped", {"easy": True, "shaped": True}))
    return {name: dict(values) for name, values in missions}


USER_MAP_CATALOG: tuple[UserMap, ...] = (
    SiteUserMap(
        name="training_facility_1",
        site="training_facility_open_1.map",
        mission_args=_with_easy_shaped_variants(extra=[("energy_intensive", {"base_mission": "energy_intensive"})]),
    ),
    SiteUserMap(
        name="training_facility_2",
        site="training_facility_open_2.map",
        mission_args=_with_easy_shaped_variants(),
    ),
    SiteUserMap(
        name="training_facility_3",
        site="training_facility_open_3.map",
        mission_args=_with_easy_shaped_variants(),
    ),
    SiteUserMap(
        name="training_facility_4",
        site="training_facility_tight_4.map",
        mission_args=_with_easy_shaped_variants(),
    ),
    SiteUserMap(
        name="training_facility_5",
        site="training_facility_tight_5.map",
        mission_args=_with_easy_shaped_variants(),
    ),
    SiteUserMap(
        name="training_facility_6",
        site="training_facility_clipped.map",
        mission_args=_with_easy_shaped_variants(),
    ),
    SiteUserMap(
        name="machina_1",
        site="cave_base_50.map",
        mission_args=_with_easy_shaped_variants(extra=[("clipped", {"map_builder_args": {"clip_rate": 0.02}})]),
    ),
    SiteUserMap(
        name="machina_2",
        site="machina_100_stations.map",
        mission_args=_with_easy_shaped_variants(),
    ),
    SiteUserMap(
        name="machina_3",
        site="machina_200_stations.map",
        mission_args=_with_easy_shaped_variants(),
    ),
    SiteUserMap(
        name="machina_1_big",
        site="canidate1_500_stations.map",
        mission_args=_with_easy_shaped_variants(),
    ),
    SiteUserMap(
        name="machina_2_bigger",
        site="canidate1_1000_stations.map",
        mission_args=_with_easy_shaped_variants(),
    ),
    SiteUserMap(
        name="machina_3_big",
        site="canidate2_500_stations.map",
        mission_args=_with_easy_shaped_variants(),
    ),
    SiteUserMap(
        name="machina_4_bigger",
        site="canidate2_1000_stations.map",
        mission_args=_with_easy_shaped_variants(),
    ),
    SiteUserMap(
        name="machina_5_big",
        site="canidate3_500_stations.map",
        mission_args=_with_easy_shaped_variants(),
    ),
    SiteUserMap(
        name="machina_6_bigger",
        site="canidate3_1000_stations.map",
        mission_args=_with_easy_shaped_variants(),
    ),
    SiteUserMap(
        name="machina_7_big",
        site="canidate4_500_stations.map",
        mission_args=_with_easy_shaped_variants(),
    ),
    SiteUserMap(
        name="vanilla",
        site="vanilla_small.map",
        mission_args={"default": {"easy": True}},
    ),
    SiteUserMap(
        name="vanilla_large",
        site="vanilla_large.map",
        mission_args={"default": {"easy": True}},
    ),
    RandomUserMap(
        name="random",
        mission_args=dict(
            default=dict(num_cogs=2),
            medium=dict(width=200, height=200, num_cogs=4),
            large=dict(width=500, height=500, num_cogs=10),
        ),
    ),
)
