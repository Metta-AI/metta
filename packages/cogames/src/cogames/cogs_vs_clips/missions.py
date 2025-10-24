from pathlib import Path
from types import MethodType
from typing import Callable, List

from cogames.cogs_vs_clips.mission import Mission, MissionVariant, Site
from cogames.cogs_vs_clips.stations import (
    CarbonExtractorConfig,
    ChargerConfig,
    CvCAssemblerConfig,
    CvCChestConfig,
    CvCWallConfig,
    GermaniumExtractorConfig,
    OxygenExtractorConfig,
    SiliconExtractorConfig,
)
from mettagrid.config.mettagrid_config import GridObjectConfig, MettaGridConfig, ProtocolConfig
from mettagrid.map_builder.map_builder import MapBuilderConfig


def get_map(site: str) -> MapBuilderConfig:
    maps_dir = Path(__file__).parent.parent / "maps"
    map_path = maps_dir / site
    return MapBuilderConfig.from_uri(str(map_path))


class MinedOutVariant(MissionVariant):
    name: str = "mined_out"
    description: str = "Some resources are depleted. You must be efficient to survive."

    def apply(self, mission: Mission) -> Mission:
        mission.carbon_extractor.efficiency -= 50
        mission.oxygen_extractor.efficiency -= 50
        mission.germanium_extractor.efficiency -= 50
        mission.silicon_extractor.efficiency -= 50
        return mission


class DarkSideVariant(MissionVariant):
    name: str = "dark_side"
    description: str = "You're on the dark side of the asteroid. You recharge slower."

    def apply(self, mission: Mission) -> Mission:
        mission.energy_regen_amount = 0
        return mission


class LonelyHeartVariant(MissionVariant):
    name: str = "lonely_heart"
    description: str = "Making hearts for one agent is easy."

    def apply(self, mission: Mission) -> Mission:
        mission.assembler.heart_cost = 1
        return mission


class BrightSideVariant(MissionVariant):
    name: str = "super_charged"
    description: str = "The sun is shining on you. You recharge faster."

    def apply(self, mission: Mission) -> Mission:
        mission.energy_regen_amount += 2
        return mission


class RoughTerrainVariant(MissionVariant):
    name: str = "rough_terrain"
    description: str = "The terrain is rough. Moving is more energy intensive."

    def apply(self, mission: Mission) -> Mission:
        mission.move_energy_cost += 2
        return mission


class SolarFlareVariant(MissionVariant):
    name: str = "solar_flare"
    description: str = "Chargers have been damaged by the solar flare."

    def apply(self, mission: Mission) -> Mission:
        mission.charger.efficiency -= 50
        return mission


class SimpleRecipesVariant(MissionVariant):
    name: str = "simple_recipes"
    description: str = "Swap in tutorial assembler protocols for easier heart crafting."

    def apply(self, mission: Mission) -> Mission:
        def modifier(cfg: MettaGridConfig) -> None:
            assembler = cfg.game.objects.get("assembler")
            if assembler is None:
                return

            energy_recipe = (
                ["default"],
                ProtocolConfig(
                    input_resources={"energy": 1},
                    output_resources={"heart": 1},
                    cooldown=1,
                ),
            )

            if not any(
                recipe.input_resources == energy_recipe[1].input_resources
                and recipe.output_resources == energy_recipe[1].output_resources
                for _, recipe in assembler.recipes
            ):
                assembler.recipes = [energy_recipe, *assembler.recipes]

        return _add_make_env_modifier(mission, modifier)


class PackRatVariant(MissionVariant):
    name: str = "pack_rat"
    description: str = "Boost heart inventory limits so agents can haul more at once."

    def apply(self, mission: Mission) -> Mission:
        mission.heart_capacity = max(mission.heart_capacity, 10)
        return mission


class NeutralFacedVariant(MissionVariant):
    name: str = "neutral_faced"
    description: str = "Keep the neutral face glyph; disable glyph swapping entirely."

    def apply(self, mission: Mission) -> Mission:
        def modifier(cfg: MettaGridConfig) -> None:
            change_glyph = cfg.game.actions.change_glyph
            change_glyph.enabled = False
            change_glyph.number_of_glyphs = 1

        return _add_make_env_modifier(mission, modifier)


# Backwards-compatible alias
class HeartChorusVariant(MissionVariant):
    name: str = "heart_chorus"
    description: str = "Heart-centric reward shaping with gentle resource bonuses."

    def apply(self, mission: Mission) -> Mission:
        def modifier(cfg: MettaGridConfig) -> None:
            cfg.game.agent.rewards.stats = {
                "heart.gained": 0.25,
                "chest.heart.deposited": 1.0,
                "carbon.gained": 0.02,
                "oxygen.gained": 0.02,
                "germanium.gained": 0.05,
                "silicon.gained": 0.02,
                "energy.gained": 0.005,
            }

        return _add_make_env_modifier(mission, modifier)


VARIANTS = [
    MinedOutVariant,
    DarkSideVariant,
    BrightSideVariant,
    RoughTerrainVariant,
    SolarFlareVariant,
    LonelyHeartVariant,
    SimpleRecipesVariant,
    PackRatVariant,
    NeutralFacedVariant,
    HeartChorusVariant,
]


# Define Sites
TRAINING_FACILITY = Site(
    name="training_facility",
    description="COG Training Facility. Basic training facility with open spaces and no obstacles.",
    map_builder=get_map("training_facility_open_1.map"),
    min_cogs=1,
    max_cogs=4,
)

HELLO_WORLD = Site(
    name="hello_world",
    description="Welcome to space..",
    map_builder=get_map("machina_100_stations.map"),
    min_cogs=1,
    max_cogs=20,
)

MACHINA_1 = Site(
    name="machina_1",
    description="Your first mission. Collect resources and assemble HEARTs.",
    map_builder=get_map("machina_200_stations.map"),
    min_cogs=1,
    max_cogs=20,
)


MACHINA_1_SMALL = Site(
    name="machina_1_small",
    description="Your first mission. Collect resources and assemble HEARTs.",
    map_builder=get_map("machina_200_stations_small.map"),
    min_cogs=1,
    max_cogs=20,
)

SITES = [
    TRAINING_FACILITY,
    HELLO_WORLD,
    MACHINA_1,
    MACHINA_1_SMALL,
]

# Sites will be updated after exploration experiments import
# This happens after the exploration_experiments import below


# Training Facility Missions
class HarvestMission(Mission):
    name: str = "harvest"
    description: str = "Collect resources and store them in the communal chest. Make sure to stay charged!"
    site: Site = TRAINING_FACILITY


class AssembleMission(Mission):
    name: str = "assemble"
    description: str = "Make HEARTs by using the assembler. Coordinate your team to maximize efficiency."
    site: Site = TRAINING_FACILITY


class VibeCheckMission(Mission):
    name: str = "vibe_check"
    description: str = "Modulate the group vibe to assemble HEARTs and Gear."
    site: Site = TRAINING_FACILITY


class RepairMission(Mission):
    name: str = "repair"
    description: str = "Repair disabled stations to restore their functionality."
    site: Site = TRAINING_FACILITY


class SignsAndPortentsMission(Mission):
    name: str = "signs_and_portents"
    description: str = "Interpret the signs and portents to discover new assembler protocols."
    site: Site = TRAINING_FACILITY


# Hello World Missions
class ExploreMission(Mission):
    name: str = "explore"
    description: str = "There are HEARTs scattered around the map. Collect them all."
    site: Site = HELLO_WORLD


class TreasureHuntMission(Mission):
    name: str = "treasure_hunt"
    description: str = (
        "The solar flare is making the germanium extractors really fiddly. "
        "A team of 4 is required to harvest germanium."
    )
    site: Site = HELLO_WORLD


class HelloWorldOpenWorldMission(Mission):
    name: str = "open_world"
    description: str = "Collect resources and assemble HEARTs."
    site: Site = HELLO_WORLD


# Machina 1 Missions
class Machina1OpenWorldMission(Mission):
    name: str = "open_world"
    description: str = "Collect resources and assemble HEARTs."
    site: Site = MACHINA_1


class Machina1SmallOpenWorldMission(Mission):
    name: str = "open_world"
    description: str = "Collect resources and assemble HEARTs."
    site: Site = MACHINA_1_SMALL


# Import exploration experiments
try:
    from cogames.cogs_vs_clips.exploration_experiments import (
        EXPLORATION_MISSIONS,
        EXPLORATION_SITES,
    )

    _exploration_available = True
except ImportError:
    EXPLORATION_MISSIONS = []
    EXPLORATION_SITES = []
    _exploration_available = False


MISSIONS = [
    HarvestMission,
    AssembleMission,
    VibeCheckMission,
    RepairMission,
    SignsAndPortentsMission,
    ExploreMission,
    TreasureHuntMission,
    HelloWorldOpenWorldMission,
    Machina1OpenWorldMission,
    Machina1SmallOpenWorldMission,
    *EXPLORATION_MISSIONS,  # Add exploration experiments
]

# Update SITES with exploration sites
if _exploration_available:
    SITES.extend(EXPLORATION_SITES)


def _get_default_map_objects() -> dict[str, GridObjectConfig]:
    """Get default map objects for cogs vs clips missions."""
    carbon_extractor = CarbonExtractorConfig()
    oxygen_extractor = OxygenExtractorConfig()
    germanium_extractor = GermaniumExtractorConfig()
    silicon_extractor = SiliconExtractorConfig()
    charger = ChargerConfig()
    chest = CvCChestConfig()
    wall = CvCWallConfig()
    assembler = CvCAssemblerConfig()

    return {
        "carbon_extractor": carbon_extractor.station_cfg(),
        "oxygen_extractor": oxygen_extractor.station_cfg(),
        "germanium_extractor": germanium_extractor.station_cfg(),
        "silicon_extractor": silicon_extractor.station_cfg(),
        "charger": charger.station_cfg(),
        "chest": chest.station_cfg(),
        "wall": wall.station_cfg(),
        "assembler": assembler.station_cfg(),
    }


def make_game(num_cogs: int = 2, map_name: str = "training_facility_open_1.map") -> MettaGridConfig:
    """Create a default cogs vs clips game configuration."""
    mission = HarvestMission()
    map_builder = get_map(map_name)
    # Use no variant (default)
    variant = MissionVariant(name="default", description="Default mission variant")
    return mission.instantiate(map_builder, num_cogs, variant).make_env()


def _add_make_env_modifier(mission: Mission, modifier: Callable[[MettaGridConfig], None]) -> Mission:
    modifiers: List[Callable[[MettaGridConfig], None]] = getattr(mission, "__env_modifiers__", None)

    if modifiers is None:
        original_make_env = mission.make_env.__func__
        original_instantiate = mission.instantiate.__func__

        def wrapped_make_env(self, *args, **kwargs):
            cfg = original_make_env(self, *args, **kwargs)
            for fn in getattr(self, "__env_modifiers__", []):
                fn(cfg)
            return cfg

        def wrapped_instantiate(self, *args, **kwargs):
            instantiated = original_instantiate(self, *args, **kwargs)
            parent_mods = getattr(self, "__env_modifiers__", [])
            if parent_mods:
                object.__setattr__(instantiated, "__env_modifiers__", list(parent_mods))
                object.__setattr__(instantiated, "make_env", MethodType(wrapped_make_env, instantiated))
                object.__setattr__(instantiated, "instantiate", MethodType(wrapped_instantiate, instantiated))
            return instantiated

        object.__setattr__(mission, "__env_modifiers__", [])
        object.__setattr__(mission, "make_env", MethodType(wrapped_make_env, mission))
        object.__setattr__(mission, "instantiate", MethodType(wrapped_instantiate, mission))
        modifiers = mission.__env_modifiers__

    modifiers.append(modifier)
    return mission
