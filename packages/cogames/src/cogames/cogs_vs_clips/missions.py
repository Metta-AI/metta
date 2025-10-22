from pathlib import Path

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
from mettagrid.config.mettagrid_config import GridObjectConfig, MettaGridConfig
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
        mission.assembler.recipe_pack = "tutorial"
        return mission


class PackRatVariant(MissionVariant):
    name: str = "pack_rat"
    description: str = "Boost heart inventory limits so agents can haul more at once."

    def apply(self, mission: Mission) -> Mission:
        mission.heart_capacity = max(mission.heart_capacity, 10)
        return mission


class BasicVibesVariant(MissionVariant):
    name: str = "basic_vibes"
    description: str = "Limit vibe swapping to the core tutorial set."

    def apply(self, mission: Mission) -> Mission:
        mission.change_glyph_enabled = True
        mission.glyph_limit = 20
        return mission


class HeartChorusVariant(MissionVariant):
    name: str = "heart_chorus"
    description: str = "Heart-centric reward shaping to guide cooperative play."

    def apply(self, mission: Mission) -> Mission:
        mission.reward_profile = "heart_focus"
        return mission


VARIANTS = [
    MinedOutVariant,
    DarkSideVariant,
    BrightSideVariant,
    RoughTerrainVariant,
    SolarFlareVariant,
    SimpleRecipesVariant,
    PackRatVariant,
    BasicVibesVariant,
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

SITES = [
    TRAINING_FACILITY,
    HELLO_WORLD,
    MACHINA_1,
]


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
]


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
