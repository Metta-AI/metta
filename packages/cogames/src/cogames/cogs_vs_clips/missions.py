from pathlib import Path
from typing import Any, cast

from pydantic import Field

from cogames.cogs_vs_clips.mission import Mission, MissionVariant, Site
from cogames.cogs_vs_clips.procedural import (
    make_hub_only_map_builder,
    make_machina_procedural_map_builder,
)
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
from mettagrid.config.mettagrid_config import (
    AssemblerConfig,
    ChestConfig,
    GridObjectConfig,
    MettaGridConfig,
)
from mettagrid.map_builder.map_builder import MapBuilderConfig


def get_map(site: str) -> MapBuilderConfig:
    maps_dir = Path(__file__).parent.parent / "maps"
    map_path = maps_dir / site
    return MapBuilderConfig.from_uri(str(map_path))


PROCEDURAL_BASE_BUILDER = make_machina_procedural_map_builder(num_cogs=4)


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

        def modifier(cfg: MettaGridConfig) -> None:
            simplified_inputs = {"carbon": 1, "oxygen": 1, "germanium": 1, "silicon": 1, "energy": 1}

            assembler = cfg.game.objects.get("assembler")
            if assembler is None:
                return

            heart_recipe = ProtocolConfig(
                input_resources=dict(input_resources), output_resources={"heart": 1}, cooldown=1
            )

            non_heart_recipes = [
                (existing_vibe_tokens, recipe)
                for existing_vibe_tokens, recipe in assembler.recipes
                if recipe.output_resources.get("heart", 0) == 0
            ]

            assembler.recipes = [(["default"], heart_recipe), *non_heart_recipes]

        return _add_make_env_modifier(mission, modifier)


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


class PackRatVariant(MissionVariant):
    name: str = "pack_rat"
    description: str = "Raise heart, cargo, energy, and gear caps to 255."

    def apply(self, mission: Mission) -> Mission:
        mission.heart_capacity = max(mission.heart_capacity, 255)
        mission.energy_capacity = max(mission.energy_capacity, 255)
        mission.cargo_capacity = max(mission.cargo_capacity, 255)
        mission.gear_capacity = max(mission.gear_capacity, 255)
        return mission


class EnergizedVariant(MissionVariant):
    name: str = "energized"
    description: str = "Max energy and full regen so agents never run dry."

    def apply(self, mission: Mission) -> Mission:
        mission.energy_capacity = max(mission.energy_capacity, 255)
        mission.energy_regen_amount = mission.energy_capacity
        return mission


class NeutralFacedVariant(MissionVariant):
    name: str = "neutral_faced"
    description: str = "Disable glyph swapping; keep neutral face."

    def apply(self, mission: Mission) -> Mission:
        mission.enable_glyph_change = False
        mission.glyph_count = 1
        return mission


class HeartChorusVariant(MissionVariant):
    name: str = "heart_chorus"
    description: str = "Hearts require a chorus of 4 heart vibes and simple inputs."

    def apply(self, mission: Mission) -> Mission:
        # Default to 4 heart tokens; inputs can be adjusted if needed
        mission.heart_chorus_length = 4
        mission.heart_chorus_inputs = {"carbon": 1, "oxygen": 1, "germanium": 1, "silicon": 1, "energy": 1}
        return mission


# Biome variants (weather) for procedural maps
class DesertBiomeVariant(MissionVariant):
    name: str = "desert"
    description: str = "The desert sands make navigation challenging."

    def apply(self, mission: Mission) -> Mission:
        if hasattr(mission, "procedural_overrides"):
            mission.procedural_overrides["biome_weights"] = {"desert": 1.0, "caves": 0.0, "forest": 0.0, "city": 0.0}
            mission.procedural_overrides["base_biome"] = "desert"
        return mission


class ForestBiomeVariant(MissionVariant):
    name: str = "forest"
    description: str = "Dense forests obscure your view."

    def apply(self, mission: Mission) -> Mission:
        if hasattr(mission, "procedural_overrides"):
            mission.procedural_overrides["biome_weights"] = {"forest": 1.0, "caves": 0.0, "desert": 0.0, "city": 0.0}
            mission.procedural_overrides["base_biome"] = "forest"
        return mission


class CityBiomeVariant(MissionVariant):
    name: str = "city"
    description: str = "Ancient city ruins provide structured pathways."

    def apply(self, mission: Mission) -> Mission:
        if hasattr(mission, "procedural_overrides"):
            mission.procedural_overrides.update(
                {
                    "base_biome": "city",
                    "biome_weights": {"city": 1.0, "caves": 0.0, "desert": 0.0, "forest": 0.0},
                    # Fill almost the entire map with the city layer
                    "density_scale": 1.0,
                    "biome_count": 1,
                    "max_biome_zone_fraction": 0.95,
                    # Disable dungeon overlays so they don't overwrite the grid
                    "dungeon_weights": {"bsp": 0.0, "maze": 0.0, "radial": 0.0},
                    "max_dungeon_zone_fraction": 0.0,
                    # Tighten the city grid itself
                }
            )
        return mission


class CavesBiomeVariant(MissionVariant):
    name: str = "caves"
    description: str = "Winding cave systems create a natural maze."

    def apply(self, mission: Mission) -> Mission:
        if hasattr(mission, "procedural_overrides"):
            mission.procedural_overrides["biome_weights"] = {"caves": 1.0, "desert": 0.0, "forest": 0.0, "city": 0.0}
            mission.procedural_overrides["base_biome"] = "caves"
        return mission


class StoreBaseVariant(MissionVariant):
    name: str = "store_base"
    description: str = "Sanctum corners hold storage chests; cross remains clear."

    def apply(self, mission: Mission) -> Mission:
        mission.procedural_overrides["hub_corner_bundle"] = "chests"
        mission.procedural_overrides["hub_cross_bundle"] = "none"
        mission.procedural_overrides["hub_cross_distance"] = 7
        return mission


class ExtractorBaseVariant(MissionVariant):
    name: str = "extractor_base"
    description: str = "Sanctum corners host extractors; cross remains clear."

    def apply(self, mission: Mission) -> Mission:
        mission.procedural_overrides["hub_corner_bundle"] = "extractors"
        mission.procedural_overrides["hub_cross_bundle"] = "none"
        mission.procedural_overrides["hub_cross_distance"] = 7
        return mission


class BothBaseVariant(MissionVariant):
    name: str = "both_base"
    description: str = "Sanctum corners store chests and cross arms host extractors."

    def apply(self, mission: Mission) -> Mission:
        mission.procedural_overrides["hub_corner_bundle"] = "chests"
        mission.procedural_overrides["hub_cross_bundle"] = "extractors"
        mission.procedural_overrides["hub_cross_distance"] = 7
        return mission


VARIANTS = [
    MinedOutVariant,
    DarkSideVariant,
    BrightSideVariant,
    RoughTerrainVariant,
    SolarFlareVariant,
    DesertBiomeVariant,
    ForestBiomeVariant,
    CityBiomeVariant,
    CavesBiomeVariant,
    StoreBaseVariant,
    ExtractorBaseVariant,
    BothBaseVariant,
    LonelyHeartVariant,
    PackRatVariant,
    EnergizedVariant,
    NeutralFacedVariant,
    HeartChorusVariant,
]


# Define Sites
TRAINING_FACILITY = Site(
    name="training_facility",
    description="COG Training Facility. Basic training facility with open spaces and no obstacles.",
    map_builder=make_hub_only_map_builder(
        num_cogs=4,
        width=21,
        height=21,
        corner_bundle="chests",
        cross_bundle="extractors",
    ),
    min_cogs=1,
    max_cogs=4,
)

HELLO_WORLD = Site(
    name="hello_world",
    description="Welcome to space..",
    map_builder=make_machina_procedural_map_builder(
        num_cogs=4,
        width=100,
        height=100,
    ),
    min_cogs=1,
    max_cogs=20,
)

MACHINA_1 = Site(
    name="machina_1",
    description="Your first mission. Collect resources and assemble HEARTs.",
    # Originally was get_map("machina_200_stations.map"), but that was hard to make missions from
    map_builder=make_machina_procedural_map_builder(
        num_cogs=4,
        width=200,
        height=200,
    ),
    min_cogs=1,
    max_cogs=20,
)

MACHINA_PROCEDURAL = Site(
    name="machina_procedural",
    description="Procedurally generated asteroid arena with sanctum hub and resource pockets.",
    map_builder=PROCEDURAL_BASE_BUILDER,
    min_cogs=1,
    max_cogs=20,
)

SITES = [
    TRAINING_FACILITY,
    HELLO_WORLD,
    MACHINA_1,
    MACHINA_PROCEDURAL,
]


# TODO Make missions accept variants directly and allow them to select which variants are allowed to be applied
# Training Facility Missions
class HarvestMission(Mission):
    name: str = "harvest"
    description: str = "Collect resources and store them in the appropriate chests. Make sure to stay charged!"
    site: Site = TRAINING_FACILITY

    # Global Mission.instantiate now applies overrides; no per-mission override needed
    def make_env(self) -> MettaGridConfig:
        env = super().make_env()
        # Log-shaped chest rewards at episode end via per-step telescoping
        if self.num_cogs and self.num_cogs > 0:
            reward_weight = 1.0 / float(self.num_cogs)
        else:
            reward_weight = 1.0 / float(max(1, getattr(env.game, "num_agents", 1)))

        env.game.agent.rewards.inventory = {}
        env.game.agent.rewards.stats = {
            "chest.carbon.amount": reward_weight,
            "chest.oxygen.amount": reward_weight,
            "chest.germanium.amount": reward_weight,
            "chest.silicon.amount": reward_weight,
        }
        env.game.agent.rewards.inventory_max = {}
        env.game.agent.rewards.stats_max = {}
        # Ensure that the extractors are configured to have high max uses
        for name in ("germanium_extractor", "carbon_extractor", "oxygen_extractor", "silicon_extractor"):
            cfg = env.game.objects.get(name)
            if cfg is not None:
                cast(Any, cfg).max_uses = 100
        return env


class AssembleMission(Mission):
    name: str = "assemble"
    description: str = "Make HEARTs by using the assembler. Coordinate your team to maximize efficiency."
    site: Site = TRAINING_FACILITY

    # ONly extractors, no chests
    def configure(self):
        self.procedural_overrides = {"hub_corner_bundle": "none"}

    def make_env(self) -> MettaGridConfig:
        env = super().make_env()
        for name in ("germanium_extractor", "carbon_extractor", "oxygen_extractor", "silicon_extractor"):
            cfg = env.game.objects.get(name)
            if cfg is not None:
                cast(Any, cfg).max_uses = 100
        return env


class VibeCheckMission(Mission):
    name: str = "vibe_check"
    description: str = "Modulate the group vibe to assemble HEARTs and Gear."
    site: Site = TRAINING_FACILITY

    # Modify the assembler recipe so that it can only make HEARTs when
    # Set the number of cogs to 4

    def make_env(self) -> MettaGridConfig:
        env = super().make_env()
        # Require exactly 4 heart vibes for HEART crafting; keep gear recipes intact
        assembler_cfg = env.game.objects.get("assembler")
        if isinstance(assembler_cfg, AssemblerConfig):
            filtered: list[tuple[list[str], Any]] = []
            for vibes_list, recipe in assembler_cfg.recipes:
                if any(v == "heart" for v in vibes_list):
                    # Keep only the 4-heart recipe for heart crafting
                    if len(vibes_list) == 4 and all(v == "heart" for v in vibes_list):
                        filtered.append((vibes_list, recipe))
                else:
                    # Preserve non-heart (e.g., gear) recipes
                    filtered.append((vibes_list, recipe))
            assembler_cfg.recipes = filtered
        return env

    def instantiate(
        self,
        map_builder: MapBuilderConfig,
        num_cogs: int,
        variant: MissionVariant | None = None,
    ) -> "Mission":
        # Respect CLI --cogs if provided (differs from site.min_cogs); otherwise default to 4
        desired = 4 if (self.site and num_cogs == self.site.min_cogs) else num_cogs
        return super().instantiate(map_builder, desired, variant)


class RepairMission(Mission):
    name: str = "repair"
    description: str = "Repair disabled stations to restore their functionality."
    site: Site = TRAINING_FACILITY

    def configure(self):
        # Place chests in corners, extractors on cross; start extractors clipped
        self.procedural_overrides = {
            "hub_corner_bundle": "chests",
            "hub_cross_bundle": "extractors",
            "hub_cross_distance": 7,
        }
        self.carbon_extractor.start_clipped = True
        self.oxygen_extractor.start_clipped = True
        self.germanium_extractor.start_clipped = True
        self.silicon_extractor.start_clipped = True

    def make_env(self) -> MettaGridConfig:
        env = super().make_env()
        # Seed resource chests with one unit each to craft gear items
        for chest_name in ("chest_carbon", "chest_oxygen", "chest_germanium", "chest_silicon"):
            chest_cfg = env.game.objects.get(chest_name)
            if isinstance(chest_cfg, ChestConfig):
                chest_cfg.initial_inventory = 1
        return env

    def instantiate(
        self,
        map_builder: MapBuilderConfig,
        num_cogs: int,
        variant: MissionVariant | None = None,
    ) -> "Mission":
        # Respect CLI --cogs if provided (differs from site.min_cogs); otherwise default to 2
        desired = 2 if (self.site and num_cogs == self.site.min_cogs) else num_cogs
        return super().instantiate(map_builder, desired, variant)


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


# Base class for procedural missions
class ProceduralMissionBase(Mission):
    site: Site = MACHINA_PROCEDURAL
    procedural_overrides: dict[str, Any] = Field(default_factory=dict)

    def instantiate(
        self,
        map_builder: MapBuilderConfig,
        num_cogs: int,
        variant: MissionVariant | None = None,
    ) -> "Mission":
        # Use standard mission instantiation first (handles configure + variants)
        mission = super().instantiate(map_builder, num_cogs, variant)

        # Build procedural map using mission-specific overrides
        overrides = dict(mission.procedural_overrides)
        procedural_builder = make_machina_procedural_map_builder(num_cogs=num_cogs, **overrides)
        mission.map = procedural_builder

        return mission


# Procedural Missions
class MachinaProceduralExploreMission(ProceduralMissionBase):
    name: str = "explore"
    description: str = "There are HEARTs scattered around the map. Collect them all."

    def configure(self):
        # Mission defaults that don't depend on num_cogs
        self.heart_capacity = 99
        # Only chests for explore mission
        self.procedural_overrides = {
            "building_names": [
                "chest",
                "charger",
                "germanium_extractor",
                "silicon_extractor",
                "oxygen_extractor",
                "carbon_extractor",
            ],
            "building_weights": {
                "chest": 1.0,
                "charger": 0.6,
                "germanium_extractor": 0.6,
                "silicon_extractor": 0.3,
                "oxygen_extractor": 0.3,
                "carbon_extractor": 0.3,
            },  # this is relative weights to each building type
            "building_coverage": 0.01,  # this is density on the map
            "hub_corner_bundle": "chests",
            "hub_cross_bundle": "none",
            "hub_cross_distance": 7,
            # Distribution examples:
            # Use uniform distribution (default):
            # "distribution": {"type": "uniform"},
            # Use normal/Gaussian distribution (cluster around center of map at (0.5, 0.5),
            # smaller std means more concentrated around the mean):
            # "distribution": {
            #     "type": "normal",
            #     "mean_x": 0.5,
            #     "mean_y": 0.5,
            #     "std_x": 0.2,
            #     "std_y": 0.2,
            # },
            # Use exponential distribution, higher decay rate means objects drop off quickly,
            # spreads in a direction from a corner of map:
            # "distribution": {"type": "exponential", "decay_rate": 5.0, "origin_x": 0.0, "origin_y": 0.0},
            # Use poisson distribution (random clumping), always divides by 5, ex: if 10 chargers--> 2 clusters:
            # "distribution": {"type": "poisson"},
            # Use bimodal distribution (two clusters):
            # "distribution": {
            #     "type": "bimodal",
            #     "center1_x": 0.25,
            #     "center1_y": 0.25,
            #     "center2_x": 0.75,
            #     "center2_y": 0.75,
            #     "cluster_std": 0.15,
            # },
            # Per-building-type distributions:
            "building_distributions": {
                "chest": {"type": "exponential", "decay_rate": 5.0, "origin_x": 0.0, "origin_y": 0.0},
                "charger": {"type": "poisson"},
            },
        }

    def make_env(self) -> MettaGridConfig:
        env = super().make_env()
        # Reward agents for hearts they personally hold
        if self.num_cogs and self.num_cogs > 0:
            reward_weight = 1.0 / float(self.num_cogs)
        else:
            reward_weight = 1.0 / float(max(1, getattr(env.game, "num_agents", 1)))
        env.game.agent.rewards.inventory = {"heart": reward_weight}
        env.game.agent.rewards.stats = {}
        env.game.agent.rewards.inventory_max = {}
        env.game.agent.rewards.stats_max = {}

        # Ensure every chest template starts with one heart
        chest_cfg = env.game.objects.get("chest")
        if isinstance(chest_cfg, ChestConfig):
            chest_cfg.initial_inventory = 1
        return env


class ProceduralOpenWorldMission(ProceduralMissionBase):
    name: str = "open_world"
    description: str = "Collect resources and assemble HEARTs."

    def configure(self):
        self.procedural_overrides = {
            "building_distributions": {
                "chest": {"type": "exponential", "decay_rate": 5.0, "origin_x": 0.0, "origin_y": 0.0},
                "charger": {"type": "poisson"},
            }
        }


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
    MachinaProceduralExploreMission,
    ProceduralOpenWorldMission,
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
