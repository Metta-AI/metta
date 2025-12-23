# These evals are a spanning set of what might show up on the leaderboard.
# They are not exhaustive, but they should cover most situations.

from __future__ import annotations

import logging

from cogames.cogs_vs_clips.mission import Mission, Site
from cogames.cogs_vs_clips.procedural import MachinaArena
from cogames.cogs_vs_clips.sites import HELLO_WORLD, TRAINING_FACILITY
from cogames.cogs_vs_clips.variants import (
    ClipHubStationsVariant,
    ClipPeriodOnVariant,
    CompassVariant,
    CyclicalUnclipVariant,
    DarkSideVariant,
    DistantResourcesVariant,
    EmptyBaseVariant,
    EnergizedVariant,
    ExtractorHeartTuneVariant,
    InventoryHeartTuneVariant,
    LonelyHeartVariant,
    PackRatVariant,
    QuadrantBuildingsVariant,
    ResourceBottleneckVariant,
    RoughTerrainVariant,
    SingleResourceUniformVariant,
    SingleToolUnclipVariant,
    SingleUseSwarmVariant,
    SuperChargedVariant,
    VibeCheckMin2Variant,
)
from mettagrid.mapgen.mapgen import MapGen

logger = logging.getLogger(__name__)

SMALL_HELLO_WORLD = Site(
    name="small_hello_world",
    description="Small hello world map.",
    map_builder=MapGen.Config(width=50, height=50, instance=MachinaArena.Config(spawn_count=20)),
    min_cogs=1,
    max_cogs=20,
)

MEDIUM_HELLO_WORLD = Site(
    name="medium_hello_world",
    description="Medium hello world map.",
    map_builder=MapGen.Config(width=100, height=100, instance=MachinaArena.Config(spawn_count=20)),
    min_cogs=1,
    max_cogs=20,
)

LARGE_HELLO_WORLD = Site(
    name="large_hello_world",
    description="Large hello world map.",
    map_builder=MapGen.Config(width=500, height=500, instance=MachinaArena.Config(spawn_count=20)),
    min_cogs=1,
    max_cogs=20,
)

# Resource Bottleneck evals (Different resources are the limiting reagents; agents must prioritize correct resource.)
OxygenBottleneck = Mission(
    name="oxygen_bottleneck",
    description="Oxygen is the limiting resource; agents must prioritize oxygen over other resources.",
    site=HELLO_WORLD,
    variants=[
        EmptyBaseVariant(missing=["oxygen_extractor"]),
        ResourceBottleneckVariant(resource=["oxygen"]),
        SingleResourceUniformVariant(building_name="oxygen_extractor"),
        PackRatVariant(),
    ],
)

# Unclipping missions (agents must craft gear to unclip oxygen and proceed)
UnclippingEasy = Mission(
    name="unclipping_easy",
    description="World grows increasingly clipped over time; agents must craft gear to unclip and proceed.",
    site=HELLO_WORLD,
    variants=[
        ClipPeriodOnVariant(clip_period=50),
        PackRatVariant(),
        SingleToolUnclipVariant(),
        ClipHubStationsVariant(clip=["oxygen_extractor", "germanium_extractor", "silicon_extractor"]),
    ],
)

UnclippingStandard = Mission(
    name="unclipping_standard",
    description="Standard unclipping; periodic clips and multiple stations clipped at start.",
    site=HELLO_WORLD,
    variants=[
        ClipPeriodOnVariant(clip_period=25),
        ClipHubStationsVariant(clip=["oxygen_extractor", "germanium_extractor", "silicon_extractor"]),
        CyclicalUnclipVariant(),
    ],
)

UnclippingHard = Mission(
    name="unclipping_hard",
    description="Hard unclipping; oxygen starts clipped with tight extractor budgets.",
    site=HELLO_WORLD,
    variants=[
        ClipPeriodOnVariant(clip_period=10),
        PackRatVariant(),
        ClipHubStationsVariant(clip=["oxygen_extractor", "germanium_extractor", "silicon_extractor"]),
        CyclicalUnclipVariant(),
    ],
)

ClippedEasyHearts = Mission(
    name="clipped_easy_hearts",
    description="Easy hearts with clipping and tool making.",
    site=HELLO_WORLD,
    variants=[
        ClipPeriodOnVariant(),
        ClipHubStationsVariant(),
        InventoryHeartTuneVariant(hearts=1),
        LonelyHeartVariant(),
        PackRatVariant(),
    ],
)

# Energy Starved evals (Low energy regen and weak chargers; requires careful charging and routing.)
EnergyStarved = Mission(
    name="energy_starved",
    description="Energy is the limiting resource; agents must prioritize energy over other resources.",
    site=HELLO_WORLD,
    variants=[
        EmptyBaseVariant(),
        ResourceBottleneckVariant(resource=["energy"]),
        DarkSideVariant(),
    ],
)

# Curated difficulty tiers per mission
# ------------------------------------------------------------
# Oxygen Bottleneck
OxygenBottleneckEasy = Mission(
    name="oxygen_bottleneck_easy",
    description="Easy: tuned oxygen focus with simple layout and generous capacities.",
    site=HELLO_WORLD,
    variants=[
        SingleResourceUniformVariant(building_name="oxygen_extractor"),
        PackRatVariant(),
    ],
)

OxygenBottleneckStandard = Mission(
    name="oxygen_bottleneck_standard",
    description="Standard: oxygen is the bottleneck; extractor missing at base.",
    site=HELLO_WORLD,
    variants=[
        EmptyBaseVariant(missing=["oxygen_extractor"]),
        ResourceBottleneckVariant(resource=["oxygen"]),
    ],
)

OxygenBottleneckHard = Mission(
    name="oxygen_bottleneck_hard",
    description="Hard: oxygen bottleneck plus distant layout and rough terrain.",
    site=HELLO_WORLD,
    variants=[
        EmptyBaseVariant(missing=["oxygen_extractor"]),
        ResourceBottleneckVariant(resource=["oxygen"]),
        RoughTerrainVariant(),
    ],
)

# Energy Starved
EnergyStarvedEasy = Mission(
    name="energy_starved_easy",
    description="Easy: abundant energy regen and capacity.",
    site=HELLO_WORLD,
    variants=[
        SuperChargedVariant(),
        EnergizedVariant(),
    ],
)

EnergyStarvedStandard = Mission(
    name="energy_starved_standard",
    description="Standard: energy is the limiting resource with dark-side regen.",
    site=HELLO_WORLD,
    variants=[
        DarkSideVariant(),
    ],
)

EnergyStarvedHard = Mission(
    name="energy_starved_hard",
    description="Hard: energy bottleneck with solar flare damage and rough terrain.",
    site=HELLO_WORLD,
    variants=[
        ResourceBottleneckVariant(resource=["energy"]),
        DarkSideVariant(),
        RoughTerrainVariant(),
    ],
)

# Collect Distant Resources evals (Resources scattered far from base; heavy routing coordination.)

DistantResources = Mission(
    name="distant_resources",
    description="Resources scattered far from base; heavy routing coordination.",
    site=HELLO_WORLD,
    variants=[
        EmptyBaseVariant(),
        DistantResourcesVariant(),
    ],
)

# Distant Resources tiers
DistantResourcesEasy = Mission(
    name="distant_resources_easy",
    description="Easy: simplified distribution and navigation aids.",
    site=HELLO_WORLD,
    variants=[
        CompassVariant(),
        PackRatVariant(),
        DistantResourcesVariant(),
    ],
)

DistantResourcesStandard = Mission(
    name="distant_resources_standard",
    description="Standard: resources scattered far from base.",
    site=HELLO_WORLD,
    variants=[
        CompassVariant(),
        DistantResourcesVariant(),
    ],
)

DistantResourcesHard = Mission(
    name="distant_resources_hard",
    description="Hard: distant resources with rough terrain, single-use constraints, and dim lighting.",
    site=HELLO_WORLD,
    variants=[
        DistantResourcesVariant(),
        RoughTerrainVariant(),
        SingleUseSwarmVariant(),
    ],
)

# Divide and Conquer evals (Resources split by regions; specialize per resource and reconvene at base.)

QuadrantBuildings = Mission(
    name="quadrant_buildings",
    description="Place buildings in the four quadrants of the map.",
    site=HELLO_WORLD,
    variants=[
        EmptyBaseVariant(),
        QuadrantBuildingsVariant(),
    ],
)

# Quadrant Buildings tiers
QuadrantBuildingsEasy = Mission(
    name="quadrant_buildings_easy",
    description="Easy: single resource uniformly distributed with navigation aid and light inventory boost.",
    site=HELLO_WORLD,
    variants=[
        QuadrantBuildingsVariant(),
        CompassVariant(),
        PackRatVariant(),
    ],
)

QuadrantBuildingsStandard = Mission(
    name="quadrant_buildings_standard",
    description="Standard: buildings placed in quadrants.",
    site=HELLO_WORLD,
    variants=[
        QuadrantBuildingsVariant(),
        EmptyBaseVariant(),
    ],
)

QuadrantBuildingsHard = Mission(
    name="quadrant_buildings_hard",
    description="Hard: quadrant distribution with empty base, rough terrain and dim lighting.",
    site=HELLO_WORLD,
    variants=[
        QuadrantBuildingsVariant(),
        EmptyBaseVariant(),
        RoughTerrainVariant(),
    ],
)

# Single Use Swarm evals (Everything is single use, so agents must fan out and reconverge with needed resources.)

SingleUseSwarm = Mission(
    name="single_use_swarm",
    description="Everything is single use, so agents must fan out and reconverge with needed resources.",
    site=HELLO_WORLD,
    variants=[
        SingleUseSwarmVariant(),
    ],
)

# Single Use Swarm tiers
SingleUseSwarmEasy = Mission(
    name="single_use_swarm_easy",
    description="Easy: single-use but with small map, compass, and better energy.",
    site=HELLO_WORLD,
    variants=[
        SingleUseSwarmVariant(),
        CompassVariant(),
        SuperChargedVariant(),
        ExtractorHeartTuneVariant(hearts=1),
    ],
)

SingleUseSwarmStandard = Mission(
    name="single_use_swarm_standard",
    description="Standard: single-use with distant resources.",
    site=HELLO_WORLD,
    variants=[
        SingleUseSwarmVariant(),
        DistantResourcesVariant(),
    ],
)

SingleUseSwarmHard = Mission(
    name="single_use_swarm_hard",
    description="Hard: single-use with distance, rough terrain, and poor lighting.",
    site=HELLO_WORLD,
    variants=[
        SingleUseSwarmVariant(),
        DistantResourcesVariant(),
        RoughTerrainVariant(),
    ],
)

# Vibe Check evals (Agents must check their vibe, either binary or full, and then coordinate others for assembly.)

VibeCheck = Mission(
    name="vibe_check",
    description="Agents must check their vibe, either binary or full, and then coordinate others for assembly.",
    site=HELLO_WORLD,
    variants=[
        EmptyBaseVariant(),
        VibeCheckMin2Variant(),
    ],
)

VibeCheckEasy = Mission(
    name="vibe_check_easy",
    description="Easy: generous hearts with shaping rewards to guide coordination.",
    site=HELLO_WORLD,
    variants=[
        EmptyBaseVariant(),
        VibeCheckMin2Variant(),
        LonelyHeartVariant(),
    ],
)

VibeCheckStandard = Mission(
    name="vibe_check_standard",
    description="Standard: vanilla vibe requirements without extra constraints.",
    site=HELLO_WORLD,
    variants=[
        EmptyBaseVariant(),
        VibeCheckMin2Variant(min_vibes=3),
    ],
)

VibeCheckHard = Mission(
    name="vibe_check_hard",
    description="Hard: minimum heart vibes plus energy pressure.",
    site=HELLO_WORLD,
    variants=[
        EmptyBaseVariant(),
        VibeCheckMin2Variant(min_vibes=4),
    ],
)

EasyHeartsTraining = Mission(
    name="easy_hearts_training",
    description="Simplified heart crafting with generous caps and extractor base.",
    site=TRAINING_FACILITY,
    variants=[
        LonelyHeartVariant(),
        PackRatVariant(),
    ],
)

EasyHeartsSmallWorld = Mission(
    name="easy_small_hearts",
    description="Simplified heart crafting with generous caps and extractor base.",
    site=SMALL_HELLO_WORLD,
    variants=[
        LonelyHeartVariant(),
        PackRatVariant(),
    ],
)

EasyHeartsMediumWorld = Mission(
    name="easy_medium_hearts",
    description="Simplified heart crafting with generous caps and extractor base.",
    site=MEDIUM_HELLO_WORLD,
    variants=[
        LonelyHeartVariant(),
        PackRatVariant(),
    ],
)

EasyHeartsLargeWorld = Mission(
    name="easy_large_hearts",
    description="Simplified heart crafting with generous caps and extractor base.",
    site=LARGE_HELLO_WORLD,
    variants=[
        LonelyHeartVariant(),
        PackRatVariant(),
    ],
)

EVAL_MISSIONS: list[Mission] = [
    # Oxygen bottleneck tiers
    OxygenBottleneckEasy,
    OxygenBottleneckStandard,
    OxygenBottleneckHard,
    # Energy starved tiers
    EnergyStarvedEasy,
    EnergyStarvedStandard,
    EnergyStarvedHard,
    # Unclipping tiers
    UnclippingEasy,
    UnclippingStandard,
    UnclippingHard,
    # Distant resources tiers
    DistantResourcesEasy,
    DistantResourcesStandard,
    DistantResourcesHard,
    # Quadrant buildings tiers
    QuadrantBuildingsEasy,
    QuadrantBuildingsStandard,
    QuadrantBuildingsHard,
    # Single use swarm tiers
    SingleUseSwarmEasy,
    SingleUseSwarmStandard,
    SingleUseSwarmHard,
    # Vibe check tiers
    VibeCheckEasy,
    VibeCheckStandard,
    VibeCheckHard,
    # Hearts missions (easy only by design)
    EasyHeartsTraining,
    EasyHeartsSmallWorld,
    EasyHeartsMediumWorld,
    EasyHeartsLargeWorld,
]
