from __future__ import annotations

import logging

from cogames.cogs_vs_clips.mission import Mission, Site
from cogames.cogs_vs_clips.procedural import MachinaArena
from cogames.cogs_vs_clips.sites import HELLO_WORLD
from cogames.cogs_vs_clips.variants import (
    DarkSideVariant,
    DistantResourcesVariant,
    EmptyBaseVariant,
    ExtractorHeartTuneVariant,
    HeartChorusVariant,
    LonelyHeartVariant,
    PackRatVariant,
    QuadrantBuildingsVariant,
    ResourceBottleneckVariant,
    SingleResourceUniformVariant,
    SingleUseSwarmVariant,
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
    map_builder=MapGen.Config(width=150, height=150, instance=MachinaArena.Config(spawn_count=20)),
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
        ExtractorHeartTuneVariant(hearts=10),
        ResourceBottleneckVariant(resource="oxygen"),
        SingleResourceUniformVariant(building_name="oxygen_extractor"),
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
        ResourceBottleneckVariant(resource="energy"),
        DarkSideVariant(),
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

# Single Use Swarm evals (Everything is single use, so agents must fan out and reconverge with needed resources.)

SingleUseSwarm = Mission(
    name="single_use_swarm",
    description="Everything is single use, so agents must fan out and reconverge with needed resources.",
    site=HELLO_WORLD,
    variants=[
        EmptyBaseVariant(),
        SingleUseSwarmVariant(),
        PackRatVariant(),
    ],
)

# Vibe Check evals (Agents must check their vibe, either binary or full, and then coordinate others for assembly.)

VibeCheck = Mission(
    name="vibe_check",
    description="Agents must check their vibe, either binary or full, and then coordinate others for assembly.",
    site=HELLO_WORLD,
    variants=[
        EmptyBaseVariant(),
        HeartChorusVariant(),
        VibeCheckMin2Variant(),
    ],
)

EasyHeartsMission = Mission(
    name="easy_hearts",
    description="Simplified heart crafting with generous caps and extractor base.",
    site=HELLO_WORLD,
    variants=[
        LonelyHeartVariant(),
        HeartChorusVariant(),
        PackRatVariant(),
    ],
)


EVAL_MISSIONS: list[Mission] = [
    OxygenBottleneck,
    EnergyStarved,
    DistantResources,
    QuadrantBuildings,
    SingleUseSwarm,
    VibeCheck,
    EasyHeartsMission,
]
