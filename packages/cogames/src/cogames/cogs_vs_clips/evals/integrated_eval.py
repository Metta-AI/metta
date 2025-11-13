from __future__ import annotations

from cogames.cogs_vs_clips.mission import Mission
from cogames.cogs_vs_clips.sites import HELLO_WORLD, TRAINING_FACILITY
from cogames.cogs_vs_clips.variants import (
    CompassVariant,
    DarkSideVariant,
    DistantResourcesVariant,
    EmptyBaseVariant,
    HeartChorusVariant,
    LonelyHeartVariant,
    NeutralFacedVariant,
    PackRatVariant,
    QuadrantBuildingsVariant,
    ResourceBottleneckVariant,
    SingleResourceUniformVariant,
    SingleUseSwarmVariant,
    VibeCheckMin2Variant,
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
        NeutralFacedVariant(),
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
        DarkSideVariant(),
        NeutralFacedVariant(),
        PackRatVariant(),
    ],
)

# Collect Distant Resources evals (Resources scattered far from base; heavy routing coordination.)

DistantResources = Mission(
    name="distant_resources",
    description="Resources scattered far from base; heavy routing coordination.",
    site=HELLO_WORLD,
    variants=[
        EmptyBaseVariant(),
        CompassVariant(),
        DistantResourcesVariant(),
        NeutralFacedVariant(),
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
        CompassVariant(),
        NeutralFacedVariant(),
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
        CompassVariant(),
        PackRatVariant(),
        NeutralFacedVariant(),
    ],
)

# Vibe Check evals (Agents must check their vibe, either binary or full, and then coordinate others for assembly.)

VibeCheck = Mission(
    name="vibe_check",
    description="Agents must check their vibe, either binary or full, and then coordinate others for assembly.",
    site=HELLO_WORLD,
    variants=[
        HeartChorusVariant(),
        VibeCheckMin2Variant(),
    ],
)

EasyHeartsTraining = Mission(
    name="easy_hearts_training",
    description="Simplified heart crafting, generous caps, extractor base, neutral vibe.",
    site=TRAINING_FACILITY,
    variants=[
        LonelyHeartVariant(),
        PackRatVariant(),
        NeutralFacedVariant(),
    ],
)

EasyHeartsMission = Mission(
    name="easy_hearts_hello_world",
    description="Simplified heart crafting, generous caps, extractor base, neutral vibe.",
    site=HELLO_WORLD,
    variants=[
        LonelyHeartVariant(),
        PackRatVariant(),
        NeutralFacedVariant(),
        EmptyBaseVariant(),
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
