from __future__ import annotations

import logging
from typing import Dict

from cogames.cogs_vs_clips.mission import Mission
from cogames.cogs_vs_clips.sites import HELLO_WORLD, TRAINING_FACILITY
from cogames.cogs_vs_clips.variants import (
    DarkSideVariant,
    DistantResourcesVariant,
    EmptyBaseVariant,
    ExtractorHeartTuneVariant,
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
from mettagrid.config.mettagrid_config import MettaGridConfig

logger = logging.getLogger(__name__)

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

EasyHeartsMission = Mission(
    name="easy_hearts",
    description="Simplified heart crafting, generous caps, extractor base, neutral vibe.",
    site=TRAINING_FACILITY,
    variants=[
        LonelyHeartVariant(),
        HeartChorusVariant(),
        PackRatVariant(),
        NeutralFacedVariant(),
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


def make_integrated_eval_envs() -> Dict[str, MettaGridConfig]:
    """
    Gridworks config maker:
    Returns a mapping from integrated eval mission name to MettaGridConfig.
    """
    envs: Dict[str, MettaGridConfig] = {}
    for mission in EVAL_MISSIONS:
        try:
            envs[mission.name] = mission.make_env()
        except Exception:
            # Best-effort: skip missions that fail to build locally
            continue
    return envs
