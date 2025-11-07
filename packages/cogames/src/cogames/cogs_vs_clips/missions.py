from cogames.cogs_vs_clips.evals.eval_missions import EVAL_MISSIONS
from cogames.cogs_vs_clips.mission import Mission
from cogames.cogs_vs_clips.mission_utils import get_map
from cogames.cogs_vs_clips.sites import HELLO_WORLD, MACHINA_1, TRAINING_FACILITY
from cogames.cogs_vs_clips.variants import (
    ChestHeartTuneVariant,
    ClipRateOnVariant,
    HeartChorusVariant,
    InventoryHeartTuneVariant,
    LonelyHeartVariant,
    NeutralFacedVariant,
    PackRatVariant,
    VibeCheckMin2Variant,
)
from mettagrid.config.mettagrid_config import MettaGridConfig

# Training Facility Missions


# Recreated with existing variant(s)
HarvestMission = Mission(
    name="harvest",
    description="Collect resources, assemble hearts, and deposit them in the chest. Make sure to stay charged!",
    site=TRAINING_FACILITY,
)


AssembleMission = Mission(
    name="assemble",
    description="Make HEARTs by using the assembler. Coordinate your team to maximize efficiency.",
    site=TRAINING_FACILITY,
    variants=[InventoryHeartTuneVariant(hearts=5), PackRatVariant()],
)


VibeCheckMission = Mission(
    name="vibe_check",
    description="Modulate the group vibe to assemble HEARTs.",
    site=TRAINING_FACILITY,
    num_cogs=4,
    variants=[VibeCheckMin2Variant(), InventoryHeartTuneVariant(hearts=1)],
)


RepairMission = Mission(
    name="repair",
    description="Repair disabled stations to restore their functionality.",
    site=TRAINING_FACILITY,
    num_cogs=2,
    variants=[InventoryHeartTuneVariant(hearts=1), ClipRateOnVariant()],
)


UnclipDrillsMission = Mission(
    name="unclip_drills",
    description="Practice unclipping hub facilities after a grid outage.",
    site=TRAINING_FACILITY,
    variants=[ClipRateOnVariant(), InventoryHeartTuneVariant(hearts=1)],
)


SignsAndPortentsMission = Mission(
    name="signs_and_portents",
    description="Interpret the signs and portents to discover new assembler protocols.",
    site=TRAINING_FACILITY,
)

# Easy Hearts: simplified heart crafting and generous limits with extractor hub
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


# Hello World Missions
ExploreMission = Mission(
    name="explore",
    description="There are HEART chests scattered around the map. Put your HEARTs in them.",
    site=HELLO_WORLD,
    variants=[InventoryHeartTuneVariant(hearts=1, heart_capacity=10), PackRatVariant()],
)


TreasureHuntMission = Mission(
    name="treasure_hunt",
    description=(
        "The solar flare is making the germanium extractors really fiddly. "
        "A team of 4 is required to harvest germanium."
    ),
    site=HELLO_WORLD,
    num_cogs=4,
    variants=[ClipRateOnVariant()],
)


HelloWorldOpenWorldMission = Mission(
    name="open_world",
    description="Collect resources and assemble HEARTs.",
    site=HELLO_WORLD,
)


# Machina 1 Missions
Machina1OpenWorldMission = Mission(
    name="open_world",
    description="Collect resources and assemble HEARTs.",
    site=MACHINA_1,
)


HelloWorldUnclipMission = Mission(
    name="hello_world_unclip",
    description="Stabilize clipped extractors scattered across the hello_world sector.",
    site=HELLO_WORLD,
    num_cogs=4,
    variants=[ClipRateOnVariant(), ChestHeartTuneVariant(hearts=2)],
)

MISSIONS: list[Mission] = [
    HarvestMission,
    AssembleMission,
    VibeCheckMission,
    RepairMission,
    UnclipDrillsMission,
    SignsAndPortentsMission,
    ExploreMission,
    TreasureHuntMission,
    EasyHeartsMission,
    HelloWorldUnclipMission,
    HelloWorldOpenWorldMission,
    Machina1OpenWorldMission,
    *EVAL_MISSIONS,
]


def make_game(num_cogs: int = 2, map_name: str = "training_facility_open_1.map") -> MettaGridConfig:
    """Create a default cogs vs clips game configuration."""
    mission = HarvestMission.model_copy(deep=True)
    mission.num_cogs = num_cogs
    env = mission.make_env()
    env.game.map_builder = get_map(map_name)
    return env
