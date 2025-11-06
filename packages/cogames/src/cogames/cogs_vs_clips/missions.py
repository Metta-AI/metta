from cogames.cogs_vs_clips.evals.eval_missions import EVAL_MISSIONS
from cogames.cogs_vs_clips.mission import Mission
from cogames.cogs_vs_clips.mission_utils import get_map
from cogames.cogs_vs_clips.sites import HELLO_WORLD, MACHINA_1, TRAINING_FACILITY
from cogames.cogs_vs_clips.variants import (
    ChestsTwoHeartsVariant,
    ClipRateOnVariant,
    HeartChorusVariant,
    LonelyHeartVariant,
    NeutralFacedVariant,
    PackRatVariant,
    SeedOneHeartInputsVariant,
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
)


VibeCheckMission = Mission(
    name="vibe_check",
    description="Modulate the group vibe to assemble HEARTs and Gear.",
    site=TRAINING_FACILITY,
    num_cogs=4,
)


RepairMission = Mission(
    name="repair",
    description="Repair disabled stations to restore their functionality.",
    site=TRAINING_FACILITY,
    num_cogs=2,
).with_variants([ChestsTwoHeartsVariant(), ClipRateOnVariant()])


UnclipDrillsMission = Mission(
    name="unclip_drills",
    description="Practice unclipping hub facilities after a grid outage.",
    site=TRAINING_FACILITY,
).with_variants([ClipRateOnVariant(), SeedOneHeartInputsVariant(), ChestsTwoHeartsVariant()])


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
).with_variants(
    [
        LonelyHeartVariant(),
        HeartChorusVariant(),
        PackRatVariant(),
        NeutralFacedVariant(),
    ]
)


# Hello World Missions
ExploreMission = Mission(
    name="explore",
    description="There are HEARTs scattered around the map. Collect them all.",
    site=HELLO_WORLD,
)


TreasureHuntMission = Mission(
    name="treasure_hunt",
    description=(
        "The solar flare is making the germanium extractors really fiddly. "
        "A team of 4 is required to harvest germanium."
    ),
    site=HELLO_WORLD,
    num_cogs=4,
).with_variants([ClipRateOnVariant()])


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
).with_variants([ClipRateOnVariant(), ChestsTwoHeartsVariant()])

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
