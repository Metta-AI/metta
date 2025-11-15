from cogames.cogs_vs_clips.evals.diagnostic_evals import DIAGNOSTIC_EVALS
from cogames.cogs_vs_clips.evals.eval_missions import EVAL_MISSIONS
from cogames.cogs_vs_clips.evals.integrated_evals import EVAL_MISSIONS as INTEGRATED_EVAL_MISSIONS
from cogames.cogs_vs_clips.machina_missions_trainer import MACHINA_TRAINER_MISSIONS
from cogames.cogs_vs_clips.mission import Mission
from cogames.cogs_vs_clips.mission_utils import get_map
from cogames.cogs_vs_clips.sites import HELLO_WORLD, MACHINA_1, TRAINING_FACILITY
from cogames.cogs_vs_clips.variants import (
    ChestHeartTuneVariant,
    ClipHubStationsVariant,
    ClipPeriodOnVariant,
    ExtractorHeartTuneVariant,
    HeartChorusVariant,
    InventoryHeartTuneVariant,
    LonelyHeartVariant,
    NeutralFacedVariant,
    PackRatVariant,
    VibeCheckMin2Variant,
)
from mettagrid.config.mettagrid_config import MettaGridConfig

# Training Facility Missions

AssembleMission = Mission(
    name="assemble",
    description="Make HEARTs by using the assembler. Coordinate your team to maximize efficiency.",
    site=TRAINING_FACILITY,
    variants=[InventoryHeartTuneVariant(hearts=5), PackRatVariant()],
)

HarvestMission = Mission(
    name="harvest",
    description="Collect resources, assemble hearts, and deposit them in the chest. Make sure to stay charged!",
    site=TRAINING_FACILITY,
    variants=[ExtractorHeartTuneVariant(hearts=5), PackRatVariant()],
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
    variants=[InventoryHeartTuneVariant(hearts=1), ClipPeriodOnVariant(), ClipHubStationsVariant()],
)


# Easy Hearts: simplified heart crafting and generous limits with extractor hub
EasyHeartsTrainingMission = Mission(
    name="easy_hearts_training_facility",
    description="Simplified heart crafting, generous caps, extractor base, neutral vibe.",
    site=TRAINING_FACILITY,
    variants=[
        LonelyHeartVariant(),
        HeartChorusVariant(),
        PackRatVariant(),
        NeutralFacedVariant(),
    ],
)

# Easy Hearts: simplified heart crafting and generous limits with extractor hub
EasyHeartsHelloWorldMission = Mission(
    name="easy_hearts_hello_world",
    description="Simplified heart crafting, generous caps, extractor base, neutral vibe.",
    site=HELLO_WORLD,
    variants=[
        LonelyHeartVariant(),
        HeartChorusVariant(),
        PackRatVariant(),
        NeutralFacedVariant(),
    ],
)


# Hello World Missions

HelloWorldOpenWorldMission = Mission(
    name="open_world",
    description="Collect resources and assemble HEARTs.",
    site=HELLO_WORLD,
)

TreasureHuntMission = Mission(
    name="clipping",
    description=("Extractors are getting clipped, and we need to unclip them to continue."),
    site=HELLO_WORLD,
    num_cogs=4,
    variants=[ClipPeriodOnVariant()],
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
    variants=[ClipPeriodOnVariant(), ChestHeartTuneVariant(hearts=1), ClipHubStationsVariant()],
)


MISSIONS: list[Mission] = [
    HarvestMission,
    AssembleMission,
    VibeCheckMission,
    RepairMission,
    TreasureHuntMission,
    EasyHeartsTrainingMission,
    EasyHeartsHelloWorldMission,
    HelloWorldUnclipMission,
    HelloWorldOpenWorldMission,
    Machina1OpenWorldMission,
    *EVAL_MISSIONS,
    *INTEGRATED_EVAL_MISSIONS,
    *[mission_cls() for mission_cls in DIAGNOSTIC_EVALS],  # type: ignore[call-arg]
    *MACHINA_TRAINER_MISSIONS,
]


def make_game(num_cogs: int = 2, map_name: str = "training_facility_open_1.map") -> MettaGridConfig:
    """Create a default cogs vs clips game configuration."""
    mission = HarvestMission.model_copy(deep=True)
    mission.num_cogs = num_cogs
    env = mission.make_env()
    env.game.map_builder = get_map(map_name)
    return env
