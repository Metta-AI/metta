from cogames.cogs_vs_clips.evals.diagnostic_evals import DIAGNOSTIC_EVALS
from cogames.cogs_vs_clips.evals.integrated_evals import EVAL_MISSIONS as INTEGRATED_EVAL_MISSIONS
from cogames.cogs_vs_clips.machina_missions_trainer import MACHINA_TRAINER_MISSIONS
from cogames.cogs_vs_clips.mission import Mission
from cogames.cogs_vs_clips.mission_utils import get_map
from cogames.cogs_vs_clips.sites import (
    EASY_MODE,
    FIXED_30,
    FIXED_50,
    FIXED_70,
    HELLO_WORLD,
    MACHINA_1,
    TRAINING_FACILITY,
)
from cogames.cogs_vs_clips.variants import (
    BalancedCornersVariant,
    ClipHubStationsVariant,
    ClipPeriodOnVariant,
    EmptyBaseVariant,
    ExtractorHeartTuneVariant,
    HeartChorusVariant,
    InventoryHeartTuneVariant,
    LonelyHeartVariant,
    PackRatVariant,
    VibeCheckMin2Variant,
)
from mettagrid.config.mettagrid_config import MettaGridConfig

# Training Facility Missions

HarvestMission = Mission(
    name="harvest",
    description="Collect resources, assemble hearts, and deposit them in the chest. Make sure to stay charged!",
    site=TRAINING_FACILITY,
    variants=[ExtractorHeartTuneVariant(hearts=10), PackRatVariant(), LonelyHeartVariant()],
)

VibeCheckMission = Mission(
    name="vibe_check",
    description="Modulate the group vibe to assemble HEARTs.",
    site=TRAINING_FACILITY,
    num_cogs=4,
    variants=[VibeCheckMin2Variant(), ExtractorHeartTuneVariant(hearts=10)],
)


RepairMission = Mission(
    name="repair",
    description="Repair disabled stations to restore their functionality.",
    site=TRAINING_FACILITY,
    num_cogs=2,
    variants=[
        InventoryHeartTuneVariant(hearts=1),
        ExtractorHeartTuneVariant(hearts=10),
        LonelyHeartVariant(),
        ClipPeriodOnVariant(),
        ClipHubStationsVariant(),
    ],
)  # If you get only two hearts you failed.


# Easy Hearts: simplified heart crafting and generous limits with extractor hub
EasyHeartsTrainingMission = Mission(
    name="easy_hearts_training_facility",
    description="Simplified heart crafting with generous caps and extractor base.",
    site=TRAINING_FACILITY,
    variants=[
        LonelyHeartVariant(),
        HeartChorusVariant(),
        PackRatVariant(),
    ],
)

# Easy Hearts: simplified heart crafting and generous limits with extractor hub
EasyHeartsHelloWorldMission = Mission(
    name="easy_hearts_hello_world",
    description="Simplified heart crafting with generous caps and extractor base.",
    site=HELLO_WORLD,
    variants=[
        LonelyHeartVariant(),
        HeartChorusVariant(),
        PackRatVariant(),
    ],
)


# Hello World Missions

HelloWorldOpenWorldMission = Mission(
    name="open_world",
    description="Collect resources and assemble HEARTs.",
    site=HELLO_WORLD,
    variants=[EmptyBaseVariant()],
)


# Machina 1 Missions
Machina1OpenWorldMission = Mission(
    name="open_world",
    description="Collect resources and assemble HEARTs.",
    site=MACHINA_1,
    variants=[EmptyBaseVariant()],
)

Machina1BalancedCornersMission = Mission(
    name="balanced_corners",
    description="Collect resources and assemble HEARTs. Map has balanced corner distances for fair spawns.",
    site=MACHINA_1,
    variants=[EmptyBaseVariant(), BalancedCornersVariant()],
)


HelloWorldUnclipMission = Mission(
    name="hello_world_unclip",
    description="Stabilize clipped extractors scattered across the hello_world sector.",
    site=HELLO_WORLD,
    num_cogs=4,
    variants=[ClipPeriodOnVariant(), InventoryHeartTuneVariant(hearts=1), ClipHubStationsVariant()],
)


# Easy Mode: Simplified training mission with extractor hub and generous variants
EasyMode = Mission(
    name="easy_mode",
    description="Easy training: extractor_hub_30 with lonely_heart, heart_chorus, and pack_rat variants.",
    site=EASY_MODE,
    variants=[
        LonelyHeartVariant(),
        HeartChorusVariant(),
        PackRatVariant(),
    ],
)

# Fixed extractor hub benchmarks (one mission per site)
ExtractorHub30 = Mission(
    name="extractor_hub",
    description="Fixed-map extractor hub benchmark (30x30).",
    site=FIXED_30,
    variants=[],
)

ExtractorHub50 = Mission(
    name="extractor_hub",
    description="Fixed-map extractor hub benchmark (50x50).",
    site=FIXED_50,
    variants=[],
)

ExtractorHub70 = Mission(
    name="extractor_hub",
    description="Fixed-map extractor hub benchmark (70x70).",
    site=FIXED_70,
    variants=[],
)


MISSIONS: list[Mission] = [
    ExtractorHub30,
    ExtractorHub50,
    ExtractorHub70,
    HarvestMission,
    VibeCheckMission,
    RepairMission,
    EasyHeartsTrainingMission,
    EasyHeartsHelloWorldMission,
    HelloWorldUnclipMission,
    HelloWorldOpenWorldMission,
    Machina1OpenWorldMission,
    Machina1BalancedCornersMission,
    EasyMode,
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
