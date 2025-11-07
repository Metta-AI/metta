import cogames.cogs_vs_clips.evals.eval_missions
import cogames.cogs_vs_clips.mission
import cogames.cogs_vs_clips.mission_utils
import cogames.cogs_vs_clips.sites
import cogames.cogs_vs_clips.variants
import mettagrid.config.mettagrid_config

# Training Facility Missions


# Recreated with existing variant(s)
HarvestMission = cogames.cogs_vs_clips.mission.Mission(
    name="harvest",
    description="Collect resources, assemble hearts, and deposit them in the chest. Make sure to stay charged!",
    site=cogames.cogs_vs_clips.sites.TRAINING_FACILITY,
)


AssembleMission = cogames.cogs_vs_clips.mission.Mission(
    name="assemble",
    description="Make HEARTs by using the assembler. Coordinate your team to maximize efficiency.",
    site=cogames.cogs_vs_clips.sites.TRAINING_FACILITY,
    variants=[cogames.cogs_vs_clips.variants.InventoryHeartTuneVariant(hearts=5), cogames.cogs_vs_clips.variants.PackRatVariant()],
)


VibeCheckMission = cogames.cogs_vs_clips.mission.Mission(
    name="vibe_check",
    description="Modulate the group vibe to assemble HEARTs.",
    site=cogames.cogs_vs_clips.sites.TRAINING_FACILITY,
    num_cogs=4,
    variants=[cogames.cogs_vs_clips.variants.VibeCheckMin2Variant(), cogames.cogs_vs_clips.variants.InventoryHeartTuneVariant(hearts=1)],
)


RepairMission = cogames.cogs_vs_clips.mission.Mission(
    name="repair",
    description="Repair disabled stations to restore their functionality.",
    site=cogames.cogs_vs_clips.sites.TRAINING_FACILITY,
    num_cogs=2,
    variants=[cogames.cogs_vs_clips.variants.InventoryHeartTuneVariant(hearts=1), cogames.cogs_vs_clips.variants.ClipRateOnVariant()],
)


UnclipDrillsMission = cogames.cogs_vs_clips.mission.Mission(
    name="unclip_drills",
    description="Practice unclipping hub facilities after a grid outage.",
    site=cogames.cogs_vs_clips.sites.TRAINING_FACILITY,
    variants=[cogames.cogs_vs_clips.variants.ClipRateOnVariant(), cogames.cogs_vs_clips.variants.InventoryHeartTuneVariant(hearts=1)],
)


SignsAndPortentsMission = cogames.cogs_vs_clips.mission.Mission(
    name="signs_and_portents",
    description="Interpret the signs and portents to discover new assembler protocols.",
    site=cogames.cogs_vs_clips.sites.TRAINING_FACILITY,
)

# Easy Hearts: simplified heart crafting and generous limits with extractor hub
EasyHeartsMission = cogames.cogs_vs_clips.mission.Mission(
    name="easy_hearts",
    description="Simplified heart crafting, generous caps, extractor base, neutral vibe.",
    site=cogames.cogs_vs_clips.sites.TRAINING_FACILITY,
    variants=[
        cogames.cogs_vs_clips.variants.LonelyHeartVariant(),
        cogames.cogs_vs_clips.variants.HeartChorusVariant(),
        cogames.cogs_vs_clips.variants.PackRatVariant(),
        cogames.cogs_vs_clips.variants.NeutralFacedVariant(),
    ],
)


# Hello World Missions
ExploreMission = cogames.cogs_vs_clips.mission.Mission(
    name="explore",
    description="There are HEART chests scattered around the map. Put your HEARTs in them.",
    site=cogames.cogs_vs_clips.sites.HELLO_WORLD,
    variants=[cogames.cogs_vs_clips.variants.InventoryHeartTuneVariant(hearts=1, heart_capacity=10), cogames.cogs_vs_clips.variants.PackRatVariant()],
)


TreasureHuntMission = cogames.cogs_vs_clips.mission.Mission(
    name="treasure_hunt",
    description=(
        "The solar flare is making the germanium extractors really fiddly. "
        "A team of 4 is required to harvest germanium."
    ),
    site=cogames.cogs_vs_clips.sites.HELLO_WORLD,
    num_cogs=4,
    variants=[cogames.cogs_vs_clips.variants.ClipRateOnVariant()],
)


HelloWorldOpenWorldMission = cogames.cogs_vs_clips.mission.Mission(
    name="open_world",
    description="Collect resources and assemble HEARTs.",
    site=cogames.cogs_vs_clips.sites.HELLO_WORLD,
)


# Machina 1 Missions
Machina1OpenWorldMission = cogames.cogs_vs_clips.mission.Mission(
    name="open_world",
    description="Collect resources and assemble HEARTs.",
    site=cogames.cogs_vs_clips.sites.MACHINA_1,
)


HelloWorldUnclipMission = cogames.cogs_vs_clips.mission.Mission(
    name="hello_world_unclip",
    description="Stabilize clipped extractors scattered across the hello_world sector.",
    site=cogames.cogs_vs_clips.sites.HELLO_WORLD,
    num_cogs=4,
    variants=[cogames.cogs_vs_clips.variants.ClipRateOnVariant(), cogames.cogs_vs_clips.variants.ChestHeartTuneVariant(hearts=2)],
)

MISSIONS: list[cogames.cogs_vs_clips.mission.Mission] = [
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
    *cogames.cogs_vs_clips.evals.eval_missions.EVAL_MISSIONS,
]


def make_game(num_cogs: int = 2, map_name: str = "training_facility_open_1.map") -> mettagrid.config.mettagrid_config.MettaGridConfig:
    """Create a default cogs vs clips game configuration."""
    mission = HarvestMission.model_copy(deep=True)
    mission.num_cogs = num_cogs
    env = mission.make_env()
    env.game.map_builder = cogames.cogs_vs_clips.mission_utils.get_map(map_name)
    return env
