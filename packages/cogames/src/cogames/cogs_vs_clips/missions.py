from typing import override

from cogames.cogs_vs_clips.evals.eval_missions import EVAL_MISSIONS
from cogames.cogs_vs_clips.mission import Mission, MissionVariant
from cogames.cogs_vs_clips.mission_utils import get_map
from cogames.cogs_vs_clips.procedural import (
    BaseHubVariant,
    MachinaArenaVariant,
)
from cogames.cogs_vs_clips.sites import HELLO_WORLD, MACHINA_1, TRAINING_FACILITY
from mettagrid.config.mettagrid_config import (
    AssemblerConfig,
    ChestConfig,
    MettaGridConfig,
    ProtocolConfig,
)
from mettagrid.mapgen.scenes.building_distributions import DistributionConfig, DistributionType

# Training Facility Missions


# Note: variants in this file are one-time only, used to produce their
# respective missions, and not registered in VARIANTS list.
# We hope to refactor these variants into more reusable classes in variants.py.
class HarvestMissionVariant(MissionVariant):
    name: str = "harvest_mission"

    @override
    def modify_env(self, mission, env):
        # Reset rewards to match pre-procedural behaviour; variants (e.g. Heart Chorus) will override as needed.
        env.game.agent.rewards.inventory = {}
        env.game.agent.rewards.stats = {}
        env.game.agent.rewards.inventory_max = {}
        env.game.agent.rewards.stats_max = {}

        # Ensure that the extractors are configured to have high max uses
        for name in ("germanium_extractor", "carbon_extractor", "oxygen_extractor", "silicon_extractor"):
            cfg = env.game.objects[name]
            if not isinstance(cfg, AssemblerConfig):
                raise TypeError(f"Expected '{name}' to be AssemblerConfig")
            cfg.max_uses = 100


HarvestMission = HarvestMissionVariant().as_mission(
    name="harvest",
    description="Collect resources, assemble hearts, and deposit them in the chest. Make sure to stay charged!",
    site=TRAINING_FACILITY,
)


class AssembleMissionVariant(MissionVariant):
    name: str = "assemble_mission"

    @override
    def modify_env(self, mission, env):
        # Only extractors, no chests
        node = BaseHubVariant.extract_node(env)
        node.corner_bundle = "none"

        for name in ("germanium_extractor", "carbon_extractor", "oxygen_extractor", "silicon_extractor"):
            cfg = env.game.objects[name]
            if not isinstance(cfg, AssemblerConfig):
                raise TypeError(f"Expected '{name}' to be AssemblerConfig")
            cfg.max_uses = 100


AssembleMission = AssembleMissionVariant().as_mission(
    name="assemble",
    description="Make HEARTs by using the assembler. Coordinate your team to maximize efficiency.",
    site=TRAINING_FACILITY,
)


class VibeCheckMissionVariant(MissionVariant):
    name: str = "vibe_check_mission"

    # Modify the assembler recipe so that it can only make HEARTs when
    # Set the number of cogs to 4
    @override
    def modify_mission(self, mission):
        mission.num_cogs = 4

    @override
    def modify_env(self, mission, env):
        # Require exactly 4 heart vibes for HEART crafting; keep gear recipes intact
        assembler_cfg = env.game.objects["assembler"]
        if not isinstance(assembler_cfg, AssemblerConfig):
            raise TypeError("Expected 'assembler' to be AssemblerConfig")
        filtered: list[ProtocolConfig] = []
        for protocol in assembler_cfg.protocols:
            if "heart" in protocol.vibes:
                # Keep only the 4-heart recipe for heart crafting
                if len(protocol.vibes) == 4 and all(v == "heart" for v in protocol.vibes):
                    filtered.append(protocol)
            else:
                # Preserve non-heart (e.g., gear) recipes
                filtered.append(protocol)
        assembler_cfg.protocols = filtered


VibeCheckMission = VibeCheckMissionVariant().as_mission(
    name="vibe_check",
    description="Modulate the group vibe to assemble HEARTs and Gear.",
    site=TRAINING_FACILITY,
)


class RepairMissionVariant(MissionVariant):
    name: str = "repair_mission"

    @override
    def modify_mission(self, mission):
        mission.num_cogs = 2

        mission.carbon_extractor.start_clipped = True
        mission.oxygen_extractor.start_clipped = True
        mission.germanium_extractor.start_clipped = True
        mission.silicon_extractor.start_clipped = True

    @override
    def modify_env(self, mission, env):
        # Place chests in corners, extractors on cross; start extractors clipped
        node = BaseHubVariant.extract_node(env)
        node.corner_bundle = "chests"
        node.cross_bundle = "extractors"
        node.cross_distance = 7

        # Seed resource chests with one unit each to craft gear items
        chest_cfg = env.game.objects["chest"]
        if not isinstance(chest_cfg, ChestConfig):
            raise TypeError("Expected 'chest' to be ChestConfig")
        chest_cfg.initial_inventory = {"carbon": 1, "oxygen": 1, "germanium": 1, "silicon": 1}


RepairMission = RepairMissionVariant().as_mission(
    name="repair",
    description="Repair disabled stations to restore their functionality.",
    site=TRAINING_FACILITY,
)


class UnclipDrillsMissionVariant(MissionVariant):
    name: str = "unclip_drills_mission"

    @override
    def modify_mission(self, mission):
        mission.clip_rate = 0.0

        for station in (
            mission.carbon_extractor,
            mission.oxygen_extractor,
            mission.germanium_extractor,
            mission.silicon_extractor,
            mission.charger,
        ):
            station.start_clipped = True

    @override
    def modify_env(self, mission, env):
        node = BaseHubVariant.extract_node(env)
        node.cross_bundle = "extractors"
        node.cross_distance = 7

        chest_cfg = env.game.objects["chest"]
        if not isinstance(chest_cfg, ChestConfig):
            raise TypeError("Expected 'chest' to be ChestConfig")
        for resource in ("carbon", "oxygen", "germanium", "silicon"):
            chest_cfg.initial_inventory[resource] = max(chest_cfg.initial_inventory.get(resource, 0), 3)

        agent_cfg = env.game.agent
        agent_cfg.initial_inventory = dict(agent_cfg.initial_inventory)
        for resource in ("decoder", "modulator", "scrambler", "resonator"):
            agent_cfg.initial_inventory[resource] = agent_cfg.initial_inventory.get(resource, 0) + 2


UnclipDrillsMission = UnclipDrillsMissionVariant().as_mission(
    name="unclip_drills",
    description="Practice unclipping hub facilities after a grid outage.",
    site=TRAINING_FACILITY,
)


SignsAndPortentsMission = Mission(
    name="signs_and_portents",
    description="Interpret the signs and portents to discover new assembler protocols.",
    site=TRAINING_FACILITY,
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


class UnclipFieldOpsMissionVariant(MachinaArenaVariant):
    name: str = "unclip_field_ops_mission"

    @override
    def modify_node(self, node):
        node.building_names = [
            "charger",
            "germanium_extractor",
            "silicon_extractor",
            "oxygen_extractor",
            "carbon_extractor",
        ]
        node.building_weights = {
            "charger": 0.6,
            "germanium_extractor": 0.6,
            "silicon_extractor": 0.5,
            "oxygen_extractor": 0.5,
            "carbon_extractor": 0.5,
        }
        node.building_coverage = 0.015
        node.hub.corner_bundle = "chests"
        node.hub.cross_bundle = "extractors"
        node.distribution = DistributionConfig(type=DistributionType.POISSON)

    @override
    def modify_mission(self, mission):
        # default to 4 cogs
        mission.num_cogs = 4
        mission.clip_rate = 0.02

        for station in (
            mission.carbon_extractor,
            mission.oxygen_extractor,
            mission.germanium_extractor,
            mission.silicon_extractor,
        ):
            station.start_clipped = True
        mission.charger.start_clipped = True

    @override
    def modify_env(self, mission: Mission, env: MettaGridConfig):
        super().modify_env(mission, env)

        chest_cfg = env.game.objects["chest"]
        if not isinstance(chest_cfg, ChestConfig):
            raise TypeError("Expected 'chest' to be ChestConfig")
        for resource in ("carbon", "oxygen", "germanium", "silicon"):
            chest_cfg.initial_inventory[resource] = max(chest_cfg.initial_inventory.get(resource, 0), 2)

        agent_cfg = env.game.agent
        agent_cfg.initial_inventory = dict(agent_cfg.initial_inventory)
        for resource in ("decoder", "modulator", "scrambler", "resonator"):
            agent_cfg.initial_inventory[resource] = agent_cfg.initial_inventory.get(resource, 0) + 1


HelloWorldUnclipMission = UnclipFieldOpsMissionVariant().as_mission(
    name="hello_world_unclip",
    description="Stabilize clipped extractors scattered across the hello_world sector.",
    site=HELLO_WORLD,
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
