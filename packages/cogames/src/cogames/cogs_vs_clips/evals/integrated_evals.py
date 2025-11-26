from __future__ import annotations

import logging
from typing import override

from cogames.cogs_vs_clips.mission import Mission, MissionVariant, NumCogsVariant, Site
from cogames.cogs_vs_clips.mission_utils import get_map
from cogames.cogs_vs_clips.procedural import MachinaArena
from cogames.cogs_vs_clips.sites import EVALS, HELLO_WORLD
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
    TinyHeartProtocolsVariant,
    VibeCheckMin2Variant,
)
from mettagrid.config.mettagrid_config import AssemblerConfig
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


class EvalVariant(MissionVariant):
    name: str = "eval_mission"

    map_name: str

    # Clipping
    clip_period: int = 0
    charger_eff: int = 120
    carbon_eff: int = 115
    oxygen_eff: int = 110
    germanium_eff: int = 80
    silicon_eff: int = 120

    # Max uses (0 means unlimited for charger; other stations respect values)
    max_uses_charger: int | None = None
    max_uses_carbon: int | None = None
    max_uses_oxygen: int | None = None
    max_uses_germanium: int | None = None
    max_uses_silicon: int | None = None
    # Agent energy regen per step
    energy_regen: int = 1
    inventory_regen_interval: int = 1

    @override
    def modify_mission(self, mission) -> None:
        # Apply pre-make_env efficiency and regen knobs
        mission.charger.efficiency = self.charger_eff
        mission.carbon_extractor.efficiency = self.carbon_eff
        mission.oxygen_extractor.efficiency = self.oxygen_eff
        mission.germanium_extractor.efficiency = self.germanium_eff
        mission.silicon_extractor.efficiency = self.silicon_eff
        mission.energy_regen_amount = self.energy_regen
        mission.inventory_regen_interval = self.inventory_regen_interval
        mission.clip_period = self.clip_period

    @override
    def modify_env(self, mission, env) -> None:
        env.game.map_builder = get_map(
            self.map_name,
            fixed_spawn_order=True,  # spawn positions are always fixed for evals
        )
        # Set episode length for all evals
        env.game.max_steps = 1000
        # Enable protocol observation for observation-based agents
        env.game.protocol_details_obs = True
        # Make HEART crafting feasible with a single agent using the heart glyph
        assembler_obj = env.game.objects.get("assembler")
        if isinstance(assembler_obj, AssemblerConfig):
            TinyHeartProtocolsVariant().modify_env(mission, env)

        if self.max_uses_charger is not None and "charger" in env.game.objects:
            env.game.objects["charger"].max_uses = self.max_uses_charger
        if self.max_uses_carbon is not None and "carbon_extractor" in env.game.objects:
            env.game.objects["carbon_extractor"].max_uses = self.max_uses_carbon
        if self.max_uses_oxygen is not None and "oxygen_extractor" in env.game.objects:
            env.game.objects["oxygen_extractor"].max_uses = self.max_uses_oxygen
        if self.max_uses_germanium is not None and "germanium_extractor" in env.game.objects:
            env.game.objects["germanium_extractor"].max_uses = self.max_uses_germanium
        if self.max_uses_silicon is not None and "silicon_extractor" in env.game.objects:
            env.game.objects["silicon_extractor"].max_uses = self.max_uses_silicon

        # Global quality-of-life tweaks for evals
        # 1) Double agent inventory caps for core resources and gear
        for limit in env.game.agent.resource_limits.values():
            for resource in (
                "carbon",
                "oxygen",
                "germanium",
                "silicon",
                "decoder",
                "modulator",
                "scrambler",
                "resonator",
                "energy",
            ):
                if resource in limit.resources:
                    limit.limit = limit.limit * 2

        # 2) Reduce depletion speed: double max_uses for extractors that are finite
        for obj_name in (
            "carbon_extractor",
            "oxygen_extractor",
            "germanium_extractor",
            "silicon_extractor",
        ):
            obj = env.game.objects.get(obj_name)
            if obj is not None and hasattr(obj, "max_uses"):
                try:
                    current = int(obj.max_uses)
                    if current > 0:
                        obj.max_uses = current * 2
                except Exception:
                    pass


GoTogether = Mission(
    name="go_together",
    description="Objects favor collective glyphing; travel and return as a pack.",
    site=EVALS,
    variants=[
        EvalVariant(
            map_name="evals/eval_balanced_spread.map",
            energy_regen=2,
            charger_eff=140,
            carbon_eff=130,
            oxygen_eff=125,
            germanium_eff=100,
            silicon_eff=135,
            max_uses_germanium=10,
            max_uses_silicon=20,
            max_uses_carbon=30,
            max_uses_oxygen=25,
        ),
        NumCogsVariant(num_cogs=4),
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
    GoTogether,
]
