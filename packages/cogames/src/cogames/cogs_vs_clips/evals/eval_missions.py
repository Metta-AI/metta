from __future__ import annotations

import logging
from typing import override

from cogames.cogs_vs_clips.mission import Mission, MissionVariant, NumCogsVariant
from cogames.cogs_vs_clips.mission_utils import get_map
from cogames.cogs_vs_clips.sites import EVALS
from mettagrid.config.mettagrid_config import ProtocolConfig

logger = logging.getLogger(__name__)


class EvalVariant(MissionVariant):
    name: str = "eval_mission"

    map_name: str

    # Clipping
    # Note: For clipping configuration, use difficulty variants (e.g., CLIPPED_OXYGEN)
    # instead of setting clip_rate directly here
    clip_rate: float = 0.0
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
        mission.clip_rate = self.clip_rate

    @override
    def modify_env(self, mission, env) -> None:
        env.game.map_builder = get_map(
            self.map_name,
            fixed_spawn_order=True,  # spawn positions are always fixed for evals
        )
        # Set episode length for all evals
        env.game.max_steps = 1000
        # Make HEART crafting feasible with a single agent using the heart glyph
        assembler_obj = env.game.objects.get("assembler")
        if assembler_obj is not None and hasattr(assembler_obj, "first_heart_cost"):
            # Set small single-agent recipe and prepend explicit heart/red-heart entries
            if hasattr(assembler_obj, "recipes"):
                tiny = ProtocolConfig(
                    input_resources={
                        "carbon": 2,
                        "oxygen": 2,
                        "germanium": 1,
                        "silicon": 3,
                        "energy": 2,
                    },
                    output_resources={"heart": 1},
                )
                heart_recipes = [(["heart"] * (i + 1), tiny) for i in range(4)]
                redheart_recipes = [(["red-heart"] * (i + 1), tiny) for i in range(4)]
                assembler_obj.recipes = [*heart_recipes, *redheart_recipes, *assembler_obj.recipes]

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
        try:
            limits = env.game.agent.resource_limits
            for key in ("carbon", "oxygen", "germanium", "silicon"):
                if key in limits and isinstance(limits[key], int):
                    limits[key] = limits[key] * 2
            for key in ("decoder", "modulator", "scrambler", "resonator"):
                if key in limits and isinstance(limits[key], int):
                    limits[key] = limits[key] * 2
            if "energy" in limits and isinstance(limits["energy"], int):
                limits["energy"] = limits["energy"] * 2
        except Exception:
            pass

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


OxygenBottleneck = Mission(
    name="oxygen_bottleneck",
    description="Oxygen paces assembly; batch other resources.",
    site=EVALS,
    variants=[
        EvalVariant(
            map_name="evals/eval_oxygen_bottleneck.map",
            charger_eff=130,
            oxygen_eff=60,
            energy_regen=2,
            max_uses_charger=0,
            max_uses_carbon=120,
            max_uses_oxygen=30,
            max_uses_germanium=15,
            max_uses_silicon=120,
        )
    ],
)


EnergyStarved = Mission(
    name="energy_starved",
    description="Low regen; requires careful charging and routing.",
    site=EVALS,
    variants=[
        EvalVariant(
            map_name="evals/eval_energy_starved.map",
            charger_eff=90,
            carbon_eff=125,
            oxygen_eff=115,
            germanium_eff=100,
            silicon_eff=125,
            energy_regen=1,
            inventory_regen_interval=2,
            max_uses_charger=0,
        )
    ],
)


# -----------------------------
# Extractor Hub Missions
# -----------------------------


ExtractorHub30 = Mission(
    name="extractor_hub_30",
    description="Small 30x30 extractor hub.",
    site=EVALS,
    variants=[
        EvalVariant(
            map_name="evals/extractor_hub_30x30.map",
            energy_regen=2,
            charger_eff=125,
            germanium_eff=90,
            max_uses_germanium=0,
        )
    ],
)


ExtractorHub50 = Mission(
    name="extractor_hub_50",
    description="Medium 50x50 extractor hub.",
    site=EVALS,
    variants=[
        EvalVariant(
            map_name="evals/extractor_hub_50x50.map",
            energy_regen=2,
            charger_eff=125,
            germanium_eff=90,
            max_uses_germanium=0,
        )
    ],
)


ExtractorHub70 = Mission(
    name="extractor_hub_70",
    description="Large 70x70 extractor hub.",
    site=EVALS,
    variants=[
        EvalVariant(
            map_name="evals/extractor_hub_70x70.map",
            energy_regen=2,
            charger_eff=130,
            germanium_eff=95,
            max_uses_germanium=0,
        )
    ],
)


ExtractorHub80 = Mission(
    name="extractor_hub_80",
    description="Large 80x80 extractor hub.",
    site=EVALS,
    variants=[
        EvalVariant(
            map_name="evals/extractor_hub_80x80.map",
            energy_regen=2,
            charger_eff=135,
            germanium_eff=95,
            max_uses_germanium=0,
        )
    ],
)


ExtractorHub100 = Mission(
    name="extractor_hub_100",
    description="Extra large 100x100 extractor hub.",
    site=EVALS,
    variants=[
        EvalVariant(
            map_name="evals/extractor_hub_100x100.map",
            energy_regen=2,
            charger_eff=140,
            germanium_eff=100,
            max_uses_germanium=0,
        )
    ],
)


# -----------------------------m
# Multi-agent Coordination Missions
# -----------------------------


CollectResourcesClassic = Mission(
    name="collect_resources_classic",
    description="Collect resources on the classic layout; balanced routing near base.",
    site=EVALS,
    variants=[
        EvalVariant(
            map_name="evals/eval_collect_resources.map",
            energy_regen=2,
            charger_eff=130,
            carbon_eff=125,
            oxygen_eff=115,
            germanium_eff=90,
            silicon_eff=125,
            max_uses_germanium=0,
            max_uses_silicon=0,
            max_uses_carbon=0,
            max_uses_oxygen=0,
        )
    ],
)


CollectResourcesSpread = Mission(
    name="collect_resources_spread",
    description="Collect resources (scattered nearby), rally and chorus glyph at assembler.",
    site=EVALS,
    variants=[
        EvalVariant(
            map_name="evals/eval_collect_resources_medium.map",
            energy_regen=2,
            charger_eff=135,
            carbon_eff=130,
            oxygen_eff=120,
            germanium_eff=95,
            silicon_eff=130,
            max_uses_germanium=0,
            max_uses_silicon=0,
            max_uses_carbon=0,
            max_uses_oxygen=0,
        )
    ],
)


CollectFar = Mission(
    name="collect_far",
    description="Collect resources scattered far; coordinate routes, chorus glyph, single carrier deposits.",
    site=EVALS,
    variants=[
        EvalVariant(
            map_name="evals/eval_collect_resources_hard.map",
            energy_regen=2,
            charger_eff=135,
            carbon_eff=130,
            oxygen_eff=120,
            germanium_eff=100,
            silicon_eff=135,
            max_uses_germanium=20,
            max_uses_silicon=25,
            max_uses_carbon=40,
            max_uses_oxygen=30,
        )
    ],
)


DivideAndConquer = Mission(
    name="divide_and_conquer",
    description="Resources split by regions; specialize per resource and reconvene at base.",
    site=EVALS,
    variants=[
        EvalVariant(
            map_name="evals/eval_divide_and_conquer.map",
            energy_regen=2,
            charger_eff=130,
            carbon_eff=125,
            oxygen_eff=120,
            germanium_eff=95,
            silicon_eff=130,
            max_uses_germanium=10,
            max_uses_silicon=15,
            max_uses_carbon=25,
            max_uses_oxygen=20,
        )
    ],
)


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
        NumCogsVariant(num_cogs=2),
    ],
)

SingleUseSwarm = Mission(
    name="single_use_swarm",
    description="Multi-agent variant of SingleUseWorld; stations max_uses=1, team must fan out and reconverge.",
    site=EVALS,
    variants=[
        EvalVariant(
            map_name="evals/eval_single_use_world.map",
            energy_regen=2,
            charger_eff=140,
            carbon_eff=130,
            oxygen_eff=125,
            germanium_eff=105,
            silicon_eff=135,
            max_uses_charger=0,
            max_uses_carbon=1,
            max_uses_oxygen=1,
            max_uses_germanium=1,
            max_uses_silicon=1,
        ),
        NumCogsVariant(num_cogs=2),
    ],
)

EVAL_MISSIONS: list[Mission] = [
    EnergyStarved,
    OxygenBottleneck,
    ExtractorHub30,
    ExtractorHub50,
    ExtractorHub70,
    ExtractorHub80,
    ExtractorHub100,
    CollectResourcesClassic,
    CollectResourcesSpread,
    CollectFar,
    DivideAndConquer,
    GoTogether,
    SingleUseSwarm,
]


# -----------------------------
# Successful Missions
# -----------------------------
# Missions where scripted agents perform well (>50% success rate)
# These are good for showcasing agent capabilities and for training curriculum
SUCCESSFUL_MISSIONS = [
    GoTogether,  # 55.0% success, 5.22 avg reward - Best overall
    OxygenBottleneck,  # 51.2% success, 3.02 avg reward
    CollectResourcesClassic,  # 50.0% success, 4.90 avg reward
    CollectResourcesSpread,  # 50.0% success, 4.45 avg reward
]

# Missions with moderate success (40-50% success rate)
# Still useful for training and evaluation
MODERATE_SUCCESS_MISSIONS = [
    ExtractorHub70,  # 43.8% success, 1.79 avg reward
    SingleUseSwarm,  # 42.5% success, 0.46 avg reward
]

# Recommended difficulty variants for scripted agents
# These are the difficulty variants where agents perform well
SUCCESSFUL_DIFFICULTIES = [
    "standard",  # 75.0% success, 4.58 avg reward - Best overall
    "story_mode",  # 72.1% success, 3.91 avg reward
    "energy_crisis",  # 73.1% success, 4.59 avg reward
    "speed_run",  # 70.2% success, 4.21 avg reward
    "single_use",  # 51.9% success, 2.96 avg reward - Moderate
]
