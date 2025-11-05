from __future__ import annotations

import logging

from cogames.cogs_vs_clips.mission import Mission, MissionVariant
from cogames.cogs_vs_clips.mission_utils import _add_make_env_modifier, get_map
from cogames.cogs_vs_clips.sites import EVALS, Site
from mettagrid.config.mettagrid_config import MettaGridConfig, ProtocolConfig
from mettagrid.map_builder.map_builder import MapBuilderConfig

logger = logging.getLogger(__name__)


class _EvalMissionBase(Mission):
    # Shared site for all eval missions
    site: Site = EVALS

    # Tunables (defaults; override in subclasses)
    map_name: str = "evals/eval_collect_resources.map"

    # Clipping
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

    def instantiate(
        self, map_builder: MapBuilderConfig, num_cogs: int, variant: MissionVariant | None = None, **kwargs
    ) -> "Mission":
        # Force map
        forced_map = get_map(self.map_name)
        mission = super().instantiate(forced_map, num_cogs, variant)

        # Apply pre-make_env efficiency and regen knobs
        mission.charger.efficiency = self.charger_eff
        mission.carbon_extractor.efficiency = self.carbon_eff
        mission.oxygen_extractor.efficiency = self.oxygen_eff
        mission.germanium_extractor.efficiency = self.germanium_eff
        mission.silicon_extractor.efficiency = self.silicon_eff
        mission.energy_regen_amount = self.energy_regen
        mission.inventory_regen_interval = self.inventory_regen_interval
        mission.clip_rate = self.clip_rate

        # Post-make_env adjust max uses on built objects
        # Note: For clipping configuration, use difficulty variants (e.g., CLIPPED_OXYGEN)
        # instead of setting clip_rate directly on missions
        def _post(cfg: MettaGridConfig) -> None:
            cfg.game.map_builder = forced_map
            # Set episode length for all evals
            cfg.game.max_steps = 1000
            # Make HEART crafting feasible with a single agent using the heart glyph
            assembler_obj = cfg.game.objects.get("assembler")
            if assembler_obj is not None and hasattr(assembler_obj, "heart_cost"):
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

            if self.max_uses_charger is not None and "charger" in cfg.game.objects:
                cfg.game.objects["charger"].max_uses = self.max_uses_charger
            if self.max_uses_carbon is not None and "carbon_extractor" in cfg.game.objects:
                cfg.game.objects["carbon_extractor"].max_uses = self.max_uses_carbon
            if self.max_uses_oxygen is not None and "oxygen_extractor" in cfg.game.objects:
                cfg.game.objects["oxygen_extractor"].max_uses = self.max_uses_oxygen
            if self.max_uses_germanium is not None and "germanium_extractor" in cfg.game.objects:
                cfg.game.objects["germanium_extractor"].max_uses = self.max_uses_germanium
            if self.max_uses_silicon is not None and "silicon_extractor" in cfg.game.objects:
                cfg.game.objects["silicon_extractor"].max_uses = self.max_uses_silicon

            # Global quality-of-life tweaks for evals
            # 1) Double agent inventory caps for core resources and gear
            try:
                limits = cfg.game.agent.resource_limits
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
                obj = cfg.game.objects.get(obj_name)
                if obj is not None and hasattr(obj, "max_uses"):
                    try:
                        current = int(obj.max_uses)
                        if current > 0:
                            obj.max_uses = current * 2
                    except Exception:
                        pass

        return _add_make_env_modifier(mission, _post)


class OxygenBottleneck(_EvalMissionBase):
    name: str = "oxygen_bottleneck"
    description: str = "Oxygen paces assembly; batch other resources."
    map_name: str = "evals/eval_oxygen_bottleneck.map"
    charger_eff: int = 130
    oxygen_eff: int = 60
    energy_regen: int = 2
    max_uses_charger: int = 0
    max_uses_carbon: int = 120
    max_uses_oxygen: int = 30
    max_uses_germanium: int = 15
    max_uses_silicon: int = 120
    site: Site = EVALS


class EnergyStarved(_EvalMissionBase):
    name: str = "energy_starved"
    description: str = "Low regen; requires careful charging and routing."
    map_name: str = "evals/eval_energy_starved.map"
    charger_eff: int = 90
    carbon_eff: int = 125
    oxygen_eff: int = 115
    germanium_eff: int = 100
    silicon_eff: int = 125
    energy_regen: int = 1
    inventory_regen_interval: int = 2
    max_uses_charger: int = 0


# -----------------------------
# Extractor Hub Missions
# -----------------------------


class ExtractorHub30(_EvalMissionBase):
    name: str = "extractor_hub_30"
    description: str = "Small 30x30 extractor hub."
    map_name: str = "evals/extractor_hub_30x30.map"
    energy_regen: int = 2
    charger_eff: int = 125
    germanium_eff: int = 90
    max_uses_germanium: int = 0


class ExtractorHub50(_EvalMissionBase):
    name: str = "extractor_hub_50"
    description: str = "Medium 50x50 extractor hub."
    map_name: str = "evals/extractor_hub_50x50.map"
    energy_regen: int = 2
    charger_eff: int = 125
    germanium_eff: int = 90
    max_uses_germanium: int = 0


class ExtractorHub70(_EvalMissionBase):
    name: str = "extractor_hub_70"
    description: str = "Large 70x70 extractor hub."
    map_name: str = "evals/extractor_hub_70x70.map"
    energy_regen: int = 2
    charger_eff: int = 130
    germanium_eff: int = 95
    max_uses_germanium: int = 0


class ExtractorHub80(_EvalMissionBase):
    name: str = "extractor_hub_80"
    description: str = "Large 80x80 extractor hub."
    map_name: str = "evals/extractor_hub_80x80.map"
    energy_regen: int = 2
    charger_eff: int = 135
    germanium_eff: int = 95
    max_uses_germanium: int = 0


class ExtractorHub100(_EvalMissionBase):
    name: str = "extractor_hub_100"
    description: str = "Extra large 100x100 extractor hub."
    map_name: str = "evals/extractor_hub_100x100.map"
    energy_regen: int = 2
    charger_eff: int = 140
    germanium_eff: int = 100
    max_uses_germanium: int = 0


# -----------------------------m
# Multi-agent Coordination Missions
# -----------------------------


class CollectResourcesClassic(_EvalMissionBase):
    name: str = "collect_resources_classic"
    description: str = "Collect resources on the classic layout; balanced routing near base."
    map_name: str = "evals/eval_collect_resources.map"
    energy_regen: int = 2
    charger_eff: int = 130
    carbon_eff: int = 125
    oxygen_eff: int = 115
    germanium_eff: int = 90
    silicon_eff: int = 125
    max_uses_germanium: int = 0
    max_uses_silicon: int = 0
    max_uses_carbon: int = 0
    max_uses_oxygen: int = 0


class CollectResourcesSpread(_EvalMissionBase):
    name: str = "collect_resources_spread"
    description: str = "Collect resources (scattered nearby), rally and chorus glyph at assembler."
    map_name: str = "evals/eval_collect_resources_medium.map"
    energy_regen: int = 2
    charger_eff: int = 135
    carbon_eff: int = 130
    oxygen_eff: int = 120
    germanium_eff: int = 95
    silicon_eff: int = 130
    max_uses_germanium: int = 0
    max_uses_silicon: int = 0
    max_uses_carbon: int = 0
    max_uses_oxygen: int = 0


class CollectFar(_EvalMissionBase):
    name: str = "collect_far"
    description: str = "Collect resources scattered far; coordinate routes, chorus glyph, single carrier deposits."
    map_name: str = "evals/eval_collect_resources_hard.map"
    energy_regen: int = 2
    charger_eff: int = 135
    carbon_eff: int = 130
    oxygen_eff: int = 120
    germanium_eff: int = 100
    silicon_eff: int = 135
    max_uses_germanium: int = 20
    max_uses_silicon: int = 25
    max_uses_carbon: int = 40
    max_uses_oxygen: int = 30


class DivideAndConquer(_EvalMissionBase):
    name: str = "divide_and_conquer"
    description: str = "Resources split by regions; specialize per resource and reconvene at base."
    map_name: str = "evals/eval_divide_and_conquer.map"
    energy_regen: int = 2
    charger_eff: int = 130
    carbon_eff: int = 125
    oxygen_eff: int = 120
    germanium_eff: int = 95
    silicon_eff: int = 130
    max_uses_germanium: int = 10
    max_uses_silicon: int = 15
    max_uses_carbon: int = 25
    max_uses_oxygen: int = 20


class GoTogether(_EvalMissionBase):
    name: str = "go_together"
    description: str = "Objects favor collective glyphing; travel and return as a pack."
    map_name: str = "evals/eval_balanced_spread.map"
    energy_regen: int = 2
    charger_eff: int = 140
    carbon_eff: int = 130
    oxygen_eff: int = 125
    germanium_eff: int = 100
    silicon_eff: int = 135
    max_uses_germanium: int = 10
    max_uses_silicon: int = 20
    max_uses_carbon: int = 30
    max_uses_oxygen: int = 25

    def instantiate(
        self, map_builder: MapBuilderConfig, num_cogs: int, variant: MissionVariant | None = None, **kwargs
    ) -> "Mission":
        # Enforce at least two agents
        enforced_cogs = max(2, num_cogs)
        return super().instantiate(map_builder, enforced_cogs, variant)


class SingleUseSwarm(_EvalMissionBase):
    name: str = "single_use_swarm"
    description: str = "Multi-agent variant of SingleUseWorld; stations max_uses=1, team must fan out and reconverge."
    map_name: str = "evals/eval_single_use_world.map"
    energy_regen: int = 2
    charger_eff: int = 140
    carbon_eff: int = 130
    oxygen_eff: int = 125
    germanium_eff: int = 105
    silicon_eff: int = 135
    max_uses_charger: int = 0
    max_uses_carbon: int = 1
    max_uses_oxygen: int = 1
    max_uses_germanium: int = 1
    max_uses_silicon: int = 1

    def instantiate(
        self, map_builder: MapBuilderConfig, num_cogs: int, variant: MissionVariant | None = None, **kwargs
    ) -> "Mission":
        # Enforce at least two agents
        enforced_cogs = max(2, num_cogs)
        return super().instantiate(map_builder, enforced_cogs, variant)


EVAL_MISSIONS = [
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
