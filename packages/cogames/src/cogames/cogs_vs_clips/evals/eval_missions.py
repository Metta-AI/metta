from __future__ import annotations

import logging
import random

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
    map_name: str = "evals/eval_collect_resources_easy.map"

    # Clipping
    clip_rate: float = 0.0
    charger_eff: int = 100
    carbon_eff: int = 100
    oxygen_eff: int = 100
    germanium_eff: int = 1
    silicon_eff: int = 100
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

        # When clip_rate > 0, use explicit immune extractor or randomly select one
        # This ensures at least one resource is always available for crafting unclip items
        immune_extractor_type = None
        if self.clip_rate > 0:
            # Prefer explicitly provided immune extractor; otherwise pick randomly excluding the clipped target
            immune_extractor_type = getattr(self, "immune_extractor", None)
            if immune_extractor_type is None:
                candidates = [
                    "carbon_extractor",
                    "oxygen_extractor",
                    "germanium_extractor",
                    "silicon_extractor",
                ]
                explicitly_clipped = getattr(self, "explicitly_clipped_extractor", None)
                if explicitly_clipped in candidates:
                    candidates = [e for e in candidates if e != explicitly_clipped]
                immune_extractor_type = random.choice(candidates)
                logger.debug(
                    "[EvalMission] clip_rate=%s: %s randomly clip-immune",
                    self.clip_rate,
                    immune_extractor_type,
                )
            else:
                logger.debug(
                    "[EvalMission] clip_rate=%s: %s explicitly set as clip-immune",
                    self.clip_rate,
                    immune_extractor_type,
                )

        # Post-make_env adjust max uses on built objects
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
                            "silicon": 5,
                            "energy": 2,
                        },
                        output_resources={"heart": 1},
                    )
                    heart_recipes = [(["heart"] * (i + 1), tiny) for i in range(4)]
                    redheart_recipes = [(["red-heart"] * (i + 1), tiny) for i in range(4)]
                    assembler_obj.recipes = [*heart_recipes, *redheart_recipes, *assembler_obj.recipes]

            # Add unclip recipes when clip_rate > 0
            if self.clip_rate > 0 and assembler_obj is not None and hasattr(assembler_obj, "recipes"):
                unclip_recipes = [
                    (["gear"], ProtocolConfig(input_resources={"carbon": 1}, output_resources={"decoder": 1})),
                    (["gear"], ProtocolConfig(input_resources={"oxygen": 1}, output_resources={"modulator": 1})),
                    (["gear"], ProtocolConfig(input_resources={"silicon": 1}, output_resources={"resonator": 1})),
                    (["gear"], ProtocolConfig(input_resources={"germanium": 1}, output_resources={"scrambler": 1})),
                ]
                # Add unclip recipes if they don't already exist
                existing_outputs = {tuple(sorted(p.output_resources.keys())) for g, p in assembler_obj.recipes}
                for glyph, protocol in unclip_recipes:
                    output_key = tuple(sorted(protocol.output_resources.keys()))
                    if output_key not in existing_outputs:
                        assembler_obj.recipes.append((glyph, protocol))
                        existing_outputs.add(output_key)
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

            # Mark immune extractor to never be clipped (when clip_rate > 0)
            if immune_extractor_type is not None and immune_extractor_type in cfg.game.objects:
                # Mark the immune extractor as clip-immune
                # The C++ field is called "clip_immune" not "is_clip_immune"
                extractor_obj = cfg.game.objects[immune_extractor_type]
                try:
                    extractor_obj.clip_immune = True
                    extractor_obj.start_clipped = False
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Cannot set clip_immune on {immune_extractor_type}: {e}")

            # ALWAYS make charger immune - it's needed for energy and can't be unclipped
            if self.clip_rate > 0 and "charger" in cfg.game.objects:
                charger_obj = cfg.game.objects["charger"]
                try:
                    charger_obj.clip_immune = True
                    charger_obj.start_clipped = False
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Cannot set clip_immune on charger: {e}")

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
    oxygen_eff: int = 50
    energy_regen: int = 1
    max_uses_charger: int = 0
    max_uses_carbon: int = 100
    max_uses_oxygen: int = 20
    max_uses_germanium: int = 10
    max_uses_silicon: int = 100
    site: Site = EVALS


class EnergyStarved(_EvalMissionBase):
    name: str = "energy_starved"
    description: str = "Low regen; requires careful charging and routing."
    map_name: str = "evals/eval_energy_starved.map"
    charger_eff: int = 80
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


class ExtractorHub50(_EvalMissionBase):
    name: str = "extractor_hub_50"
    description: str = "Medium 50x50 extractor hub."
    map_name: str = "evals/extractor_hub_50x50.map"


class ExtractorHub70(_EvalMissionBase):
    name: str = "extractor_hub_70"
    description: str = "Large 70x70 extractor hub."
    map_name: str = "evals/extractor_hub_70x70.map"


class ExtractorHub80(_EvalMissionBase):
    name: str = "extractor_hub_80"
    description: str = "Large 80x80 extractor hub."
    map_name: str = "evals/extractor_hub_80x80.map"


class ExtractorHub100(_EvalMissionBase):
    name: str = "extractor_hub_100"
    description: str = "Extra large 100x100 extractor hub."
    map_name: str = "evals/extractor_hub_100x100.map"


# -----------------------------m
# Multi-agent Coordination Missions
# -----------------------------


class CollectResourcesBase(_EvalMissionBase):
    name: str = "collect_resources_base"
    description: str = "Collect resources (near base), rally and chorus glyph; single carrier deposits."
    map_name: str = "evals/eval_collect_resources_easy.map"


class CollectResourcesClassic(_EvalMissionBase):
    name: str = "collect_resources_classic"
    description: str = "Collect resources on the classic layout; balanced routing near base."
    map_name: str = "evals/eval_collect_resources.map"


class CollectResourcesSpread(_EvalMissionBase):
    name: str = "collect_resources_spread"
    description: str = "Collect resources (scattered nearby), rally and chorus glyph at assembler."
    map_name: str = "evals/eval_collect_resources_medium.map"


class CollectFar(_EvalMissionBase):
    name: str = "collect_far"
    description: str = "Collect resources scattered far; coordinate routes, chorus glyph, single carrier deposits."
    map_name: str = "evals/eval_collect_resources_hard.map"


class DivideAndConquer(_EvalMissionBase):
    name: str = "divide_and_conquer"
    description: str = "Resources split by regions; specialize per resource and reconvene at base."
    map_name: str = "evals/eval_divide_and_conquer.map"


class GoTogether(_EvalMissionBase):
    name: str = "go_together"
    description: str = "Objects favor collective glyphing; travel and return as a pack."
    map_name: str = "evals/eval_balanced_spread.map"

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


# -----------------------------
# Clipping Evaluation Missions
# -----------------------------


class ClipOxygen(_EvalMissionBase):
    name: str = "clip_oxygen"
    description: str = "Oxygen extractor starts clipped; unclip via gear crafted from carbon/silicon/germanium."
    map_name: str = "evals/eval_clip_oxygen.map"
    clip_rate: float = 0.0

    def instantiate(
        self, map_builder: MapBuilderConfig, num_cogs: int, variant: MissionVariant | None = None
    ) -> "Mission":
        mission = super().instantiate(map_builder, num_cogs, variant)
        mission.oxygen_extractor.start_clipped = True

        # Deterministic unclipping: require decoder (crafted from carbon)
        def _filter_unclip(cfg: MettaGridConfig) -> None:
            if cfg.game.clipper is None:
                return
            cfg.game.clipper.unclipping_recipes = [
                r for r in cfg.game.clipper.unclipping_recipes if r.input_resources == {"decoder": 1}
            ]

        # Single-agent gear crafting: gear glyph crafts decoder (consumes carbon)
        def _tweak_assembler(cfg: MettaGridConfig) -> None:
            asm = cfg.game.objects.get("assembler")
            if asm is None:
                return
            recipe = ProtocolConfig(input_resources={"carbon": 1}, output_resources={"decoder": 1})
            single_gear = (["gear"], recipe)
            if not any(g == ["gear"] and getattr(p, "output_resources", {}) == {"decoder": 1} for g, p in asm.recipes):
                asm.recipes = [single_gear, *asm.recipes]

        mission = _add_make_env_modifier(mission, _filter_unclip)
        return _add_make_env_modifier(mission, _tweak_assembler)


EVAL_MISSIONS = [
    EnergyStarved,
    OxygenBottleneck,
    ExtractorHub30,
    ExtractorHub50,
    ExtractorHub70,
    ExtractorHub80,
    ExtractorHub100,
    CollectResourcesBase,
    CollectResourcesClassic,
    CollectResourcesSpread,
    CollectFar,
    DivideAndConquer,
    GoTogether,
    SingleUseSwarm,
    ClipOxygen,
]


# -----------------------------------------------------------------------------
# Generic clipping profile helper
# -----------------------------------------------------------------------------


def apply_clip_profile(
    mission: Mission,
    *,
    target: str | None,
    clip_rate: float | None = None,
) -> Mission:
    """Apply a clipping profile to a mission to match behavior of explicit Clip* missions.

    - Sets mission.clip_rate if provided
    - start_clipped on the target extractor (or charger) if target provided
    - Filters unclipping recipes to the gear associated with the target
    - Adds a single-agent assembler recipe to craft that gear from the correct resource
    - Ensures the gear resource extractor is clip-immune and not start_clipped

    Mapping:
      carbon -> require modulator (from oxygen), make oxygen->modulator recipe
      oxygen -> require decoder (from carbon), make carbon->decoder recipe
      germanium -> require resonator (from silicon), make silicon->resonator recipe
      silicon -> require scrambler (from germanium), make germanium->scrambler recipe
      charger -> start_clipped charger; unclipping recipes unchanged
    """
    if clip_rate is not None:
        try:
            mission.clip_rate = float(clip_rate)
        except Exception:
            pass

    if not target or target == "none":
        return mission

    # Set the specific station to start clipped
    try:
        if target == "carbon":
            mission.carbon_extractor.start_clipped = True
        elif target == "oxygen":
            mission.oxygen_extractor.start_clipped = True
        elif target == "germanium":
            mission.germanium_extractor.start_clipped = True
        elif target == "silicon":
            mission.silicon_extractor.start_clipped = True
        elif target == "charger":
            mission.charger.start_clipped = True
    except Exception:
        # Some missions may not expose all station configs; ignore if missing
        pass

    # Determine gear and resource mapping for unclipping
    gear_by_target: dict[str, tuple[str, str]] = {
        "carbon": ("modulator", "oxygen"),
        "oxygen": ("decoder", "carbon"),
        "germanium": ("resonator", "silicon"),
        "silicon": ("scrambler", "germanium"),
    }

    if target in gear_by_target:
        required_gear, resource_for_gear = gear_by_target[target]

        def _filter_unclip(cfg: MettaGridConfig) -> None:
            if cfg.game.clipper is None:
                return
            try:
                cfg.game.clipper.unclipping_recipes = [
                    r for r in cfg.game.clipper.unclipping_recipes if r.input_resources == {required_gear: 1}
                ]
            except Exception:
                pass

        def _tweak_assembler(cfg: MettaGridConfig) -> None:
            asm = cfg.game.objects.get("assembler")
            if asm is None:
                return
            try:
                recipe = ProtocolConfig(input_resources={resource_for_gear: 1}, output_resources={required_gear: 1})
                single_gear = (["gear"], recipe)
                if not any(
                    g == ["gear"] and getattr(p, "output_resources", {}) == {required_gear: 1} for g, p in asm.recipes
                ):
                    asm.recipes = [single_gear, *asm.recipes]
            except Exception:
                pass

        def _ensure_gear_resource_immune(cfg: MettaGridConfig) -> None:
            # Make the extractor for the gear resource immune (available at start)
            name = f"{resource_for_gear}_extractor"
            obj = cfg.game.objects.get(name)
            if obj is None:
                return
            try:
                if hasattr(obj, "clip_immune"):
                    obj.clip_immune = True
                if hasattr(obj, "start_clipped"):
                    obj.start_clipped = False
            except Exception:
                pass

        def _ensure_critical_stations_immune(cfg: MettaGridConfig) -> None:
            # Make charger, assembler, and chest immune when clip_rate > 0
            # These are essential for core gameplay and cannot be unclipped
            for station_name in ["charger", "assembler", "chest"]:
                obj = cfg.game.objects.get(station_name)
                if obj is None:
                    continue
                try:
                    if hasattr(obj, "clip_immune"):
                        obj.clip_immune = True
                    if hasattr(obj, "start_clipped"):
                        obj.start_clipped = False
                except Exception:
                    pass

        mission = _add_make_env_modifier(mission, _filter_unclip)
        mission = _add_make_env_modifier(mission, _tweak_assembler)
        mission = _add_make_env_modifier(mission, _ensure_gear_resource_immune)

    # When clip_rate > 0, always make critical stations immune
    if clip_rate is not None and clip_rate > 0:

        def _ensure_critical_stations_immune_global(cfg: MettaGridConfig) -> None:
            for station_name in ["charger", "assembler", "chest"]:
                obj = cfg.game.objects.get(station_name)
                if obj is None:
                    continue
                try:
                    if hasattr(obj, "clip_immune"):
                        obj.clip_immune = True
                    if hasattr(obj, "start_clipped"):
                        obj.start_clipped = False
                except Exception:
                    pass

        mission = _add_make_env_modifier(mission, _ensure_critical_stations_immune_global)

    # Charger target: only start_clipped like ClipCharger
    return mission
