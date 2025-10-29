from __future__ import annotations

from cogames.cogs_vs_clips.mission import Mission, MissionVariant, Site
from cogames.cogs_vs_clips.missions import _add_make_env_modifier, get_map
from mettagrid.config.mettagrid_config import MettaGridConfig, ProtocolConfig
from mettagrid.map_builder.map_builder import MapBuilderConfig

# Dedicated site for eval missions
MACHINA_EVAL = Site(
    name="machina_eval",
    description="Evaluation missions on machina_eval_template-based maps.",
    map_builder=get_map("machina_eval_template.map"),
    min_cogs=1,
    max_cogs=8,
)


class _EvalMissionBase(Mission):
    site: Site = MACHINA_EVAL

    # Tunables (defaults; override in subclasses)
    map_name: str = "machina_eval_template.map"
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

    def instantiate(
        self, map_builder: MapBuilderConfig, num_cogs: int, variant: MissionVariant | None = None
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
        mission.clip_rate = self.clip_rate

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
                        current = int(getattr(obj, "max_uses"))
                        if current > 0:
                            setattr(obj, "max_uses", current * 2)
                    except Exception:
                        pass

        return _add_make_env_modifier(mission, _post)


class OxygenBottleneck(_EvalMissionBase):
    name: str = "oxygen_bottleneck"
    description: str = "Oxygen paces assembly; batch other resources."
    map_name: str = "machina_eval_exp02.map"
    oxygen_eff: int = 50
    energy_regen: int = 1
    max_uses_charger: int = 0
    max_uses_carbon: int = 100
    max_uses_oxygen: int = 20
    max_uses_germanium: int = 10
    max_uses_silicon: int = 100


class GermaniumRush(_EvalMissionBase):
    name: str = "germanium_rush"
    description: str = "Race to limited germanium before it runs out."
    map_name: str = "machina_eval_exp03.map"
    max_uses_germanium: int = 10
    max_uses_carbon: int = 100
    max_uses_oxygen: int = 50
    max_uses_silicon: int = 100
    max_uses_charger: int = 0


class SiliconWorkbench(_EvalMissionBase):
    name: str = "silicon_workbench"
    description: str = "Silicon-rich environment; convert energy to silicon."
    map_name: str = "machina_eval_exp04.map"
    silicon_eff: int = 150
    max_uses_silicon: int = 200
    max_uses_oxygen: int = 50
    max_uses_carbon: int = 100
    max_uses_germanium: int = 10
    max_uses_charger: int = 0


class CarbonDesert(_EvalMissionBase):
    name: str = "carbon_desert"
    description: str = "Sparse carbon dictates routes."
    map_name: str = "machina_eval_exp05.map"
    max_uses_carbon: int = 30
    max_uses_oxygen: int = 50
    max_uses_germanium: int = 10
    max_uses_silicon: int = 100
    max_uses_charger: int = 0


class SingleUseWorld(_EvalMissionBase):
    name: str = "single_use_world"
    description: str = "Every station can be used exactly once."
    map_name: str = "machina_eval_exp06.map"
    max_uses_charger: int = 1
    max_uses_carbon: int = 1
    max_uses_oxygen: int = 1
    max_uses_germanium: int = 1
    max_uses_silicon: int = 1


class SlowOxygen(_EvalMissionBase):
    name: str = "slow_oxygen"
    description: str = "Very slow oxygen; interleave partial-usage taps."
    map_name: str = "machina_eval_exp07.map"
    oxygen_eff: int = 25
    energy_regen: int = 2
    max_uses_oxygen: int = 100
    max_uses_carbon: int = 100
    max_uses_germanium: int = 10
    max_uses_silicon: int = 100
    max_uses_charger: int = 0


class HighRegenSprint(_EvalMissionBase):
    name: str = "high_regen_sprint"
    description: str = "High regen; minimize charger dependency."
    map_name: str = "machina_eval_exp08.map"
    energy_regen: int = 3
    max_uses_carbon: int = 100
    max_uses_oxygen: int = 50
    max_uses_germanium: int = 10
    max_uses_silicon: int = 100
    max_uses_charger: int = 0


class SparseBalanced(_EvalMissionBase):
    name: str = "sparse_balanced"
    description: str = "Evenly sparse resources; balanced routing."
    map_name: str = "machina_eval_exp09.map"
    max_uses_carbon: int = 50
    max_uses_oxygen: int = 50
    max_uses_germanium: int = 10
    max_uses_silicon: int = 50
    max_uses_charger: int = 0


class GermaniumClutch(_EvalMissionBase):
    name: str = "germanium_clutch"
    description: str = "A single germanium line determines success."
    map_name: str = "machina_eval_exp10.map"
    max_uses_germanium: int = 2
    max_uses_carbon: int = 100
    max_uses_oxygen: int = 50
    max_uses_silicon: int = 100
    max_uses_charger: int = 0


# -----------------------------
# New Single-Agent Eval Missions
# -----------------------------


class CollectTheResourcesEasy(_EvalMissionBase):
    name: str = "collect_the_resources_easy"
    description: str = "Collect all four resources, assemble a HEART, and deposit in chest (small/easy)."
    map_name: str = "eval_collect_resources_easy.map"


class CollectTheResourcesMedium(_EvalMissionBase):
    name: str = "collect_the_resources_medium"
    description: str = "Collect all four resources, assemble a HEART, and deposit in chest (medium)."
    map_name: str = "eval_collect_resources_medium.map"


class CollectTheResourcesHard(_EvalMissionBase):
    name: str = "collect_the_resources_hard"
    description: str = "Collect all four resources, assemble a HEART, and deposit in chest (large/hard)."
    map_name: str = "eval_collect_resources_hard.map"


class EnergyStarved(_EvalMissionBase):
    name: str = "energy_starved"
    description: str = "Low regen; requires careful charging and routing."
    map_name: str = "eval_energy_starved.map"
    charger_eff: int = 80
    energy_regen: int = 0
    max_uses_charger: int = 0


class OxygenBottleneck(_EvalMissionBase):
    name: str = "oxygen_bottleneck"
    description: str = "Oxygen paces assembly; low efficiency and limited uses."
    map_name: str = "eval_oxygen_bottleneck.map"
    oxygen_eff: int = 50
    max_uses_oxygen: int = 20


class GeraniumForage(_EvalMissionBase):
    name: str = "geranium_forage"
    description: str = "All resources near base except germanium; forage further away to find it."
    map_name: str = "eval_germanium_forage.map"


class SingleUseWorld(_EvalMissionBase):
    name: str = "single_use_world"
    description: str = "Every station can be used exactly once across more complex terrain."
    map_name: str = "eval_single_use_world.map"
    max_uses_charger: int = 1
    max_uses_carbon: int = 1
    max_uses_oxygen: int = 1
    max_uses_germanium: int = 1
    max_uses_silicon: int = 1

    def instantiate(
        self, map_builder: MapBuilderConfig, num_cogs: int, variant: MissionVariant | None = None
    ) -> "Mission":
        # Use base behavior to set up env and QoL tweaks
        mission = super().instantiate(map_builder, num_cogs, variant)

        # Re-enforce strict single-use on extractors AFTER any global doubling logic
        def _enforce_strict_single_use(cfg: MettaGridConfig) -> None:
            for key in ("carbon_extractor", "oxygen_extractor", "germanium_extractor", "silicon_extractor"):
                obj = cfg.game.objects.get(key)
                if obj is None:
                    continue
                try:
                    obj.max_uses = 1
                    # Prefer partial usage semantics if supported by the object type
                    if hasattr(obj, "allow_partial_usage"):
                        setattr(obj, "allow_partial_usage", True)
                except Exception:
                    pass

        return _add_make_env_modifier(mission, _enforce_strict_single_use)


class BalancedSpread(_EvalMissionBase):
    name: str = "balanced_spread"
    description: str = "Resources spread out; agent must forage far and return efficiently."
    map_name: str = "eval_balanced_spread.map"


# Legacy missions from daphne branch
class GermaniumRush(_EvalMissionBase):
    name: str = "germanium_rush"
    description: str = "Germanium-focused resource collection."
    map_name: str = "eval_germanium_rush.map"


class SiliconWorkbench(_EvalMissionBase):
    name: str = "silicon_workbench"
    description: str = "Silicon-heavy evaluation mission."
    map_name: str = "eval_silicon_workbench.map"


class CarbonDesert(_EvalMissionBase):
    name: str = "carbon_desert"
    description: str = "Limited carbon sources."
    map_name: str = "eval_carbon_desert.map"


class SlowOxygen(_EvalMissionBase):
    name: str = "slow_oxygen"
    description: str = "Reduced oxygen extraction efficiency."
    map_name: str = "eval_slow_oxygen.map"
    oxygen_eff: int = 25


class HighRegenSprint(_EvalMissionBase):
    name: str = "high_regen_sprint"
    description: str = "High energy regen for fast-paced missions."
    map_name: str = "eval_high_regen_sprint.map"
    energy_regen: int = 3


class SparseBalanced(_EvalMissionBase):
    name: str = "sparse_balanced"
    description: str = "Sparse but balanced resource distribution."
    map_name: str = "eval_sparse_balanced.map"


class GermaniumClutch(_EvalMissionBase):
    name: str = "germanium_clutch"
    description: str = "Critical germanium shortage scenario."
    map_name: str = "eval_germanium_clutch.map"
    max_uses_charger: int = 0


# -----------------------------
# Clipping Evaluation Missions
# -----------------------------


class ClipCarbon(_EvalMissionBase):
    name: str = "clip_carbon"
    description: str = "Carbon extractor starts clipped; unclip using gear crafted from another resource."
    map_name: str = "eval_clip_carbon.map"
    clip_rate: float = 0.0

    def instantiate(
        self, map_builder: MapBuilderConfig, num_cogs: int, variant: MissionVariant | None = None
    ) -> "Mission":
        mission = super().instantiate(map_builder, num_cogs, variant)
        # Ensure carbon is initially clipped; O2/G/Si remain available to craft gear
        mission.carbon_extractor.start_clipped = True

        # Deterministic unclipping: require modulator (crafted from oxygen)
        def _filter_unclip(cfg: MettaGridConfig) -> None:
            if cfg.game.clipper is None:
                return
            cfg.game.clipper.unclipping_recipes = [
                r for r in cfg.game.clipper.unclipping_recipes if r.input_resources == {"modulator": 1}
            ]

        # Single-agent gear crafting: gear glyph crafts modulator (consumes oxygen)
        def _tweak_assembler(cfg: MettaGridConfig) -> None:
            asm = cfg.game.objects.get("assembler")
            if asm is None:
                return
            recipe = ProtocolConfig(input_resources={"oxygen": 1}, output_resources={"modulator": 1})
            single_gear = (["gear"], recipe)
            if not any(
                g == ["gear"] and getattr(p, "output_resources", {}) == {"modulator": 1} for g, p in asm.recipes
            ):
                asm.recipes = [single_gear, *asm.recipes]

        mission = _add_make_env_modifier(mission, _filter_unclip)
        return _add_make_env_modifier(mission, _tweak_assembler)


class ClipOxygen(_EvalMissionBase):
    name: str = "clip_oxygen"
    description: str = "Oxygen extractor starts clipped; unclip via gear crafted from carbon/silicon/germanium."
    map_name: str = "eval_clip_oxygen.map"
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

        # Increase agent inventory caps for this eval
        def _expand_inventory(cfg: MettaGridConfig) -> None:
            try:
                limits = cfg.game.agent.resource_limits
                # Bump core resources and gear caps; keep heart/energy as-is
                limits.update(
                    {
                        "carbon": 200,
                        "oxygen": 200,
                        "germanium": 200,
                        "silicon": 200,
                        "decoder": 10,
                        "modulator": 10,
                        "scrambler": 10,
                        "resonator": 10,
                    }
                )
            except Exception:
                pass

        mission = _add_make_env_modifier(mission, _filter_unclip)
        mission = _add_make_env_modifier(mission, _tweak_assembler)
        return _add_make_env_modifier(mission, _expand_inventory)


class ClipGermanium(_EvalMissionBase):
    name: str = "clip_germanium"
    description: str = "Germanium extractor starts clipped; craft gear from carbon/oxygen/silicon to unclip."
    map_name: str = "machina_eval_exp03.map"
    clip_rate: float = 0.0

    def instantiate(
        self, map_builder: MapBuilderConfig, num_cogs: int, variant: MissionVariant | None = None
    ) -> "Mission":
        mission = super().instantiate(map_builder, num_cogs, variant)
        mission.germanium_extractor.start_clipped = True

        # Deterministic unclipping: require resonator (crafted from silicon)
        def _filter_unclip(cfg: MettaGridConfig) -> None:
            if cfg.game.clipper is None:
                return
            cfg.game.clipper.unclipping_recipes = [
                r for r in cfg.game.clipper.unclipping_recipes if r.input_resources == {"resonator": 1}
            ]

        # Single-agent gear crafting: gear glyph crafts resonator (consumes silicon)
        def _tweak_assembler(cfg: MettaGridConfig) -> None:
            asm = cfg.game.objects.get("assembler")
            if asm is None:
                return
            recipe = ProtocolConfig(input_resources={"silicon": 1}, output_resources={"resonator": 1})
            single_gear = (["gear"], recipe)
            if not any(
                g == ["gear"] and getattr(p, "output_resources", {}) == {"resonator": 1} for g, p in asm.recipes
            ):
                asm.recipes = [single_gear, *asm.recipes]

        mission = _add_make_env_modifier(mission, _filter_unclip)
        return _add_make_env_modifier(mission, _tweak_assembler)


class ClipSilicon(_EvalMissionBase):
    name: str = "clip_silicon"
    description: str = "Silicon extractor starts clipped; unclip using gear crafted from carbon/oxygen/germanium."
    map_name: str = "machina_eval_exp04.map"
    clip_rate: float = 0.0

    def instantiate(
        self, map_builder: MapBuilderConfig, num_cogs: int, variant: MissionVariant | None = None
    ) -> "Mission":
        mission = super().instantiate(map_builder, num_cogs, variant)
        mission.silicon_extractor.start_clipped = True

        # Deterministic unclipping: require scrambler (crafted from germanium)
        def _filter_unclip(cfg: MettaGridConfig) -> None:
            if cfg.game.clipper is None:
                return
            cfg.game.clipper.unclipping_recipes = [
                r for r in cfg.game.clipper.unclipping_recipes if r.input_resources == {"scrambler": 1}
            ]

        # Single-agent gear crafting: gear glyph crafts scrambler (consumes germanium)
        def _tweak_assembler(cfg: MettaGridConfig) -> None:
            asm = cfg.game.objects.get("assembler")
            if asm is None:
                return
            recipe = ProtocolConfig(input_resources={"germanium": 1}, output_resources={"scrambler": 1})
            single_gear = (["gear"], recipe)
            if not any(
                g == ["gear"] and getattr(p, "output_resources", {}) == {"scrambler": 1} for g, p in asm.recipes
            ):
                asm.recipes = [single_gear, *asm.recipes]

        mission = _add_make_env_modifier(mission, _filter_unclip)
        return _add_make_env_modifier(mission, _tweak_assembler)


class ClipCharger(_EvalMissionBase):
    name: str = "clip_charger"
    description: str = "Charger starts clipped; craft any gear to unclip and manage energy carefully."
    map_name: str = "machina_eval_exp05.map"
    clip_rate: float = 0.0

    def instantiate(
        self, map_builder: MapBuilderConfig, num_cogs: int, variant: MissionVariant | None = None
    ) -> "Mission":
        mission = super().instantiate(map_builder, num_cogs, variant)
        mission.charger.start_clipped = True
        return mission


EVAL_MISSIONS = [
    CollectTheResourcesEasy,
    CollectTheResourcesMedium,
    CollectTheResourcesHard,
    EnergyStarved,
    OxygenBottleneck,
    GeraniumForage,
    SingleUseWorld,
    BalancedSpread,
    GermaniumRush,
    SiliconWorkbench,
    CarbonDesert,
    SlowOxygen,
    HighRegenSprint,
    SparseBalanced,
    GermaniumClutch,
    ClipCarbon,
    ClipOxygen,
]
