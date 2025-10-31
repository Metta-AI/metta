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

        # Post-make_env adjust max uses and collective glyph requirements
        def _post(cfg: MettaGridConfig) -> None:
            cfg.game.map_builder = forced_map
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

            # Universal assembler collective glyphing: require min(num_cogs, 4) glyphers for heart recipes
            required = min(num_cogs, 4)
            asm = cfg.game.objects.get("assembler")
            if asm is not None and hasattr(asm, "recipes"):
                filtered = []
                for glyphs, proto in asm.recipes:
                    out = getattr(proto, "output_resources", {}) or {}
                    if isinstance(out, dict) and out.get("heart", 0):
                        # Keep only heart recipes that exactly match the required glyph count
                        if len(glyphs) == max(1, required):
                            filtered.append((glyphs, proto))
                    else:
                        filtered.append((glyphs, proto))
                # If no heart recipe matched (unexpected), fall back to original
                if any(getattr(p, "output_resources", {}).get("heart", 0) for _, p in asm.recipes) and not any(
                    getattr(p, "output_resources", {}).get("heart", 0) for _, p in filtered
                ):
                    filtered = asm.recipes
                asm.recipes = filtered

        return _add_make_env_modifier(mission, _post)


class EnergyStarved(_EvalMissionBase):
    name: str = "energy_starved"
    description: str = "Tight energy budget; route via chargers and batch trips."
    map_name: str = "eval_energy_starved.map"
    energy_regen: int = 0
    max_uses_charger: int = 0
    max_uses_carbon: int = 100
    max_uses_oxygen: int = 30
    max_uses_germanium: int = 5
    max_uses_silicon: int = 100


class OxygenBottleneck(_EvalMissionBase):
    name: str = "oxygen_bottleneck"
    description: str = "Oxygen paces assembly; batch other resources."
    map_name: str = "eval_oxygen_bottleneck.map"
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
    map_name: str = "eval_single_use_world.map"
    max_uses_charger: int = 0
    max_uses_carbon: int = 1
    max_uses_oxygen: int = 1
    max_uses_germanium: int = 1
    max_uses_silicon: int = 1

    def instantiate(
        self, map_builder: MapBuilderConfig, num_cogs: int, variant: MissionVariant | None = None
    ) -> "Mission":
        mission = super().instantiate(map_builder, num_cogs, variant)

        def _balance_outputs(cfg: MettaGridConfig) -> None:
            # Determine the active heart recipe after assembler filtering
            asm = cfg.game.objects.get("assembler")
            required_g = 3  # default fallback
            if asm is not None and getattr(asm, "recipes", None):
                for glyphs, proto in asm.recipes:
                    out = getattr(proto, "output_resources", {}) or {}
                    if out.get("heart", 0):
                        req = getattr(proto, "input_resources", {}) or {}
                        required_g = max(1, int(req.get("germanium", required_g)))
                        break

            # Set per-use outputs to match assembler requirements in one activation
            car = cfg.game.objects.get("carbon_extractor")
            if car is not None and getattr(car, "recipes", None):
                glyphs, proto = car.recipes[0]
                proto.output_resources = {"carbon": 20}
                car.recipes = [(glyphs, proto)]

            oxy = cfg.game.objects.get("oxygen_extractor")
            if oxy is not None and getattr(oxy, "recipes", None):
                glyphs, proto = oxy.recipes[0]
                proto.output_resources = {"oxygen": 20}
                oxy.recipes = [(glyphs, proto)]

            sil = cfg.game.objects.get("silicon_extractor")
            if sil is not None and getattr(sil, "recipes", None):
                glyphs, proto = sil.recipes[0]
                proto.output_resources = {"silicon": 50}
                sil.recipes = [(glyphs, proto)]

            ger = cfg.game.objects.get("germanium_extractor")
            if ger is not None and getattr(ger, "recipes", None):
                glyphs, proto = ger.recipes[0]
                proto.output_resources = {"germanium": required_g}
                ger.recipes = [(glyphs, proto)]

        return _add_make_env_modifier(mission, _balance_outputs)


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
    map_name: str = "eval_balanced_spread.map"
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
# Extractor Hub Missions
# -----------------------------


class ExtractorHub30(_EvalMissionBase):
    name: str = "extractor_hub_30"
    description: str = "Small 30x30 extractor hub."
    map_name: str = "extractor_hub_30x30.map"


class ExtractorHub50(_EvalMissionBase):
    name: str = "extractor_hub_50"
    description: str = "Medium 50x50 extractor hub."
    map_name: str = "extractor_hub_50x50.map"


class ExtractorHub70(_EvalMissionBase):
    name: str = "extractor_hub_70"
    description: str = "Large 70x70 extractor hub."
    map_name: str = "extractor_hub_70x70.map"


class ExtractorHub80(_EvalMissionBase):
    name: str = "extractor_hub_80"
    description: str = "Large 80x80 extractor hub."
    map_name: str = "extractor_hub_80x80.map"


class ExtractorHub100(_EvalMissionBase):
    name: str = "extractor_hub_100"
    description: str = "Extra large 100x100 extractor hub."
    map_name: str = "extractor_hub_100x100.map"


EVAL_MISSIONS: list[type[Mission]] = [
    EnergyStarved,
    OxygenBottleneck,
    SingleUseWorld,
    SparseBalanced,
]


# -----------------------------
# Multi-agent Coordination Missions
# -----------------------------


class CollectResourcesBase(_EvalMissionBase):
    name: str = "collect_resources_base"
    description: str = (
        "Collect resources (near base), rally and chorus glyph; single carrier deposits."
    )
    map_name: str = "eval_collect_resources_easy.map"


class CollectResourcesSpread(_EvalMissionBase):
    name: str = "collect_resources_spread"
    description: str = (
        "Collect resources (scattered nearby), rally and chorus glyph at assembler."
    )
    map_name: str = "eval_collect_resources_medium.map"


class CollectFar(_EvalMissionBase):
    name: str = "collect_far"
    description: str = (
        "Collect resources scattered far; coordinate routes, chorus glyph, single carrier deposits."
    )
    map_name: str = "eval_collect_the_resources_hard.map"


class DivideAndConquer(_EvalMissionBase):
    name: str = "divide_and_conquer"
    description: str = (
        "Resources split by regions; specialize per resource and reconvene at base."
    )
    map_name: str = "eval_divide_and_conquer.map"


class GoTogether(_EvalMissionBase):
    name: str = "go_together"
    description: str = "Objects favor collective glyphing; travel and return as a pack."
    map_name: str = "eval_balanced_spread.map"

    def instantiate(
        self, map_builder: MapBuilderConfig, num_cogs: int, variant: MissionVariant | None = None
    ) -> "Mission":
        # Enforce at least two agents
        enforced_cogs = max(2, num_cogs)
        mission = super().instantiate(map_builder, enforced_cogs, variant)

        # Require 2 glyphers at extractors; charger remains single-agent
        def _collective(cfg: MettaGridConfig) -> None:
            for key in [
                "carbon_extractor",
                "oxygen_extractor",
                "germanium_extractor",
                "silicon_extractor",
            ]:
                obj = cfg.game.objects.get(key)
                if obj is None or not hasattr(obj, "recipes"):
                    continue
                adjusted = []
                for glyphs, proto in obj.recipes:
                    gl = list(glyphs)
                    if len(gl) < 2:
                        gl = gl + ["heart"] * (2 - len(gl))
                    adjusted.append((gl, proto))
                obj.recipes = adjusted

        return _add_make_env_modifier(mission, _collective)


class SingleUseSwarm(_EvalMissionBase):
    name: str = "single_use_swarm"
    description: str = (
        "Multi-agent variant of SingleUseWorld; stations max_uses=1, team must fan out and reconverge."
    )
    map_name: str = "eval_single_use_world.map"
    max_uses_charger: int = 0
    max_uses_carbon: int = 1
    max_uses_oxygen: int = 1
    max_uses_germanium: int = 1
    max_uses_silicon: int = 1

    def instantiate(
        self, map_builder: MapBuilderConfig, num_cogs: int, variant: MissionVariant | None = None
    ) -> "Mission":
        # Enforce at least two agents
        enforced_cogs = max(2, num_cogs)
        mission = super().instantiate(map_builder, enforced_cogs, variant)

        def _balance_outputs(cfg: MettaGridConfig) -> None:
            asm = cfg.game.objects.get("assembler")
            required_g = 3
            if asm is not None and getattr(asm, "recipes", None):
                for glyphs, proto in asm.recipes:
                    out = getattr(proto, "output_resources", {}) or {}
                    if out.get("heart", 0):
                        req = getattr(proto, "input_resources", {}) or {}
                        required_g = max(1, int(req.get("germanium", required_g)))
                        break

            car = cfg.game.objects.get("carbon_extractor")
            if car is not None and getattr(car, "recipes", None):
                glyphs, proto = car.recipes[0]
                proto.output_resources = {"carbon": 20}
                car.recipes = [(glyphs, proto)]

            oxy = cfg.game.objects.get("oxygen_extractor")
            if oxy is not None and getattr(oxy, "recipes", None):
                glyphs, proto = oxy.recipes[0]
                proto.output_resources = {"oxygen": 20}
                oxy.recipes = [(glyphs, proto)]

            sil = cfg.game.objects.get("silicon_extractor")
            if sil is not None and getattr(sil, "recipes", None):
                glyphs, proto = sil.recipes[0]
                proto.output_resources = {"silicon": 50}
                sil.recipes = [(glyphs, proto)]

            ger = cfg.game.objects.get("germanium_extractor")
            if ger is not None and getattr(ger, "recipes", None):
                glyphs, proto = ger.recipes[0]
                proto.output_resources = {"germanium": required_g}
                ger.recipes = [(glyphs, proto)]

        return _add_make_env_modifier(mission, _balance_outputs)


# Register coordination missions
EVAL_MISSIONS.extend(
    [
        CollectResourcesBase,
        CollectResourcesSpread,
        CollectFar,
        DivideAndConquer,
        GoTogether,
        SingleUseSwarm,
    ]
)


# -----------------------------
# Clipping Evaluation Missions
# -----------------------------


class ClipCarbon(_EvalMissionBase):
    name: str = "clip_carbon"
    description: str = "Carbon extractor starts clipped; unclip using gear crafted from another resource."
    map_name: str = "machina_eval_exp01.map"
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

        mission = _add_make_env_modifier(mission, _filter_unclip)
        return _add_make_env_modifier(mission, _tweak_assembler)





class ClipExtractorHub30(_EvalMissionBase):
    name: str = "clip_extractor_hub_30"
    description: str = "Small 30x30 extractor hub with clipped oxygen."
    map_name: str = "extractor_hub_30x30.map"
    clip_rate: float = 0.0

    def instantiate(
        self, map_builder: MapBuilderConfig, num_cogs: int, variant: MissionVariant | None = None
    ) -> "Mission":
        mission = super().instantiate(map_builder, num_cogs, variant)
        mission.oxygen_extractor.start_clipped = True

        def _filter_unclip(cfg: MettaGridConfig) -> None:
            if cfg.game.clipper is None:
                return
            cfg.game.clipper.unclipping_recipes = [
                r for r in cfg.game.clipper.unclipping_recipes if r.input_resources == {"decoder": 1}
            ]

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


class ClipExtractorHub50(_EvalMissionBase):
    name: str = "clip_extractor_hub_50"
    description: str = "Medium 50x50 extractor hub with clipped oxygen."
    map_name: str = "extractor_hub_50x50.map"
    clip_rate: float = 0.0

    def instantiate(
        self, map_builder: MapBuilderConfig, num_cogs: int, variant: MissionVariant | None = None
    ) -> "Mission":
        mission = super().instantiate(map_builder, num_cogs, variant)
        mission.oxygen_extractor.start_clipped = True

        def _filter_unclip(cfg: MettaGridConfig) -> None:
            if cfg.game.clipper is None:
                return
            cfg.game.clipper.unclipping_recipes = [
                r for r in cfg.game.clipper.unclipping_recipes if r.input_resources == {"decoder": 1}
            ]

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


class ClipExtractorHub70(_EvalMissionBase):
    name: str = "clip_extractor_hub_70"
    description: str = "Large 70x70 extractor hub with clipped oxygen."
    map_name: str = "extractor_hub_70x70.map"
    clip_rate: float = 0.0

    def instantiate(
        self, map_builder: MapBuilderConfig, num_cogs: int, variant: MissionVariant | None = None
    ) -> "Mission":
        mission = super().instantiate(map_builder, num_cogs, variant)
        mission.oxygen_extractor.start_clipped = True

        def _filter_unclip(cfg: MettaGridConfig) -> None:
            if cfg.game.clipper is None:
                return
            cfg.game.clipper.unclipping_recipes = [
                r for r in cfg.game.clipper.unclipping_recipes if r.input_resources == {"decoder": 1}
            ]

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


class ClipExtractorHub80(_EvalMissionBase):
    name: str = "clip_extractor_hub_80"
    description: str = "Large 80x80 extractor hub with clipped oxygen."
    map_name: str = "extractor_hub_80x80.map"
    clip_rate: float = 0.0

    def instantiate(
        self, map_builder: MapBuilderConfig, num_cogs: int, variant: MissionVariant | None = None
    ) -> "Mission":
        mission = super().instantiate(map_builder, num_cogs, variant)
        mission.oxygen_extractor.start_clipped = True

        def _filter_unclip(cfg: MettaGridConfig) -> None:
            if cfg.game.clipper is None:
                return
            cfg.game.clipper.unclipping_recipes = [
                r for r in cfg.game.clipper.unclipping_recipes if r.input_resources == {"decoder": 1}
            ]

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


class ClipExtractorHub100(_EvalMissionBase):
    name: str = "clip_extractor_hub_100"
    description: str = "Extra large 100x100 extractor hub with clipped oxygen."
    map_name: str = "extractor_hub_100x100.map"
    clip_rate: float = 0.0

    def instantiate(
        self, map_builder: MapBuilderConfig, num_cogs: int, variant: MissionVariant | None = None
    ) -> "Mission":
        mission = super().instantiate(map_builder, num_cogs, variant)
        mission.oxygen_extractor.start_clipped = True

        def _filter_unclip(cfg: MettaGridConfig) -> None:
            if cfg.game.clipper is None:
                return
            cfg.game.clipper.unclipping_recipes = [
                r for r in cfg.game.clipper.unclipping_recipes if r.input_resources == {"decoder": 1}
            ]

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


# Register clipping missions
EVAL_MISSIONS.extend(
    [
        ClipCarbon,
        ClipOxygen,
        ClipExtractorHub30,
        ClipExtractorHub50,
        ClipExtractorHub70,
        ClipExtractorHub80,
        ClipExtractorHub100,
    ]
)
