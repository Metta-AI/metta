from __future__ import annotations

from cogames.cogs_vs_clips.mission import Mission, MissionVariant, Site
from cogames.cogs_vs_clips.missions import _add_make_env_modifier, get_map
from mettagrid.config.mettagrid_config import MettaGridConfig
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

        # Post-make_env adjust max uses on built objects
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

        return _add_make_env_modifier(mission, _post)


class EnergyStarved(_EvalMissionBase):
    name = "energy_starved"
    description = "Very low energy regen; plan charger routes."
    map_name = "machina_eval_exp01.map"
    charger_eff = 80
    energy_regen = 0
    max_uses_charger = 0
    max_uses_carbon = 100
    max_uses_oxygen = 30
    max_uses_germanium = 5
    max_uses_silicon = 100


class OxygenBottleneck(_EvalMissionBase):
    name = "oxygen_bottleneck"
    description = "Oxygen paces assembly; batch other resources."
    map_name = "machina_eval_exp02.map"
    oxygen_eff = 50
    energy_regen = 1
    max_uses_charger = 0
    max_uses_carbon = 100
    max_uses_oxygen = 20
    max_uses_germanium = 10
    max_uses_silicon = 100


class GermaniumRush(_EvalMissionBase):
    name = "germanium_rush"
    description = "Race to limited germanium before it runs out."
    map_name = "machina_eval_exp03.map"
    max_uses_germanium = 10
    max_uses_carbon = 100
    max_uses_oxygen = 50
    max_uses_silicon = 100
    max_uses_charger = 0


class SiliconWorkbench(_EvalMissionBase):
    name = "silicon_workbench"
    description = "Silicon-rich environment; convert energy to silicon."
    map_name = "machina_eval_exp04.map"
    silicon_eff = 150
    max_uses_silicon = 200
    max_uses_oxygen = 50
    max_uses_carbon = 100
    max_uses_germanium = 10
    max_uses_charger = 0


class CarbonDesert(_EvalMissionBase):
    name = "carbon_desert"
    description = "Sparse carbon dictates routes."
    map_name = "machina_eval_exp05.map"
    max_uses_carbon = 30
    max_uses_oxygen = 50
    max_uses_germanium = 10
    max_uses_silicon = 100
    max_uses_charger = 0


class SingleUseWorld(_EvalMissionBase):
    name = "single_use_world"
    description = "Every station can be used exactly once."
    map_name = "machina_eval_exp06.map"
    max_uses_charger = 1
    max_uses_carbon = 1
    max_uses_oxygen = 1
    max_uses_germanium = 1
    max_uses_silicon = 1


class SlowOxygen(_EvalMissionBase):
    name = "slow_oxygen"
    description = "Very slow oxygen; interleave partial-usage taps."
    map_name = "machina_eval_exp07.map"
    oxygen_eff = 25
    energy_regen = 2
    max_uses_oxygen = 100
    max_uses_carbon = 100
    max_uses_germanium = 10
    max_uses_silicon = 100
    max_uses_charger = 0


class HighRegenSprint(_EvalMissionBase):
    name = "high_regen_sprint"
    description = "High regen; minimize charger dependency."
    map_name = "machina_eval_exp08.map"
    energy_regen = 3
    max_uses_carbon = 100
    max_uses_oxygen = 50
    max_uses_germanium = 10
    max_uses_silicon = 100
    max_uses_charger = 0


class SparseBalanced(_EvalMissionBase):
    name = "sparse_balanced"
    description = "Evenly sparse resources; balanced routing."
    map_name = "machina_eval_exp09.map"
    max_uses_carbon = 50
    max_uses_oxygen = 50
    max_uses_germanium = 10
    max_uses_silicon = 50
    max_uses_charger = 0


class GermaniumClutch(_EvalMissionBase):
    name = "germanium_clutch"
    description = "A single germanium line determines success."
    map_name = "machina_eval_exp10.map"
    max_uses_germanium = 2
    max_uses_carbon = 100
    max_uses_oxygen = 50
    max_uses_silicon = 100
    max_uses_charger = 0


EVAL_MISSIONS: list[type[Mission]] = [
    EnergyStarved,
    OxygenBottleneck,
    GermaniumRush,
    SiliconWorkbench,
    CarbonDesert,
    SingleUseWorld,
    SlowOxygen,
    HighRegenSprint,
    SparseBalanced,
    GermaniumClutch,
]


