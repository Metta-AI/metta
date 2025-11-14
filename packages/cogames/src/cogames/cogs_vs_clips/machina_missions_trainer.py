"""Single Machina mission locked to machinatrainer4 with one-use extractors."""

from __future__ import annotations

from cogames.cogs_vs_clips.mission import Mission, MissionVariant
from cogames.cogs_vs_clips.mission_utils import get_map
from cogames.cogs_vs_clips.sites import MACHINA_1
from cogames.cogs_vs_clips.stations import (
    CarbonExtractorConfig,
    GermaniumExtractorConfig,
    OxygenExtractorConfig,
    SiliconExtractorConfig,
)
from mettagrid.config.mettagrid_config import MettaGridConfig

TRAINER_MAP_NAME = "evals/machinatrainer4.map"


class MachinaTrainerMapVariant(MissionVariant):
    """Swap in the fixed machinatrainer4 ASCII map."""

    name: str = "machina_trainer_map"
    map_name: str
    fixed_spawn_order: bool

    def __init__(self, map_name: str = TRAINER_MAP_NAME, fixed_spawn_order: bool = True):
        self.map_name = map_name
        self.fixed_spawn_order = fixed_spawn_order

    def modify_env(self, mission: Mission, env: MettaGridConfig) -> None:  # noqa: D401 - inherited docstring
        env.game.map_builder = get_map(self.map_name, fixed_spawn_order=self.fixed_spawn_order)


class MachinaTrainerRulesVariant(MissionVariant):
    """Tune episode length without changing other site defaults."""

    name: str = "machina_trainer_rules"
    description: str = "Fix Machina trainer runs to 2000 steps."
    max_steps: int = 2000

    def modify_env(self, mission: Mission, env: MettaGridConfig) -> None:  # noqa: D401 - inherited docstring
        env.game.max_steps = self.max_steps


def _single_use_extractor_kwargs() -> dict[str, object]:
    """Return station kwargs enforcing single-use extractors."""

    return {
        "carbon_extractor": CarbonExtractorConfig(max_uses=1),
        "oxygen_extractor": OxygenExtractorConfig(max_uses=1),
        "germanium_extractor": GermaniumExtractorConfig(max_uses=1),
        "silicon_extractor": SiliconExtractorConfig(max_uses=1),
    }


MACHINA_TRAINER_MISSIONS: list[Mission] = [
    Mission(
        name="machina_single_use",
        description="Machinatrainer4 map with 10 cogs, 2000 steps, and single-use extractors.",
        site=MACHINA_1,
        num_cogs=10,
        variants=[MachinaTrainerMapVariant(), MachinaTrainerRulesVariant()],
        **_single_use_extractor_kwargs(),
    )
]


def get_config() -> MettaGridConfig:
    """Allow `cogames play -m path/to/machina_missions_trainer.py`."""

    return MACHINA_TRAINER_MISSIONS[0].make_env()


__all__ = [
    "MACHINA_TRAINER_MISSIONS",
    "get_config",
]
