"""Machina missions locked to trainer maps with one-use extractors."""

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

TRAINER_BIG_MAP_NAME = "machinatrainerbig.map"
TRAINER_SMALL_MAP_NAME = "machinatrainersmall.map"


def _single_use_extractor_kwargs() -> dict:
    """Return kwargs to configure all extractors as single-use (max_uses=1)."""
    return {
        "carbon_extractor": CarbonExtractorConfig(max_uses=1),
        "oxygen_extractor": OxygenExtractorConfig(max_uses=1),
        "germanium_extractor": GermaniumExtractorConfig(),
        "silicon_extractor": SiliconExtractorConfig(max_uses=1),
    }


class MachinaTrainerRulesVariant(MissionVariant):
    """Apply trainer-specific rules (e.g., max_steps=2000)."""

    name: str = "machinatrainer_rules"

    def modify_env(self, mission: Mission, env: MettaGridConfig) -> None:
        """Set max_steps to 2000 for trainer missions."""
        env.game.max_steps = 2000


class MachinaTrainerBigMapVariant(MissionVariant):
    """Swap in the fixed machinatrainerbig ASCII map."""

    name: str = "machinatrainerbig_map"
    map_name: str = TRAINER_BIG_MAP_NAME
    fixed_spawn_order: bool = True

    def modify_env(self, mission: Mission, env: MettaGridConfig) -> None:  # noqa: D401 - inherited docstring
        env.game.map_builder = get_map(self.map_name, fixed_spawn_order=self.fixed_spawn_order)


class MachinaTrainerSmallMapVariant(MissionVariant):
    """Swap in the fixed machinatrainersmall ASCII map."""

    name: str = "machinatrainersmall_map"
    map_name: str = TRAINER_SMALL_MAP_NAME
    fixed_spawn_order: bool = True

    def modify_env(self, mission: Mission, env: MettaGridConfig) -> None:  # noqa: D401 - inherited docstring
        env.game.map_builder = get_map(self.map_name, fixed_spawn_order=self.fixed_spawn_order)


MACHINA_TRAINER_MISSIONS: list[Mission] = [
    Mission(
        name="machinatrainerbig",
        description="Machinatrainerbig map with 60 cogs, 2000 steps, and single-use extractors.",
        site=MACHINA_1,
        num_cogs=60,
        variants=[MachinaTrainerBigMapVariant()],
        **_single_use_extractor_kwargs(),
    ),
    Mission(
        name="machinatrainersmall",
        description=(
            "Machinatrainersmall map (middle 25% of big trainer) with 24 cogs, 2000 steps, single-use extractors."
        ),
        site=MACHINA_1,
        num_cogs=24,  # Matches available spawn points in map
        variants=[MachinaTrainerSmallMapVariant(), MachinaTrainerRulesVariant()],
        **_single_use_extractor_kwargs(),
    ),
]


def get_config() -> MettaGridConfig:
    """Allow `cogames play -m path/to/machina_missions_trainer.py`."""

    return MACHINA_TRAINER_MISSIONS[0].make_env()


__all__ = [
    "MACHINA_TRAINER_MISSIONS",
    "get_config",
]
