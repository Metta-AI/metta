"""George training missions with single-use resources."""

from __future__ import annotations

from cogames.cogs_vs_clips.mission import Mission, MissionVariant
from cogames.cogs_vs_clips.mission_utils import get_map
from cogames.cogs_vs_clips.sites import MACHINA_1
from cogames.cogs_vs_clips.stations import (
    AssemblerConfig,
    CarbonExtractorConfig,
    OxygenExtractorConfig,
    SiliconExtractorConfig,
)
from mettagrid.config.mettagrid_config import MettaGridConfig


def _single_use_extractor_kwargs() -> dict:
    """Return kwargs to configure extractors as single-use (max_uses=1).

    Note: GermaniumExtractorConfig is already single-use by default.
    """
    return {
        "carbon_extractor": CarbonExtractorConfig(max_uses=1),
        "oxygen_extractor": OxygenExtractorConfig(max_uses=1),
        "silicon_extractor": SiliconExtractorConfig(max_uses=1),
    }


class MapVariant(MissionVariant):
    """Swap in a specific ASCII map."""

    name: str = "map_variant"
    map_name: str
    fixed_spawn_order: bool = True

    def modify_env(self, mission: Mission, env: MettaGridConfig) -> None:
        env.game.map_builder = get_map(self.map_name, fixed_spawn_order=self.fixed_spawn_order)


class SingleUseChargerVariant(MissionVariant):
    """Set charger to single-use (max_uses=1)."""

    name: str = "single_use_charger"

    def modify_env(self, mission: Mission, env: MettaGridConfig) -> None:
        charger = env.game.objects.get("charger")
        if isinstance(charger, AssemblerConfig):
            charger.max_uses = 1


class MaxStepsVariant(MissionVariant):
    """Set the maximum number of steps for an episode."""

    name: str = "max_steps"
    max_steps: int

    def modify_env(self, mission: Mission, env: MettaGridConfig) -> None:
        env.game.max_steps = self.max_steps


GeorgeTraining4Agents = Mission(
    name="george_training_4agents",
    description="Machinatrainer4agents map with single-use resources.",
    site=MACHINA_1,
    variants=[
        MapVariant(map_name="machinatrainer4agents.map"),
        SingleUseChargerVariant(),
        MaxStepsVariant(max_steps=1400),
    ],
    **_single_use_extractor_kwargs(),
)

GeorgeTraining4AgentsBase = Mission(
    name="george_training_4agentsbase",
    description="Machinatrainer4agentsbase map with single-use resources.",
    site=MACHINA_1,
    variants=[
        MapVariant(map_name="machinatrainer4agentsbase.map"),
        SingleUseChargerVariant(),
        MaxStepsVariant(max_steps=1400),
    ],
    **_single_use_extractor_kwargs(),
)

GeorgeTrainingBig = Mission(
    name="george_training_big",
    description="Machinatrainerbig map with single-use resources.",
    site=MACHINA_1,
    variants=[
        MapVariant(map_name="machinatrainerbig.map"),
        SingleUseChargerVariant(),
        MaxStepsVariant(max_steps=1400),
    ],
    **_single_use_extractor_kwargs(),
)

GeorgeTrainingSmall = Mission(
    name="george_training_small",
    description="Machinatrainersmall map with single-use resources.",
    site=MACHINA_1,
    variants=[
        MapVariant(map_name="machinatrainersmall.map"),
        SingleUseChargerVariant(),
        MaxStepsVariant(max_steps=1400),
    ],
    **_single_use_extractor_kwargs(),
)

GEORGE_TRAINING_MISSIONS: list[Mission] = [
    GeorgeTraining4Agents,
    GeorgeTraining4AgentsBase,
    GeorgeTrainingBig,
    GeorgeTrainingSmall,
]


def get_config() -> MettaGridConfig:
    """Allow `cogames play -m path/to/george_training.py`."""
    return GEORGE_TRAINING_MISSIONS[0].make_env()


__all__ = [
    "GEORGE_TRAINING_MISSIONS",
    "GeorgeTraining4Agents",
    "GeorgeTraining4AgentsBase",
    "GeorgeTrainingBig",
    "GeorgeTrainingSmall",
    "get_config",
]
