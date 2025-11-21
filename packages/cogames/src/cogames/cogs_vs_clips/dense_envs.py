"""Dense training missions for curriculum learning.
"""

from __future__ import annotations

from cogames.cogs_vs_clips.mission import Mission, MissionVariant
from cogames.cogs_vs_clips.mission_utils import get_map
from cogames.cogs_vs_clips.sites import MACHINA_1
from mettagrid.config.mettagrid_config import MettaGridConfig


class MapVariant(MissionVariant):
    """Swap in a specific ASCII map."""

    name: str = "map_variant"
    map_name: str
    fixed_spawn_order: bool = True

    def modify_env(self, mission: Mission, env: MettaGridConfig) -> None:
        env.game.map_builder = get_map(self.map_name, fixed_spawn_order=self.fixed_spawn_order)


class MaxStepsVariant(MissionVariant):
    """Set the maximum number of steps for an episode."""

    name: str = "max_steps"
    max_steps: int

    def modify_env(self, mission: Mission, env: MettaGridConfig) -> None:
        env.game.max_steps = self.max_steps


DenseTraining4Agents = Mission(
    name="dense_training_4agents",
    description="Machinatrainer4agents map.",
    site=MACHINA_1,
    variants=[
        MapVariant(map_name="machinatrainer4agents.map"),
        MaxStepsVariant(max_steps=1400),
    ],
)

DenseTraining4AgentsBase = Mission(
    name="dense_training_4agentsbase",
    description="Machinatrainer4agentsbase map.",
    site=MACHINA_1,
    variants=[
        MapVariant(map_name="machinatrainer4agentsbase.map"),
        MaxStepsVariant(max_steps=1400),
    ],
)

DenseTrainingBig = Mission(
    name="dense_training_big",
    description="Machinatrainerbig map.",
    site=MACHINA_1,
    variants=[
        MapVariant(map_name="machinatrainerbig.map"),
        MaxStepsVariant(max_steps=1400),
    ],
)

DenseTrainingSmall = Mission(
    name="dense_training_small",
    description="Machinatrainersmall map.",
    site=MACHINA_1,
    variants=[
        MapVariant(map_name="machinatrainersmall.map"),
        MaxStepsVariant(max_steps=1400),
    ],
)

DENSE_TRAINING_MISSIONS: list[Mission] = [
    DenseTraining4Agents,
    DenseTraining4AgentsBase,
    DenseTrainingBig,
    DenseTrainingSmall,
]


def get_config() -> MettaGridConfig:
    """Allow `cogames play -m path/to/dense_envs.py`."""
    return DENSE_TRAINING_MISSIONS[0].make_env()


__all__ = [
    "DENSE_TRAINING_MISSIONS",
    "DenseTraining4Agents",
    "DenseTraining4AgentsBase",
    "DenseTrainingBig",
    "DenseTrainingSmall",
    "get_config",
    "MapVariant",
    "MaxStepsVariant",
]
