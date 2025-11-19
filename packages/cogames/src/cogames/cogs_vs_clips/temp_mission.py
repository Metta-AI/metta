"""Temporary mission for playing machinatrainerbig with resource reduction."""

from __future__ import annotations

from cogames.cogs_vs_clips.mission import Mission, MissionVariant
from cogames.cogs_vs_clips.sites import MACHINA_1
from cogames.map_utils.resource_reducer import reduce_map_resources
from mettagrid.config.mettagrid_config import MettaGridConfig


class ModifiedMapVariant(MissionVariant):
    """Apply resource reduction to the map."""

    name: str = "modified_map"

    def modify_env(self, mission: Mission, env: MettaGridConfig) -> None:
        # Apply resource reduction
        resource_levels = {
            "C": 5,  # 50% kept
            "O": 4,  # 40% kept
            "G": 2,  # 20% kept
            "S": 10,  # 100% kept
            "&": 10,  # normal
            "+": 10,  # normal
            "=": 10,  # normal
        }
        modified_map = reduce_map_resources(
            "machinatrainerbig.map",
            resource_levels=resource_levels,
            seed=42,
        )
        env.game.map_builder = modified_map


MachinaTrainerBigReduced = Mission(
    name="machinatrainerbig_reduced",
    description="Machinatrainerbig with reduced resources (C=5, O=4, G=2, S=10)",
    site=MACHINA_1,
    variants=[ModifiedMapVariant()],
)

