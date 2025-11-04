"""
StatsWriter is an abstract interface for writing statistics in MettaGrid.
It is used to record the outcomes of episodes.
"""

import datetime
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict


class StatsWriter(ABC):
    """
    Abstract interface for tracking statistics in MettaGrid; can be used by multiple environments simultaneously.
    """

    def __init__(self, dir: Path) -> None:
        self.dir = dir

    @abstractmethod
    def record_episode(
        self,
        attributes: Dict[str, str],
        agent_metrics: Dict[int, Dict[str, float]],
        agent_groups: Dict[int, int],
        step_count: int,
        replay_url: str | None,
        created_at: datetime.datetime,
    ) -> int:
        """Record episode statistics.

        Args:
            attributes: Episode attributes/metadata as key-value pairs
            agent_metrics: Per-agent metrics as {agent_id: {metric_name: value}}
            agent_groups: Agent group assignments as {agent_id: group_id}
            step_count: Number of steps in the episode
            replay_url: Optional URL to the episode replay
            created_at: When the episode was created

        Returns:
            The episode ID
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close any open connections or resources."""
        raise NotImplementedError
