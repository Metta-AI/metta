from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mettagrid.core import MettaGridCore


class ReplayWriter(ABC):
    """Abstract base class for replay writers.

    Provides an interface for logging game episodes and writing replay data.
    Implementations can write to different storage backends (S3, local files, etc.).
    """

    @abstractmethod
    def start_episode(self, episode_id: str, env: MettaGridCore) -> None:
        """Start recording a new episode.

        Args:
            episode_id: Unique identifier for the episode
            env: The MettaGrid environment instance
        """
        pass

    @abstractmethod
    def log_step(self, episode_id: str, actions: np.ndarray, rewards: np.ndarray) -> None:
        """Log a single step in an episode.

        Args:
            episode_id: Unique identifier for the episode
            actions: Array of actions taken by agents
            rewards: Array of rewards received by agents
        """
        pass

    @abstractmethod
    def write_replay(self, episode_id: str) -> str | None:
        """Write the replay data and return the URL/path to access it.

        Args:
            episode_id: Unique identifier for the episode

        Returns:
            URL or path to the written replay, or None if replay writing is disabled
        """
        pass
