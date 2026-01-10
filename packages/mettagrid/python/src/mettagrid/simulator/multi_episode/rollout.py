"""Result containers for multi-episode evaluation runs."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from mettagrid.types import EpisodeStats


class EpisodeRolloutResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    assignments: np.ndarray  # agent_id -> policy_idx
    rewards: np.ndarray  # agent_id -> reward
    action_timeouts: np.ndarray  # agent_id -> timeout_count
    stats: EpisodeStats
    replay_path: str | None
    steps: int
    max_steps: int


class MultiEpisodeRolloutResult(BaseModel):
    episodes: list[EpisodeRolloutResult]
