"""Evaluation helpers for CoGames policies."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from mettagrid import MettaGridConfig
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.simulator import SimulatorEventHandler
from mettagrid.simulator.replay_log_writer import ReplayLogWriter
from mettagrid.simulator.rollout import Rollout
from mettagrid.types import EpisodeStats

_SKIP_STATS = [r"^action\.invalid_arg\..+$"]

ProgressCallback = Callable[[int], None]


class MultiEpisodeRolloutResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    assignments: list[np.ndarray]
    rewards: list[np.ndarray]
    action_timeouts: list[np.ndarray]
    stats: list[EpisodeStats]
    replay_paths: list[str] = Field(default_factory=list)


def _compute_policy_agent_counts(num_agents: int, proportions: list[float]) -> list[int]:
    total = sum(proportions)
    if total <= 0:
        raise ValueError("Total policy proportion must be positive.")
    fractions = [proportion / total for proportion in proportions]

    ideals = [num_agents * f for f in fractions]
    counts = [math.floor(x) for x in ideals]
    remaining = num_agents - sum(counts)

    # distribute by largest remainder
    remainders = [(i, ideals[i] - counts[i]) for i in range(len(fractions))]
    remainders.sort(key=lambda x: x[1], reverse=True)
    for i in range(remaining):
        counts[remainders[i][0]] += 1
    return counts


def multi_episode_rollout(
    env_cfg: MettaGridConfig,
    policies: list[MultiAgentPolicy],
    episodes: int,
    seed: int = 0,
    proportions: Optional[Sequence[float]] = None,
    progress_callback: Optional[ProgressCallback] = None,
    save_replay: Optional[Path] = None,
    max_action_time_ms: int | None = None,
    event_handlers: Optional[list[SimulatorEventHandler]] = None,
) -> MultiEpisodeRolloutResult:
    """
    Runs rollout for multiple episodes, randomizing agent assignments for each episode in proportions
    specified by the input policy specs (default uniform).

    Returns per-episode rewards, stats, assignments, and action timeouts.

    The same event handlers (if provided via event_handlers or save_replay) are reused for all episodes.
    When save_replay is provided, a new ReplayLogWriter is created for each episode to ensure
    each replay is saved with a unique filename.

    Args:
        save_replay: Optional directory path to save replays. If provided, creates ReplayLogWriter event handlers.
            Directory will be created if it doesn't exist. Each episode will be saved with a unique UUID-based filename.
    """
    if proportions is not None and len(proportions) != len(policies):
        raise ValueError("Number of proportions must match number of policies.")

    policy_counts = _compute_policy_agent_counts(
        env_cfg.game.num_agents, list(proportions) if proportions is not None else [1.0] * len(policies)
    )
    assignments = np.repeat(np.arange(len(policies)), policy_counts)

    per_episode_rewards: list[np.ndarray] = []
    per_episode_stats: list[EpisodeStats] = []
    per_episode_assignments: list[np.ndarray] = []
    per_episode_timeouts: list[np.ndarray] = []
    all_replay_paths: list[str] = []

    rng = np.random.default_rng(seed)
    for episode_idx in range(episodes):
        rng.shuffle(assignments)
        agent_policies: list[AgentPolicy] = [
            policies[assignments[agent_id]].agent_policy(agent_id) for agent_id in range(env_cfg.game.num_agents)
        ]
        handlers = list(event_handlers or [])

        # Create a new replay writer for each episode if save_replay is provided
        episode_replay_writer = None
        if save_replay is not None:
            episode_replay_writer = ReplayLogWriter(str(save_replay))
            handlers.append(episode_replay_writer)

        rollout = Rollout(
            env_cfg,
            agent_policies,
            max_action_time_ms=max_action_time_ms,
            event_handlers=handlers,
        )

        rollout.run_until_done()

        per_episode_rewards.append(np.array(rollout._sim.episode_rewards, dtype=float))
        per_episode_stats.append(rollout._sim.episode_stats)
        per_episode_timeouts.append(np.array(rollout.timeout_counts, dtype=float))
        per_episode_assignments.append(assignments.copy())

        # Collect replay paths from this episode's writer
        if episode_replay_writer is not None:
            all_replay_paths.extend(episode_replay_writer.get_written_replay_paths())

        if progress_callback is not None:
            progress_callback(episode_idx)

    return MultiEpisodeRolloutResult(
        rewards=per_episode_rewards,
        stats=per_episode_stats,
        action_timeouts=per_episode_timeouts,
        assignments=per_episode_assignments,
        replay_paths=all_replay_paths,
    )
