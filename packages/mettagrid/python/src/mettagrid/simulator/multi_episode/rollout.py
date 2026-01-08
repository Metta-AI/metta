"""Evaluation helpers for CoGames policies."""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from mettagrid import MettaGridConfig
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.renderer.renderer import RenderMode
from mettagrid.simulator import SimulatorEventHandler
from mettagrid.simulator.replay_log_writer import ReplayLogWriter
from mettagrid.simulator.rollout import Rollout
from mettagrid.types import EpisodeStats

_SKIP_STATS = [r"^action\.invalid_arg\..+$"]

ProgressCallback = Callable[[int], None]


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
    save_replay: Optional[str] = None,
    max_action_time_ms: int | None = None,
    render_mode: Optional[RenderMode] = None,
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

    episode_results: list[EpisodeRolloutResult] = []

    rng = np.random.default_rng(seed)
    for episode_idx in range(episodes):
        rng.shuffle(assignments)
        agent_policies: list[AgentPolicy] = [
            policies[assignments[agent_id]].agent_policy(agent_id) for agent_id in range(env_cfg.game.num_agents)
        ]

        # Create a new replay writer for each episode if save_replay is provided
        handlers: list[SimulatorEventHandler] = []
        episode_replay_writer = None
        if save_replay is not None:
            episode_replay_writer = ReplayLogWriter(save_replay)
            handlers.append(episode_replay_writer)

        rollout = Rollout(
            env_cfg,
            agent_policies,
            max_action_time_ms=max_action_time_ms,
            render_mode=render_mode,
            seed=seed + episode_idx,
            event_handlers=handlers,
        )

        rollout.run_until_done()

        replay_path = None
        if episode_replay_writer is not None:
            all_replay_paths = episode_replay_writer.get_written_replay_urls()
            replay_path = None if not all_replay_paths else list(all_replay_paths.values())[0]

        result = EpisodeRolloutResult(
            assignments=assignments.copy(),
            rewards=np.array(rollout._sim.episode_rewards, dtype=float),
            action_timeouts=np.array(rollout.timeout_counts, dtype=float),
            stats=rollout._sim.episode_stats,
            replay_path=replay_path,
            steps=rollout._sim.current_step,
            max_steps=rollout._sim.config.game.max_steps,
        )

        episode_results.append(result)

        if progress_callback is not None:
            progress_callback(episode_idx)

    return MultiEpisodeRolloutResult(episodes=episode_results)
