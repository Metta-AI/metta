"""Evaluation helpers for CoGames policies."""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _run_single_episode_rollout(
    env_cfg: MettaGridConfig,
    policies: list[MultiAgentPolicy],
    assignments: np.ndarray,
    episode_seed: int,
    max_action_time_ms: int | None,
    save_replay: Optional[Path],
    event_handlers: Optional[list[SimulatorEventHandler]],
) -> tuple[np.ndarray, EpisodeStats, np.ndarray, np.ndarray, list[str]]:
    """Run a single episode rollout and return its results.

    This is a helper function that can be used by both serial and parallel execution.
    """
    agent_policies: list[AgentPolicy] = [
        policies[assignments[agent_id]].agent_policy(agent_id) for agent_id in range(env_cfg.game.num_agents)
    ]
    handlers = list(event_handlers or [])

    # Create a new replay writer for this episode if save_replay is provided
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

    rewards = np.array(rollout._sim.episode_rewards, dtype=float)
    stats = rollout._sim.episode_stats
    timeouts = np.array(rollout.timeout_counts, dtype=float)

    replay_paths: list[str] = []
    if episode_replay_writer is not None:
        replay_paths.extend(episode_replay_writer.get_written_replay_paths())

    return rewards, stats, timeouts, assignments.copy(), replay_paths


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
        rewards, stats, timeouts, episode_assignments, replay_paths = _run_single_episode_rollout(
            env_cfg,
            policies,
            assignments,
            seed + episode_idx,  # Deterministic seed per episode
            max_action_time_ms,
            save_replay,
            event_handlers,
        )

        per_episode_rewards.append(rewards)
        per_episode_stats.append(stats)
        per_episode_timeouts.append(timeouts)
        per_episode_assignments.append(episode_assignments)
        all_replay_paths.extend(replay_paths)

        if progress_callback is not None:
            progress_callback(episode_idx)

    return MultiEpisodeRolloutResult(
        rewards=per_episode_rewards,
        stats=per_episode_stats,
        action_timeouts=per_episode_timeouts,
        assignments=per_episode_assignments,
        replay_paths=all_replay_paths,
    )


class ParallelRollout:
    """Parallel episode execution wrapper for multi-episode rollouts.

    This class provides the same interface as `multi_episode_rollout()` but executes
    episodes in parallel using a thread pool. It's a drop-in replacement that can
    significantly speed up evaluation when multiple CPU cores are available.

    Example:
        ```python
        # Serial execution
        result = multi_episode_rollout(env_cfg, policies, episodes=100)

        # Parallel execution (drop-in replacement)
        parallel_rollout = ParallelRollout(max_workers=8)
        result = parallel_rollout(
            env_cfg, policies, episodes=100,
            seed=42, proportions=[1.0], max_action_time_ms=250
        )
        ```
    """

    def __init__(self, max_workers: Optional[int] = None):
        """Initialize parallel rollout executor.

        Args:
            max_workers: Maximum number of parallel workers. If None, defaults to
                CPU count. If 1, runs serially (useful for debugging).
        """
        self.max_workers = max_workers

    def __call__(
        self,
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
        """Run multiple episodes in parallel.

        This method has the same signature and return type as `multi_episode_rollout()`,
        making it a drop-in replacement for parallel execution.

        Args:
            env_cfg: Environment configuration
            policies: List of MultiAgentPolicy instances
            episodes: Number of episodes to run
            seed: Base random seed (each episode gets a deterministic seed)
            proportions: Optional policy proportions for agent assignment
            progress_callback: Optional callback for progress updates
            save_replay: Optional directory to save replay files
            max_action_time_ms: Maximum action generation time in milliseconds
            event_handlers: Optional list of event handlers

        Returns:
            MultiEpisodeRolloutResult with per-episode results
        """
        if proportions is not None and len(proportions) != len(policies):
            raise ValueError("Number of proportions must match number of policies.")

        policy_counts = _compute_policy_agent_counts(
            env_cfg.game.num_agents, list(proportions) if proportions is not None else [1.0] * len(policies)
        )
        base_assignments = np.repeat(np.arange(len(policies)), policy_counts)

        # Pre-compute all episode seeds and assignments for determinism
        seed_rng = np.random.default_rng(seed)
        episode_seeds = [int(seed_rng.integers(0, 2**31)) for _ in range(episodes)]

        # Pre-shuffle assignments for each episode
        episode_assignments: list[np.ndarray] = []
        for episode_idx in range(episodes):
            assignments = base_assignments.copy()
            episode_rng = np.random.default_rng(episode_seeds[episode_idx])
            episode_rng.shuffle(assignments)
            episode_assignments.append(assignments)

        # Initialize result arrays
        per_episode_rewards: list[np.ndarray | None] = [None] * episodes
        per_episode_stats: list[EpisodeStats | None] = [None] * episodes
        per_episode_assignments: list[np.ndarray | None] = [None] * episodes
        per_episode_timeouts: list[np.ndarray | None] = [None] * episodes
        all_replay_paths: list[str] = []

        # Determine number of workers
        max_workers = self.max_workers if self.max_workers is not None else max(1, os.cpu_count() or 1)
        use_parallel = max_workers > 1 and episodes > 1

        if use_parallel:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _run_single_episode_rollout,
                        env_cfg,
                        policies,
                        episode_assignments[episode_idx],
                        episode_seeds[episode_idx],
                        max_action_time_ms,
                        save_replay,
                        event_handlers,
                    ): episode_idx
                    for episode_idx in range(episodes)
                }

                for future in as_completed(futures):
                    episode_idx = futures[future]
                    rewards, stats, timeouts, assignments, replay_paths = future.result()

                    per_episode_rewards[episode_idx] = rewards
                    per_episode_stats[episode_idx] = stats
                    per_episode_timeouts[episode_idx] = timeouts
                    per_episode_assignments[episode_idx] = assignments
                    all_replay_paths.extend(replay_paths)

                    if progress_callback is not None:
                        progress_callback(episode_idx)
        else:
            # Serial execution (fallback for single worker or single episode)
            for episode_idx in range(episodes):
                rewards, stats, timeouts, assignments, replay_paths = _run_single_episode_rollout(
                    env_cfg,
                    policies,
                    episode_assignments[episode_idx],
                    episode_seeds[episode_idx],
                    max_action_time_ms,
                    save_replay,
                    event_handlers,
                )

                per_episode_rewards[episode_idx] = rewards
                per_episode_stats[episode_idx] = stats
                per_episode_timeouts[episode_idx] = timeouts
                per_episode_assignments[episode_idx] = assignments
                all_replay_paths.extend(replay_paths)

                if progress_callback is not None:
                    progress_callback(episode_idx)

        return MultiEpisodeRolloutResult(
            rewards=per_episode_rewards,  # type: ignore[arg-type]
            stats=per_episode_stats,  # type: ignore[arg-type]
            action_timeouts=per_episode_timeouts,  # type: ignore[arg-type]
            assignments=per_episode_assignments,  # type: ignore[arg-type]
            replay_paths=all_replay_paths,
        )
