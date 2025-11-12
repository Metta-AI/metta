"""Evaluation helpers for CoGames policies."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import numpy as np
from pydantic import BaseModel, ConfigDict

from mettagrid import MettaGridConfig
from mettagrid.policy.policy import AgentPolicy, MultiAgentPolicy
from mettagrid.simulator.rollout import Rollout

if TYPE_CHECKING:
    from mettagrid.mettagrid_c import EpisodeStats

    EpisodeStatsT = EpisodeStats
else:
    EpisodeStatsT = dict

_SKIP_STATS = [r"^action\.invalid_arg\..+$"]

ProgressCallback = Callable[[int], None]


class MultiEpisodeRolloutResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    assignments: list[np.ndarray]
    rewards: list[np.ndarray]
    action_timeouts: list[np.ndarray]
    stats: list[EpisodeStatsT]


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
    proportions: list[float] | None = None,
    progress_callback: ProgressCallback | None = None,
    **kwargs,
) -> MultiEpisodeRolloutResult:
    """
    Runs rollout for multiple episodes, randomizing agent assignments for each episode in proportions
    specified by the input policy specs (default uniform).

    Returns per-episode rewards, stats, assignments, and action timeouts.
    """
    if proportions is not None and len(proportions) != len(policies):
        raise ValueError("Number of proportions must match number of policies.")
    policy_counts = _compute_policy_agent_counts(
        env_cfg.game.num_agents, proportions if proportions is not None else [1.0] * len(policies)
    )
    assignments = np.repeat(np.arange(len(policies)), policy_counts)

    per_episode_rewards: list[np.ndarray] = []
    per_episode_stats: list[EpisodeStatsT] = []
    per_episode_assignments: list[np.ndarray] = []
    per_episode_timeouts: list[np.ndarray] = []
    rng = np.random.default_rng(seed)
    for episode_idx in range(episodes):
        rng.shuffle(assignments)
        agent_policies: list[AgentPolicy] = [
            policies[assignments[agent_id]].agent_policy(agent_id) for agent_id in range(env_cfg.game.num_agents)
        ]

        rollout = Rollout(env_cfg, agent_policies, **kwargs)

        rollout.run_until_done()

        per_episode_rewards.append(np.array(rollout._sim.episode_rewards, dtype=float))
        per_episode_stats.append(rollout._sim.episode_stats)
        per_episode_timeouts.append(np.array(rollout.timeout_counts, dtype=float))
        per_episode_assignments.append(assignments.copy())

        if progress_callback is not None:
            progress_callback(episode_idx)

    return MultiEpisodeRolloutResult(
        rewards=per_episode_rewards,
        stats=per_episode_stats,
        action_timeouts=per_episode_timeouts,
        assignments=per_episode_assignments,
    )
