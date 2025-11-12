"""Evaluation helpers for CoGames policies."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import numpy as np
from pydantic import BaseModel, ConfigDict

from mettagrid import MettaGridConfig
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy import PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
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


def _compute_policy_agent_counts(num_agents: int, policy_specs: list[PolicySpec]) -> list[int]:
    total = sum(spec.proportion for spec in policy_specs)
    if total <= 0:
        raise ValueError("Total policy proportion must be positive.")
    fractions = [spec.proportion / total for spec in policy_specs]

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
    policy_specs: list[PolicySpec],
    episodes: int,
    seed: int = 0,
    progress_callback: ProgressCallback | None = None,
    **kwargs,
) -> MultiEpisodeRolloutResult:
    """
    Runs rollout for multiple episodes, randomizing agent assignments for each episode in proportions
    specified by the input policy specs.

    Returns per-episode rewards, stats, assignments, and action timeouts.
    """
    policy_instances = [
        initialize_or_load_policy(
            PolicyEnvInterface.from_mg_cfg(env_cfg),
            spec.policy_class_path,
            spec.policy_data_path,
        )
        for spec in policy_specs
    ]
    policy_counts = _compute_policy_agent_counts(env_cfg.game.num_agents, policy_specs)

    assignments = np.repeat(np.arange(len(policy_specs)), policy_counts)

    assert len(assignments) == env_cfg.game.num_agents

    per_episode_rewards: list[np.ndarray] = []
    per_episode_stats: list[EpisodeStatsT] = []
    per_episode_assignments: list[np.ndarray] = []
    per_episode_timeouts: list[np.ndarray] = []
    rng = np.random.default_rng(seed)
    for episode_idx in range(episodes):
        rng.shuffle(assignments)
        agent_policies = [
            policy_instances[assignments[agent_id]].agent_policy(agent_id)
            for agent_id in range(env_cfg.game.num_agents)
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
