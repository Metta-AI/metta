from __future__ import annotations

from collections import defaultdict

import numpy as np
from pydantic import BaseModel

from mettagrid.simulator.multi_episode.rollout import MultiEpisodeRolloutResult


class MultiEpisodeRolloutPolicySummary(BaseModel):
    """Summary of a policy's performance on an single env."""

    # Number of agents assigned to this policy for this mission
    agent_count: int
    # Average metrics across agents assigned to this policy for this mission
    avg_agent_metrics: dict[str, float]
    # Number of action timeouts experienced for this policy for this mission
    action_timeouts: int


class MultiEpisodeRolloutSummary(BaseModel):
    """Summary of results across multiple episodes on one env"""

    # Total number of episodes simulated for this mission
    episodes: int
    # Summaries for each policy for this mission
    policy_summaries: list[MultiEpisodeRolloutPolicySummary]
    # Averaged game stats across all episodes for this mission
    avg_game_stats: dict[str, float]
    # per_episode_per_policy_avg_rewards[episode_idx][policy_idx] = \
    #     average reward per policy for this episode (or None if the policy had no agents in this episode)
    per_episode_per_policy_avg_rewards: dict[int, list[float | None]]


def build_multi_episode_rollout_summaries(
    rollout_results: list[MultiEpisodeRolloutResult],
    num_policies: int,
) -> list[MultiEpisodeRolloutSummary]:
    if not rollout_results:
        return []

    summaries: list[MultiEpisodeRolloutSummary] = []

    for mission_result in rollout_results:
        policy_counts = np.bincount(mission_result.episodes[0].assignments, minlength=num_policies)

        summed_game_stats: defaultdict[str, float] = defaultdict(float)
        summed_policy_stats: list[defaultdict[str, float]] = [defaultdict(float) for _ in range(num_policies)]

        for e in mission_result.episodes:
            game_stats = e.stats.get("game", {})
            for key, value in game_stats.items():
                summed_game_stats[key] += float(value)

            agent_stats_list = e.stats.get("agent", [])
            for agent_id, agent_stats in enumerate(agent_stats_list):
                if agent_id >= len(e.assignments):
                    continue
                policy_idx = int(e.assignments[agent_id])
                for key, value in agent_stats.items():
                    summed_policy_stats[policy_idx][key] += float(value)

        transpired_episodes = len(mission_result.episodes)
        if transpired_episodes:
            avg_game_stats = {key: value / transpired_episodes for key, value in summed_game_stats.items()}
        else:
            avg_game_stats = {}

        materialized_policy_stats = [dict(stats) for stats in summed_policy_stats]

        per_episode_per_policy_avg_rewards: dict[int, list[float | None]] = {}
        for episode_idx, e in enumerate(mission_result.episodes):
            per_policy_totals = np.zeros(num_policies, dtype=float)
            per_policy_counts = np.zeros(num_policies, dtype=int)
            for agent_id, reward in enumerate(e.rewards):
                if agent_id >= len(e.assignments):
                    continue
                policy_idx = int(e.assignments[agent_id])
                per_policy_totals[policy_idx] += float(reward)
                per_policy_counts[policy_idx] += 1
            per_episode_per_policy_avg_rewards[episode_idx] = [
                (per_policy_totals[i] / per_policy_counts[i]) if per_policy_counts[i] > 0 else None
                for i in range(num_policies)
            ]

        policy_summaries: list[MultiEpisodeRolloutPolicySummary] = []
        for policy_idx in range(num_policies):
            agent_count = int(policy_counts[policy_idx]) if policy_idx < len(policy_counts) else 0
            average_metrics = (
                {key: value / agent_count for key, value in sorted(materialized_policy_stats[policy_idx].items())}
                if agent_count > 0
                else {}
            )
            action_timeouts = 0
            for e in mission_result.episodes:
                for agent_index, timeout_count in enumerate(e.action_timeouts):
                    if agent_index >= len(e.assignments):
                        continue
                    if int(e.assignments[agent_index]) == policy_idx:
                        action_timeouts += int(timeout_count)

            policy_summaries.append(
                MultiEpisodeRolloutPolicySummary(
                    agent_count=agent_count,
                    avg_agent_metrics=average_metrics,
                    action_timeouts=action_timeouts,
                )
            )

        summaries.append(
            MultiEpisodeRolloutSummary(
                episodes=transpired_episodes,
                policy_summaries=policy_summaries,
                avg_game_stats=avg_game_stats,
                per_episode_per_policy_avg_rewards=per_episode_per_policy_avg_rewards,
            )
        )

    return summaries
