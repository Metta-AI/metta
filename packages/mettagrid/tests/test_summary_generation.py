import numpy as np
import pytest

from mettagrid.simulator.multi_episode.rollout import EpisodeRolloutResult, MultiEpisodeRolloutResult
from mettagrid.simulator.multi_episode.summary import build_multi_episode_rollout_summaries


def test_build_results_summary_multi_mission_policy_episode() -> None:
    mission_one = MultiEpisodeRolloutResult(
        episodes=[
            EpisodeRolloutResult(
                assignments=np.array([0, 0, 1], dtype=int),
                rewards=np.array([2.0, 4.0, 3.0], dtype=float),
                action_timeouts=np.array([1.0, 0.0, 1.0], dtype=float),
                stats={
                    "game": {"game_metric": 4.0, "failures": 1.0},
                    "agent": [{"stat_a": 2.0, "stat_b": 1.0}, {"stat_a": 2.0, "stat_b": 1.0}, {"stat_a": 4.0}],
                },
                replay_path=None,
                steps=82,
                max_steps=100,
            ),
            EpisodeRolloutResult(
                assignments=np.array([0, 0, 1], dtype=int),
                rewards=np.array([1.0, 5.0, 6.0], dtype=float),
                action_timeouts=np.array([0.0, 0.0, 1.0], dtype=float),
                stats={
                    "game": {"game_metric": 4.0, "failures": 1.0},
                    "agent": [{"stat_a": 3.0, "stat_b": 2.0}, {"stat_a": 3.0, "stat_b": 2.0}, {"stat_a": 5.0}],
                },
                replay_path=None,
                steps=97,
                max_steps=100,
            ),
        ]
    )

    mission_two = MultiEpisodeRolloutResult(
        episodes=[
            EpisodeRolloutResult(
                assignments=np.array([0, 1, 1], dtype=int),
                rewards=np.array([10.0, 2.0, 4.0], dtype=float),
                action_timeouts=np.array([0.0, 2.0, 1.0], dtype=float),
                stats={
                    "game": {"game_metric": 6.0},
                    "agent": [{"stat_a": 3.0}, {"stat_a": 2.0, "stat_b": 1.0}, {"stat_a": 4.0, "stat_b": 2.0}],
                },
                replay_path=None,
                steps=118,
                max_steps=150,
            ),
            EpisodeRolloutResult(
                assignments=np.array([1, 0, 1], dtype=int),
                rewards=np.array([8.0, 6.0, 2.0], dtype=float),
                action_timeouts=np.array([2.0, 0.0, 0.0], dtype=float),
                stats={
                    "game": {"game_metric": 6.0},
                    "agent": [{"stat_a": 3.0, "stat_b": 1.0}, {"stat_a": 4.0}, {"stat_a": 4.0, "stat_b": 2.0}],
                },
                replay_path=None,
                steps=105,
                max_steps=150,
            ),
            EpisodeRolloutResult(
                assignments=np.array([1, 1, 0], dtype=int),
                rewards=np.array([3.0, 12.0, 6.0], dtype=float),
                action_timeouts=np.array([0.0, 0.0, 0.0], dtype=float),
                stats={
                    "game": {"game_metric": 6.0},
                    "agent": [{"stat_a": 2.0, "stat_b": 1.0}, {"stat_a": 3.0, "stat_b": 2.0}, {"stat_a": 5.0}],
                },
                replay_path=None,
                steps=150,
                max_steps=150,
            ),
        ]
    )

    summary = build_multi_episode_rollout_summaries(rollout_results=[mission_one, mission_two], num_policies=2)

    assert len(summary) == 2

    mission_one_summary = summary[0]
    assert mission_one_summary.episodes == 2
    assert mission_one_summary.avg_game_stats == pytest.approx({"failures": 1.0, "game_metric": 4.0})

    r1 = mission_one_summary.per_episode_per_policy_avg_rewards
    assert len(r1) == 2
    assert r1[0] == pytest.approx([3.0, 3.0])
    assert r1[1] == pytest.approx([3.0, 6.0])

    assert len(mission_one_summary.policy_summaries) == 2
    policy_a, policy_b = mission_one_summary.policy_summaries
    assert policy_a.agent_count == 2
    assert policy_a.avg_agent_metrics == pytest.approx({"stat_a": 5.0, "stat_b": 3.0})
    assert policy_a.action_timeouts == 1

    assert policy_b.agent_count == 1
    assert policy_b.avg_agent_metrics == pytest.approx({"stat_a": 9.0})
    assert policy_b.action_timeouts == 2

    mission_two_summary = summary[1]

    assert mission_two_summary.episodes == 3
    assert mission_two_summary.avg_game_stats == pytest.approx({"game_metric": 6.0})

    r2 = mission_two_summary.per_episode_per_policy_avg_rewards
    assert len(r2) == 3
    assert r2[0] == pytest.approx([10.0, 3.0])
    assert r2[1] == pytest.approx([6.0, 5.0])
    assert r2[2] == pytest.approx([6.0, 7.5])

    policy_c, policy_d = mission_two_summary.policy_summaries
    assert policy_c.agent_count == 1
    assert policy_c.avg_agent_metrics == pytest.approx({"stat_a": 12.0})
    assert policy_c.action_timeouts == 0

    assert policy_d.agent_count == 2
    assert policy_d.avg_agent_metrics == pytest.approx({"stat_a": 9.0, "stat_b": 4.5})
    assert policy_d.action_timeouts == 5
