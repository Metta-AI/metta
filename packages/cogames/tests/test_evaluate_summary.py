from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from cogames.evaluate import MissionEvaluationResult, _build_results_summary
from mettagrid.policy.policy import PolicySpec


def test_build_results_summary_multi_mission_policy_episode() -> None:
    policy_specs = [
        PolicySpec(policy_class_path="cogames.policy.MockPolicy", policy_data_path=None, proportion=1.0),
        PolicySpec(policy_class_path="cogames.policy.MockPolicy", policy_data_path=None, proportion=1.0),
    ]

    mission_one = MissionEvaluationResult(
        mission_name="mission_one",
        policy_counts=[2, 1],
        policy_names=["MockPolicy", "MockPolicy"],
        aggregated_policy_stats=[
            {"stat_a": 10.0, "stat_b": 6.0},
            {"stat_a": 9.0},
        ],
        aggregated_game_stats={"game_metric": 8.0, "failures": 2.0},
        per_episode_rewards=[
            np.array([2.0, 4.0, 3.0], dtype=float),
            np.array([1.0, 5.0, 6.0], dtype=float),
        ],
        per_episode_assignments=[
            np.array([0, 0, 1], dtype=int),
            np.array([0, 0, 1], dtype=int),
        ],
        per_policy_timeouts={0: 1, 1: 2},
        episodes=2,
    )

    mission_two = MissionEvaluationResult(
        mission_name="mission_two",
        policy_counts=[1, 2],
        policy_names=["MockPolicy", "MockPolicy"],
        aggregated_policy_stats=[
            {"stat_a": 12.0},
            {"stat_a": 18.0, "stat_b": 9.0},
        ],
        aggregated_game_stats={"game_metric": 18.0},
        per_episode_rewards=[
            np.array([10.0, 2.0, 4.0], dtype=float),
            np.array([8.0, 6.0, 2.0], dtype=float),
            np.array([3.0, 12.0, 6.0], dtype=float),
        ],
        per_episode_assignments=[
            np.array([0, 1, 1], dtype=int),
            np.array([1, 0, 1], dtype=int),
            np.array([1, 1, 0], dtype=int),
        ],
        per_policy_timeouts={0: 0, 1: 5},
        episodes=3,
    )

    summary = _build_results_summary(
        mission_results=[mission_one, mission_two],
        policy_specs=policy_specs,
    )

    assert isinstance(summary.generated_at, datetime)
    assert len(summary.missions) == 2

    mission_one_summary = summary.missions[0]
    assert mission_one_summary.mission_name == "mission_one"
    assert mission_one_summary.episodes == 2
    assert mission_one_summary.avg_game_stats == pytest.approx({"failures": 1.0, "game_metric": 4.0})

    r1 = mission_one_summary.per_episode_per_policy_avg_rewards
    assert len(r1) == 2
    assert r1[0] == pytest.approx([3.0, 3.0])
    assert r1[1] == pytest.approx([3.0, 6.0])

    assert len(mission_one_summary.policy_summaries) == 2
    policy_a, policy_b = mission_one_summary.policy_summaries
    assert policy_a.policy_name == "MockPolicy"
    assert policy_a.agent_count == 2
    assert policy_a.avg_agent_metrics == pytest.approx({"stat_a": 5.0, "stat_b": 3.0})
    assert policy_a.action_timeouts == 1

    assert policy_b.policy_name == "MockPolicy"
    assert policy_b.agent_count == 1
    assert policy_b.avg_agent_metrics == pytest.approx({"stat_a": 9.0})
    assert policy_b.action_timeouts == 2

    mission_two_summary = summary.missions[1]
    assert mission_two_summary.mission_name == "mission_two"
    assert mission_two_summary.episodes == 3
    assert mission_two_summary.avg_game_stats == pytest.approx({"game_metric": 6.0})

    r2 = mission_two_summary.per_episode_per_policy_avg_rewards
    assert len(r2) == 3
    assert r2[0] == pytest.approx([10.0, 3.0])
    assert r2[1] == pytest.approx([6.0, 5.0])
    assert r2[2] == pytest.approx([6.0, 7.5])

    policy_c, policy_d = mission_two_summary.policy_summaries
    assert policy_c.policy_name == "MockPolicy"
    assert policy_c.agent_count == 1
    assert policy_c.avg_agent_metrics == pytest.approx({"stat_a": 12.0})
    assert policy_c.action_timeouts == 0

    assert policy_d.policy_name == "MockPolicy"
    assert policy_d.agent_count == 2
    assert policy_d.avg_agent_metrics == pytest.approx({"stat_a": 9.0, "stat_b": 4.5})
    assert policy_d.action_timeouts == 5
