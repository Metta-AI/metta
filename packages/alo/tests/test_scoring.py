from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import numpy as np
import pytest

from alo.scoring import (
    VorScenarioSummary,
    VorTotals,
    compute_weighted_scores,
    overall_value_over_replacement,
    summarize_vor_scenario,
    value_over_replacement,
)
from mettagrid.simulator.multi_episode.rollout import EpisodeRolloutResult, MultiEpisodeRolloutResult

EMPTY_STATS: dict[str, object] = {"game": {}, "agent": []}


@dataclass
class DummyMatch:
    assignments: list[int]
    policy_version_ids: list[UUID]
    policy_scores: dict[UUID, float]


def _episode(assignments: list[int], rewards: list[float]) -> EpisodeRolloutResult:
    return EpisodeRolloutResult(
        assignments=np.array(assignments, dtype=int),
        rewards=np.array(rewards, dtype=float),
        action_timeouts=np.zeros(len(rewards), dtype=float),
        stats=EMPTY_STATS,
        replay_path=None,
        steps=3,
        max_steps=10,
    )


def test_compute_weighted_scores() -> None:
    policy_a = UUID(int=1)
    policy_b = UUID(int=2)
    matches = [
        DummyMatch(
            assignments=[0, 0, 1],
            policy_version_ids=[policy_a, policy_b],
            policy_scores={policy_a: 1.0, policy_b: 3.0},
        ),
        DummyMatch(
            assignments=[1, 1, 1],
            policy_version_ids=[policy_a, policy_b],
            policy_scores={policy_a: 5.0, policy_b: 1.0},
        ),
    ]

    scores = compute_weighted_scores([policy_a, policy_b], matches)

    assert scores[policy_a] == pytest.approx(1.0)
    assert scores[policy_b] == pytest.approx(1.5)


def test_value_over_replacement() -> None:
    assert value_over_replacement(3.5, 2.0) == 1.5


def test_overall_value_over_replacement() -> None:
    assert overall_value_over_replacement(10.0, 5, 1.0) == pytest.approx(1.0)
    assert overall_value_over_replacement(10.0, 0, 1.0) is None


def test_summarize_vor_scenario_candidate() -> None:
    rollout = MultiEpisodeRolloutResult(
        episodes=[
            _episode([0, 1], [1.0, 3.0]),
            _episode([1, 1], [4.0, 5.0]),
        ]
    )

    summary = summarize_vor_scenario(rollout, candidate_policy_index=0, candidate_count=1)

    assert summary.candidate_mean == pytest.approx(1.0)
    assert summary.candidate_episode_count == 1
    assert summary.replacement_mean is None


def test_summarize_vor_scenario_replacement() -> None:
    rollout = MultiEpisodeRolloutResult(
        episodes=[
            _episode([0, 1], [1.0, 3.0]),
            _episode([0, 1], [2.0, 2.0]),
        ]
    )

    summary = summarize_vor_scenario(rollout, candidate_policy_index=0, candidate_count=0)

    assert summary.candidate_mean is None
    assert summary.replacement_mean == pytest.approx(2.0)


def test_vor_totals_update() -> None:
    totals = VorTotals()
    summary = VorScenarioSummary(candidate_mean=1.5, replacement_mean=None, candidate_episode_count=2)

    totals.update(2, summary)
    totals.update(0, VorScenarioSummary(candidate_mean=None, replacement_mean=2.0, candidate_episode_count=0))

    assert totals.total_candidate_weighted_sum == pytest.approx(6.0)
    assert totals.total_candidate_agents == 4
    assert totals.replacement_mean == pytest.approx(2.0)
