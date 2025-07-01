"""
Integration tests for metta.eval.eval_stats_db.EvalStatsDB.

They confirm correct normalization when metrics are recorded only
for non‑zero values.
"""

from __future__ import annotations

import datetime
import tempfile
import uuid
from pathlib import Path
from typing import List, Tuple

import pytest

from metta.eval.eval_stats_db import EvalStatsDB
from tests.fixtures import MockPolicyRecord


def _create_test_db_with_missing_metrics(db_path: Path) -> Tuple[EvalStatsDB, List[str], str]:
    db = EvalStatsDB(db_path)

    sim_id = str(uuid.uuid4())
    policy_key = "test_policy"
    policy_version = 1

    db._insert_simulation(
        sim_id=sim_id,
        name="test_sim",
        suite="test_suite",
        env="env_test",
        policy_key=policy_key,
        policy_version=policy_version,
    )

    episodes: List[str] = []
    for i in range(5):
        ep_id = str(uuid.uuid4())
        episodes.append(ep_id)

        attributes = {"seed": str(i)}
        created_at = datetime.datetime.now()

        agent_groups = {0: 0, 1: 1}
        agent_metrics = {
            0: {"reward": 1.0 + i},
            1: {"reward": 1.5 + i},
        }
        if i < 2:  # only first two episodes log hearts
            agent_metrics[0]["hearts_collected"] = 3.0
            agent_metrics[1]["hearts_collected"] = 2.0  # belongs to other policy

        db.record_episode(
            ep_id,
            attributes,
            agent_metrics,
            agent_groups,
            step_count=100,
            replay_url=None,
            created_at=created_at,
        )
        db.con.execute("UPDATE episodes SET simulation_id = ? WHERE id = ?", (sim_id, ep_id))

    for ep_id in episodes:
        db._insert_agent_policies([ep_id], {0: (policy_key, policy_version)})  # agent‑0 → test_policy
        db._insert_agent_policies([ep_id], {1: ("other_policy", 2)})  # agent‑1 → other_policy

    db.con.commit()
    return db, episodes, sim_id


# -------- Pytest fixtures -------------------------------------------------- #
@pytest.fixture
def test_db():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / f"{uuid.uuid4().hex}.duckdb"
        db, eps, sid = _create_test_db_with_missing_metrics(p)
        yield db, eps, sid
        db.close()


@pytest.fixture(autouse=True)
def _patch_policyrecord(monkeypatch):
    monkeypatch.setattr("metta.eval.eval_stats_db.PolicyRecord", MockPolicyRecord)


# -------- Tests ------------------------------------------------------------ #
def test_metrics_normalisation(test_db):
    db, _, _ = test_db
    policy = MockPolicyRecord("test_policy", 1)

    # hearts_collected: only 2/5 potential samples recorded (value 3 each)
    avg_hearts = db.get_average_metric_by_filter("hearts_collected", policy)
    assert 1.15 <= avg_hearts <= 1.25, f"expected ≈1.2 got {avg_hearts}"

    potential = db.potential_samples_for_metric(policy._policy_key, policy._policy_version)
    assert potential == 5

    recorded = db.count_metric_agents(policy._policy_key, policy._policy_version, "hearts_collected")
    assert recorded == 2

    # reward recorded for every sample → mean unaffected
    avg_reward = db.get_average_metric_by_filter("reward", policy)
    assert avg_reward is not None

    # filter condition
    avg_filtered = db.get_average_metric_by_filter("hearts_collected", policy, "sim_suite = 'test_suite'")
    assert 1.15 <= avg_filtered <= 1.25

    # non‑matching filter
    assert db.get_average_metric_by_filter("hearts_collected", policy, "sim_suite = 'none'") is None


def test_simulation_scores_normalisation(test_db):
    db, _, _ = test_db
    policy = MockPolicyRecord("test_policy", 1)

    scores = db.simulation_scores(policy, "hearts_collected")
    assert len(scores) == 1

    key = next(iter(scores))
    exp = scores[key]
    assert key == ("test_suite", "test_sim", "env_test")
    assert 1.15 <= exp <= 1.25

    # Compare to raw (non‑normalised) mean
    raw = db.query("""
        SELECT AVG(value) AS a FROM policy_simulation_agent_metrics
         WHERE policy_key='test_policy' AND policy_version=1 AND metric='hearts_collected'
    """)["a"][0]
    assert 2.9 <= raw <= 3.1  # expected ≈3


def test_sum_metric_normalisation(test_db):
    db, _, _ = test_db
    policy = MockPolicyRecord("test_policy", 1)

    sum_norm = db.get_sum_metric_by_filter("hearts_collected", policy)
    assert 1.15 <= sum_norm <= 1.25  # (6 / 5) ≈ 1.2


def test_no_metrics(test_db):
    db, _, _ = test_db
    policy = MockPolicyRecord("test_policy", 1)

    assert db.get_average_metric_by_filter("nonexistent", policy) == 0.0

    bad_policy = MockPolicyRecord("none", 99)
    assert db.get_average_metric_by_filter("hearts_collected", bad_policy) is None


def test_empty_database():
    with tempfile.TemporaryDirectory() as tmp:
        db = EvalStatsDB(Path(tmp) / "empty.duckdb")
        policy = MockPolicyRecord("test", 1)

        assert db.get_average_metric_by_filter("reward", policy) is None
        assert db.potential_samples_for_metric("test", 1) == 0
        db.close()


def test_metric_by_policy_eval(test_db):
    """metric_by_policy_eval should return a normalised mean per policy and eval."""
    db, _, _ = test_db

    policy = MockPolicyRecord("test_policy", 1)
    df = db.metric_by_policy_eval("hearts_collected", policy)

    # Expect one row (env_test) with ≈1.2
    assert len(df) == 1

    row = df.iloc[0]
    assert row["policy_uri"] == "test_policy:v1"
    assert row["eval_name"] == "env_test"
    assert 1.15 <= row["value"] <= 1.25, f"expected ≈1.2 got {row['value']}"
