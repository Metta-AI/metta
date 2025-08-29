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
from typing import List

import pytest
from typing_extensions import Generator

from metta.eval.eval_stats_db import EvalStatsDB
from metta.rl.checkpoint_manager import parse_checkpoint_filename

TestEvalStatsDb = tuple[EvalStatsDB, list[str], str]


def _create_test_db_with_missing_metrics(db_path: Path) -> TestEvalStatsDb:
    db = EvalStatsDB(db_path)

    checkpoint_filename = "test_policy.e1.s1000.t10.sc0.pt"
    pk, pv, agent_step, total_time, score = parse_checkpoint_filename(checkpoint_filename)

    sim_id = str(uuid.uuid4())

    db._insert_simulation(
        sim_id=sim_id,
        name="test_sim",
        env_name="test_env",
        policy_key=pk,
        policy_version=pv,
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
        db._insert_agent_policies([ep_id], {0: (pk, pv)})  # agent‑0 → test_policy
        db._insert_agent_policies([ep_id], {1: ("other_policy", 2)})  # agent‑1 → other_policy

    db.con.commit()
    return db, episodes, sim_id


# -------- Pytest fixtures -------------------------------------------------- #
@pytest.fixture
def test_db() -> Generator[TestEvalStatsDb, None, None]:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / f"{uuid.uuid4().hex}.duckdb"
        db, eps, sid = _create_test_db_with_missing_metrics(p)
        yield db, eps, sid
        db.close()


# -------- Tests ------------------------------------------------------------ #
def test_metrics_normalization(test_db: TestEvalStatsDb) -> None:
    db, _, _ = test_db
    checkpoint_filename = "test_policy.e1.s1000.t10.sc0.pt"
    pk, pv, agent_step, total_time, score = parse_checkpoint_filename(checkpoint_filename)

    # hearts_collected: only 2/5 potential samples recorded (value 3 each)
    # Create proper URI from the checkpoint components
    policy_uri = f"file:///tmp/{pk}.e{pv}.s{agent_step}.t{total_time}.sc{int(score * 10000)}.pt"
    avg_hearts = db.get_average_metric("hearts_collected", policy_uri)
    assert avg_hearts is not None
    assert 1.15 <= avg_hearts <= 1.25, f"expected ≈1.2 got {avg_hearts}"

    potential = db.potential_samples_for_metric(pk, pv)
    assert potential == 5

    recorded = db.count_metric_agents(pk, pv, "hearts_collected")
    assert recorded == 2

    # reward recorded for every sample → mean unaffected
    avg_reward = db.get_average_metric("reward", policy_uri)
    assert avg_reward is not None

    # filter condition
    avg_filtered = db.get_average_metric("hearts_collected", policy_uri, "sim_env = 'test_env'")
    assert avg_filtered is not None
    assert 1.15 <= avg_filtered <= 1.25

    # non‑matching filter
    assert db.get_average_metric("hearts_collected", policy_uri, "sim_env = 'none'") is None


def test_simulation_scores_normalization(test_db: TestEvalStatsDb) -> None:
    db, _, _ = test_db
    checkpoint_filename = "test_policy.e1.s1000.t10.sc0.pt"
    pk, pv, agent_step, total_time, score = parse_checkpoint_filename(checkpoint_filename)
    policy_uri = f"file:///tmp/{pk}.e{pv}.s{agent_step}.t{total_time}.sc{int(score * 10000)}.pt"

    scores = db.simulation_scores(policy_uri, "hearts_collected")
    assert len(scores) == 1

    key = next(iter(scores))
    exp = scores[key]
    assert key == ("test_sim", "test_env")
    assert 1.15 <= exp <= 1.25

    # Compare to raw (non‑normalized) mean
    raw = db.query("""
        SELECT AVG(value) AS a FROM policy_simulation_agent_metrics
         WHERE policy_key='test_policy' AND policy_version=1 AND metric='hearts_collected'
    """)["a"][0]
    assert 2.9 <= raw <= 3.1  # expected ≈3


def test_sum_metric_normalization(test_db: TestEvalStatsDb) -> None:
    db, _, _ = test_db
    checkpoint_filename = "test_policy.e1.s1000.t10.sc0.pt"
    pk, pv, agent_step, total_time, score = parse_checkpoint_filename(checkpoint_filename)
    policy_uri = f"file:///tmp/{pk}.e{pv}.s{agent_step}.t{total_time}.sc{int(score * 10000)}.pt"

    sum_norm = db.get_sum_metric("hearts_collected", policy_uri)
    assert sum_norm is not None
    assert 1.15 <= sum_norm <= 1.25  # (6 / 5) ≈ 1.2


def test_no_metrics(test_db: TestEvalStatsDb) -> None:
    db, _, _ = test_db
    checkpoint_filename = "test_policy.e1.s1000.t10.sc0.pt"
    pk, pv, agent_step, total_time, score = parse_checkpoint_filename(checkpoint_filename)
    policy_uri = f"file:///tmp/{pk}.e{pv}.s{agent_step}.t{total_time}.sc{int(score * 10000)}.pt"

    assert db.get_average_metric("nonexistent", policy_uri) == 0.0

    # Test with invalid URI
    invalid_uri = "file:///tmp/none.e99.s0.t0.sc0.pt"
    assert db.get_average_metric("hearts_collected", invalid_uri) is None


def test_empty_database():
    with tempfile.TemporaryDirectory() as tmp:
        db = EvalStatsDB(Path(tmp) / "empty.duckdb")
        test_uri = "file:///tmp/test.e1.s0.t0.sc0.pt"
        assert db.get_average_metric("reward", test_uri) is None
        pk, pv = "test", 1
        assert db.potential_samples_for_metric(pk, pv) == 0
        db.close()
