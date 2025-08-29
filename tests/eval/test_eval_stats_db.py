#!/usr/bin/env python3

import tempfile
import uuid
from pathlib import Path

import pytest

from metta.eval.eval_stats_db import EvalStatsDB
from metta.rl.checkpoint_manager import CheckpointManager

# Type alias for the test database fixture
TestEvalStatsDb = tuple[EvalStatsDB, str, str]  # (db, policy_key, policy_version)


# -------- Test Database Creation ----------------------------------------- #
def _create_test_db_with_missing_metrics(db_path: Path) -> TestEvalStatsDb:
    db = EvalStatsDB(db_path)

    checkpoint_filename = "test_policy.e1.s1000.t10.sc0.pt"
    metadata = CheckpointManager.get_policy_metadata(CheckpointManager.normalize_uri(f"/tmp/{checkpoint_filename}"))
    pk, pv = metadata["run_name"], metadata["epoch"]
    _agent_step, _total_time, _score = (
        metadata.get("agent_step", 1000),
        metadata.get("total_time", 10),
        metadata.get("score", 0.0),
    )

    sim_id = str(uuid.uuid4())

    db._insert_simulation(
        sim_id=sim_id,
        name="test_sim",
        env_name="test_env",
        policy_key=pk,
        policy_version=pv,
    )

    # Create 5 episodes, each with 1 agent
    for i in range(5):
        episode_id = f"episode_{i}"
        db._insert_episode(episode_id, sim_id, step_count=100)

        # Insert agent policy
        db._insert_agent_policy(episode_id=episode_id, agent_id=0, policy_key=pk, policy_version=pv)

        # Insert reward for all episodes (this is always recorded)
        db._insert_agent_metric(episode_id, agent_id=0, metric="reward", value=2.0)

        # ONLY insert hearts_collected for episodes 0 and 1 (simulates missing data)
        if i < 2:
            db._insert_agent_metric(episode_id, agent_id=0, metric="hearts_collected", value=3.0)

    return db, pk, str(pv)


@pytest.fixture
def test_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.duckdb"
        db, pk, pv = _create_test_db_with_missing_metrics(db_path)
        yield db, pk, pv
        db.close()


# -------- Tests ------------------------------------------------------------ #
def test_metrics_normalization(test_db: TestEvalStatsDb) -> None:
    db, _, _ = test_db
    checkpoint_filename = "test_policy.e1.s1000.t10.sc0.pt"
    # Use CheckpointManager to get metadata and normalize URI
    policy_uri = CheckpointManager.normalize_uri(f"/tmp/{checkpoint_filename}")
    metadata = CheckpointManager.get_policy_metadata(policy_uri)
    pk, pv = metadata["run_name"], metadata["epoch"]

    # hearts_collected: only 2/5 potential samples recorded (value 3 each)
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


def test_simulation_scores_normalization(test_db: TestEvalStatsDb) -> None:
    db, _, _ = test_db
    checkpoint_filename = "test_policy.e1.s1000.t10.sc0.pt"
    policy_uri = CheckpointManager.normalize_uri(f"/tmp/{checkpoint_filename}")

    scores = db.simulation_scores(policy_uri, "hearts_collected")
    assert len(scores) == 1

    key = next(iter(scores))
    exp = scores[key]
    assert key == ("test_sim", "test_env")
    assert 1.15 <= exp <= 1.25

    # Compare to raw (non‑normalized) mean
    raw = db.query(
        """
        SELECT AVG(value) AS a FROM policy_simulation_agent_metrics
         WHERE policy_key='test_policy' AND policy_version=1 AND metric='hearts_collected'
    """
    )["a"][0]
    assert 2.9 <= raw <= 3.1  # expected ≈3


def test_sum_metric_normalization(test_db: TestEvalStatsDb) -> None:
    db, _, _ = test_db
    checkpoint_filename = "test_policy.e1.s1000.t10.sc0.pt"
    policy_uri = CheckpointManager.normalize_uri(f"/tmp/{checkpoint_filename}")

    sum_norm = db.get_sum_metric("hearts_collected", policy_uri)
    assert sum_norm is not None
    assert 1.15 <= sum_norm <= 1.25  # (6 / 5) ≈ 1.2


def test_no_metrics(test_db: TestEvalStatsDb) -> None:
    db, _, _ = test_db
    checkpoint_filename = "test_policy.e1.s1000.t10.sc0.pt"
    policy_uri = CheckpointManager.normalize_uri(f"/tmp/{checkpoint_filename}")

    assert db.get_average_metric("nonexistent", policy_uri) == 0.0

    # Test with invalid URI
    invalid_uri = CheckpointManager.normalize_uri("/tmp/none.e99.s0.t0.sc0.pt")
    assert db.get_average_metric("hearts_collected", invalid_uri) is None


def test_empty_database():
    with tempfile.TemporaryDirectory() as tmp:
        db = EvalStatsDB(Path(tmp) / "empty.duckdb")
        test_uri = CheckpointManager.normalize_uri("/tmp/test.e1.s0.t0.sc0.pt")
        assert db.get_average_metric("reward", test_uri) is None
        pk, pv = "test", 1
        assert db.potential_samples_for_metric(pk, pv) == 0
        db.close()
