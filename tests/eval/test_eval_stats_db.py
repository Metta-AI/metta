#!/usr/bin/env python3

import datetime
import tempfile
import uuid
from pathlib import Path

import pytest

from metta.eval.eval_stats_db import EvalStatsDB
from metta.rl.checkpoint_manager import CheckpointManager

TestEvalStatsDb = tuple[EvalStatsDB, str, str]  # (db, policy_key, policy_version)


def _create_test_db_with_missing_metrics(db_path: Path) -> TestEvalStatsDb:
    db = EvalStatsDB(db_path)

    checkpoint_filename = "test_policy/checkpoints/test_policy:v1.pt"
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
    episode_ids = []
    for i in range(5):
        episode_id = f"episode_{i}"
        episode_ids.append(episode_id)

        # Create agent metrics - hearts_collected only for episodes 0 and 1
        agent_metrics = {0: {"reward": 2.0}}
        if i < 2:
            agent_metrics[0]["hearts_collected"] = 3.0

        # Record the episode with metrics
        db.record_episode(
            episode_id=episode_id,
            attributes={},
            agent_metrics=agent_metrics,
            agent_groups={0: 0},  # agent 0 in group 0
            step_count=100,
            replay_url=None,
            created_at=datetime.datetime.now(),
        )

        # Update the episode to link it to our simulation
        db.con.execute("UPDATE episodes SET simulation_id = ? WHERE id = ?", (sim_id, episode_id))

    # Insert agent policy mappings for all episodes
    agent_map = {0: (pk, pv)}  # agent 0 uses our policy
    db._insert_agent_policies(episode_ids, agent_map)

    return db, pk, str(pv)


@pytest.fixture
def test_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.duckdb"
        db, pk, pv = _create_test_db_with_missing_metrics(db_path)
        yield db, pk, pv
        db.close()


def test_metrics_normalization(test_db: TestEvalStatsDb) -> None:
    db, _, _ = test_db
    checkpoint_filename = "test_policy/checkpoints/test_policy:v1.pt"
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

    avg_reward = db.get_average_metric("reward", policy_uri)
    assert avg_reward is not None
    avg_filtered = db.get_average_metric("hearts_collected", policy_uri, "sim_env = 'test_env'")
    assert avg_filtered is not None
    assert 1.15 <= avg_filtered <= 1.25


def test_simulation_scores_normalization(test_db: TestEvalStatsDb) -> None:
    db, _, _ = test_db
    checkpoint_filename = "test_policy/checkpoints/test_policy:v1.pt"
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


def test_no_metrics(test_db: TestEvalStatsDb) -> None:
    db, _, _ = test_db
    checkpoint_filename = "test_policy/checkpoints/test_policy:v1.pt"
    policy_uri = CheckpointManager.normalize_uri(f"/tmp/{checkpoint_filename}")

    assert db.get_average_metric("nonexistent", policy_uri) == 0.0

    invalid_uri = CheckpointManager.normalize_uri("/tmp/none/checkpoints/none:v99.pt")
    assert db.get_average_metric("hearts_collected", invalid_uri) is None


def test_empty_database():
    with tempfile.TemporaryDirectory() as tmp:
        db = EvalStatsDB(Path(tmp) / "empty.duckdb")
        test_uri = CheckpointManager.normalize_uri("/tmp/test/checkpoints/test:v1.pt")
        assert db.get_average_metric("reward", test_uri) is None
        pk, pv = "test", 1
        assert db.potential_samples_for_metric(pk, pv) == 0
        db.close()
