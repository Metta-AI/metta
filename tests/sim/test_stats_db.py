"""tests/sim/test_stats_db.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from metta.sim.stats_db import StatsDB


# --------------------------------------------------------------------------- #
# helper fixtures                                                             #
# --------------------------------------------------------------------------- #
@pytest.fixture
def sample_rollout():
    """Return *meta*, *agents*, *metrics* dictionaries."""
    meta = dict(
        env_name="mettagrid.nav_easy",
        map_w=16,
        map_h=16,
        epoch=0,
        batch_idx=3,
        agent_steps=512,
    )
    agents = {
        0: ("wandb://run/learner_policy", 7),  # agent-id → (uri, version)
        1: ("wandb://run/npc_policy", 2),
    }
    # one metric per agent for brevity
    metrics = {
        0: {"episode_reward": 0.8, "kills": 3},
        1: {"episode_reward": 0.6, "kills": 1},
    }
    return meta, agents, metrics

@mock_aws
def test_stats_db_roundtrip_s3(sample_rollout):
    meta, agents, metrics = sample_rollout

    # working dir ------------ #
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # 1. create DB & write data
        local_db = tmpdir_path / "stats.duckdb"
        db = StatsDB(local_db)
        rid = db.log_rollout(meta, agents, metrics)
        db.close()

        # 2. mock S3 bucket + upload
        s3 = boto3.client("s3", region_name="us-east-1")
        bucket, key = "metta-test-bucket", "artifacts/stats.duckdb"
        s3.create_bucket(Bucket=bucket)
        s3.upload_file(str(local_db), bucket, key)

        # 3. download elsewhere
        download_dir = tmpdir_path / "dl"
        download_dir.mkdir()
        dl_path = download_dir / "stats.duckdb"
        s3.download_file(bucket, key, str(dl_path))

        # 4. open & verify data integrity directly from tables
        db2 = StatsDB(dl_path)
        
        # Check rollouts table
        rollouts_df = db2.query("SELECT * FROM rollouts WHERE rollout_id = ?", rid)
        assert len(rollouts_df) == 1
        assert rollouts_df.iloc[0]["env_name"] == "mettagrid.nav_easy"
        assert rollouts_df.iloc[0]["agent_steps"] == 512
        
        # Check rollout_agents table
        agents_df = db2.query("SELECT * FROM rollout_agents WHERE rollout_id = ?", rid)
        assert len(agents_df) == 2
        assert set(agents_df["agent_id"].tolist()) == {0, 1}
        
        # Check rollout_agent_metrics table
        metrics_df = db2.query("""
            SELECT agent_id, metric, value 
            FROM rollout_agent_metrics 
            WHERE rollout_id = ? AND metric = 'episode_reward'
            ORDER BY agent_id
        """, rid)
        
        assert len(metrics_df) == 2
        assert metrics_df.iloc[0]["agent_id"] == 0
        assert metrics_df.iloc[0]["value"] == 0.8
        assert metrics_df.iloc[1]["agent_id"] == 1
        assert metrics_df.iloc[1]["value"] == 0.6
        
        # Verify the average episode_reward across both agents is 0.7
        avg_reward_df = db2.query("""
            SELECT AVG(value) as mean, COUNT(*) as n
            FROM rollout_agent_metrics
            WHERE rollout_id = ? AND metric = 'episode_reward'
        """, rid)
        
        assert len(avg_reward_df) == 1
        assert pytest.approx(avg_reward_df.iloc[0]["mean"], rel=1e-6) == 0.7
        assert avg_reward_df.iloc[0]["n"] == 2
        
        db2.close()
