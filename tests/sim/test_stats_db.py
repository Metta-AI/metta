"""tests/sim/test_stats_db.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Integration-style tests for the new StatsDB schema.

Requires Moto for the S3 round-trip.
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


# --------------------------------------------------------------------------- #
# 1. Basic insert / aggregate test                                            #
# --------------------------------------------------------------------------- #
def test_policy_env_stats_aggregation(sample_rollout):
    meta, agents, metrics = sample_rollout
    db = StatsDB(":memory:")
    rid = db.log_rollout(meta, agents, metrics)

    # check raw tables
    df_raw = db.query("SELECT COUNT(*) AS n FROM rollout_metrics")
    assert df_raw.iloc[0]["n"] == 4  # 2 agents × 2 metrics

    # check aggregate table
    df_agg = db.query("SELECT * FROM policy_environment_stats")
    assert len(df_agg) == 2  # two different metrics

    # episode_reward aggregate should be (0.8, 0.6)
    er = df_agg[df_agg["metric"] == "episode_reward"].iloc[0]
    assert pytest.approx(er["mean"], rel=1e-6) == 0.7
    assert er["n"] == 2


# --------------------------------------------------------------------------- #
# 2. S3 round-trip using Moto                                                 #
# --------------------------------------------------------------------------- #
@mock_aws
def test_stats_db_roundtrip_s3(sample_rollout):
    meta, agents, metrics = sample_rollout

    # working dir ------------ #
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # 1. create DB & write data
        local_db = tmpdir_path / "stats.duckdb"
        db = StatsDB(local_db)
        _ = db.log_rollout(meta, agents, metrics)
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

        # 4. open & verify aggregate still there
        db2 = StatsDB(dl_path)
        df = db2.query("SELECT metric, mean, n FROM policy_environment_stats WHERE metric='episode_reward'")
        assert len(df) == 1
        assert pytest.approx(df.iloc[0]["mean"], rel=1e-6) == 0.7
        assert df.iloc[0]["n"] == 2
        db2.close()
