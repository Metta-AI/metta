from __future__ import annotations

import datetime
import uuid
from pathlib import Path

from metta.sim.simulation_stats_db import SimulationStatsDB


def _create_worker_db(path: Path, sim_steps: int = 0) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = SimulationStatsDB(path)

    for table_name, sql in db.tables().items():
        db.con.execute(sql)

    db.con.execute("ALTER TABLE episodes ADD COLUMN IF NOT EXISTS simulation_id TEXT")

    episode_id = str(uuid.uuid4())
    attributes = {"seed": "0", "map_w": "1", "map_h": "1"}
    groups = [[0]]
    agent_metrics = {0: {"reward": 1.0}}
    group_metrics = {0: {"reward": 1.0}}
    replay_url = None
    created_at = datetime.datetime.now()

    db.record_episode(
        episode_id,
        attributes,
        groups,
        agent_metrics,
        group_metrics,
        sim_steps,
        replay_url,
        created_at,
    )

    db.close()
    return episode_id


_DUMMY_AGENT_MAP = {0: ("dummy_policy", 0)}


def test_from_uri_context_manager(tmp_path: Path):
    db_path = tmp_path / "test_db.duckdb"
    ep_id = _create_worker_db(db_path)

    with SimulationStatsDB.from_uri(str(db_path)) as db:
        episode_ids = [row[0] for row in db.con.execute("SELECT id FROM episodes").fetchall()]
        assert ep_id in episode_ids


def test_insert_agent_policies(tmp_path: Path):
    db_path = tmp_path / "policies.duckdb"
    db = SimulationStatsDB(db_path)

    for table_name, sql in db.tables().items():
        db.con.execute(sql)

    episode_id = str(uuid.uuid4())
    db.con.execute("INSERT INTO episodes (id) VALUES (?)", (episode_id,))

    agent_map = {0: ("policy_a", 1), 1: ("policy_b", 0)}
    db._insert_agent_policies([episode_id], agent_map)

    rows = db.con.execute("SELECT * FROM agent_policies").fetchall()
    assert len(rows) == 2

    db.close()


def test_insert_agent_policies_empty_inputs(tmp_path: Path):
    db_path = tmp_path / "empty_policies.duckdb"
    db = SimulationStatsDB(db_path)

    for table_name, sql in db.tables().items():
        db.con.execute(sql)

    db._insert_agent_policies([], _DUMMY_AGENT_MAP)
    db._insert_agent_policies([str(uuid.uuid4())], {})

    count = db.con.execute("SELECT COUNT(*) FROM agent_policies").fetchone()[0]
    assert count == 0

    db.close()


def test_merge_in(tmp_path: Path):
    db1_path = tmp_path / "db1.duckdb"
    db2_path = tmp_path / "db2.duckdb"

    ep1 = _create_worker_db(db1_path)
    ep2 = _create_worker_db(db2_path)

    db1 = SimulationStatsDB(db1_path)
    db2 = SimulationStatsDB(db2_path)

    db1.merge_in(db2)

    episodes = db1.con.execute("SELECT id FROM episodes").fetchall()
    episode_ids = [row[0] for row in episodes]
    assert len(episode_ids) == 2
    assert sorted(episode_ids) == sorted([ep1, ep2])

    db1.close()
    db2.close()


def test_tables(tmp_path: Path):
    db_path = tmp_path / "tables.duckdb"
    db = SimulationStatsDB(db_path)

    tables = db.tables()
    assert "episodes" in tables
    assert "agent_metrics" in tables
    assert "simulations" in tables
    assert "agent_policies" in tables

    db.close()


def test_insert_simulation(tmp_path: Path):
    db_path = tmp_path / "sim_table.duckdb"
    db = SimulationStatsDB(db_path)

    for table_name, sql in db.tables().items():
        db.con.execute(sql)

    sim_id = str(uuid.uuid4())
    policy_key = "test_policy"
    policy_version = 1
    db._insert_simulation(sim_id, "test_sim", "test_suite", "test_env", policy_key, policy_version)

    rows = db.con.execute("SELECT id, name, suite, env, policy_key, policy_version FROM simulations").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == sim_id
    assert rows[0][1] == "test_sim"
    assert rows[0][2] == "test_suite"
    assert rows[0][3] == "test_env"
    assert rows[0][4] == policy_key
    assert rows[0][5] == policy_version

    db.close()


def test_get_replay_urls(tmp_path: Path):
    """Test retrieving replay URLs with various filters."""
    db_path = tmp_path / "replay_urls.duckdb"
    db = SimulationStatsDB(db_path)

    # Create tables
    for table_name, sql in db.tables().items():
        db.con.execute(sql)

    # Add the simulation_id column to episodes if it doesn't exist
    db.con.execute("ALTER TABLE episodes ADD COLUMN IF NOT EXISTS simulation_id TEXT")

    # Create a few episodes with replay URLs
    episodes = []
    replay_urls = [
        "https://example.com/replay1.json",
        "https://example.com/replay2.json",
        "https://example.com/replay3.json",
    ]

    # Create 3 episodes with different replay URLs
    for i in range(3):
        episode_id = str(uuid.uuid4())
        episodes.append(episode_id)

        # Add episode
        db.con.execute(
            """
            INSERT INTO episodes (id, replay_url)
            VALUES (?, ?)
            """,
            (episode_id, replay_urls[i]),
        )

    # Create simulations with different policies and environments
    simulation_data = [
        (str(uuid.uuid4()), "sim1", "suite1", "env1", "policy1", 1),
        (str(uuid.uuid4()), "sim2", "suite1", "env2", "policy1", 2),
        (str(uuid.uuid4()), "sim3", "suite2", "env1", "policy2", 1),
    ]

    for i, (sim_id, name, suite, env, policy_key, policy_version) in enumerate(simulation_data):
        # Add simulation
        db._insert_simulation(sim_id, name, suite, env, policy_key, policy_version)

        # Link episode to simulation
        db._update_episode_simulations([episodes[i]], sim_id)

    # Test with no filters (should return all replay URLs)
    all_urls = db.get_replay_urls()
    assert len(all_urls) == 3
    for url in replay_urls:
        assert url in all_urls

    # Test filtering by policy key
    policy1_urls = db._get_replay_urls(policy_key="policy1")
    assert len(policy1_urls) == 2
    assert replay_urls[0] in policy1_urls
    assert replay_urls[1] in policy1_urls

    # Test filtering by policy version
    version1_urls = db._get_replay_urls(policy_version=1)
    assert len(version1_urls) == 2
    assert replay_urls[0] in version1_urls
    assert replay_urls[2] in version1_urls

    # Test filtering by environment
    env1_urls = db.get_replay_urls(env="env1")
    assert len(env1_urls) == 2
    assert replay_urls[0] in env1_urls
    assert replay_urls[2] in env1_urls

    # Test combining filters
    combined_urls = db._get_replay_urls(policy_key="policy1", policy_version=1, env="env1")
    assert len(combined_urls) == 1
    assert replay_urls[0] in combined_urls

    db.close()
