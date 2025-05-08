"""
Integration tests for metta.sim.stats_db.StatsDB.

These tests spin up tiny worker shards with the vanilla MettaGrid StatsDB,
then merge them through Metta's helper and verify:

* shards merge correctly,
* episodes carry env‑names,
* simulations table + simulation_id FK are populated,
* agent‑policy multiplexing works,
* helper utilities behave,
* edge‑cases (empty shards / empty inputs) don't crash.

Version note
------------
`policy_version` is now an **INT NOT NULL** in the schema.  All literal
versions below therefore use integers (e.g. `1`, `0`, `i`) instead of the
former string tokens like "v1" or "default".
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest
from duckdb import CatalogException

from metta.sim.simulation_stats_db import StatsDB
from mettagrid.stats_writer import StatsDB as MGStatsDB

# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #


def _create_worker_db(path: Path, sim_steps: int = 0) -> str:
    """Create a shard with **one** episode + a single agent‑metric row.

    Returns
    -------
    episode_id : str
        The UUID of the single episode – used by the tests for assertions.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    db = MGStatsDB(str(path), read_only=False)

    ep_id = db.create_episode(seed=0, map_w=1, map_h=1)
    db.add_agent_metrics(ep_id, agent_id=0, metrics={"reward": 1.0})
    db.finish_episode(ep_id, step_count=sim_steps)
    db.close()
    return ep_id


# Integer versions are now required by the schema
_DUMMY_AGENT_MAP = {0: ("dummy_policy", 0)}


# --------------------------------------------------------------------------- #
# merge & context                                                             #
# --------------------------------------------------------------------------- #


def test_merge_two_shards_with_context(tmp_path: Path):
    shards_dir = tmp_path / "shards"
    ep_a = _create_worker_db(shards_dir / "worker0.duckdb")
    ep_b = _create_worker_db(shards_dir / "worker1.duckdb")

    merged: StatsDB = StatsDB.from_shards_and_context(shards_dir, _DUMMY_AGENT_MAP, "sim_alpha", "suite_nav", "env_a")

    # rows copied -----------------------------------------------------------
    df = merged.query("SELECT id FROM episodes ORDER BY id")
    assert sorted(df["id"]) == sorted([ep_a, ep_b])

    # simulation linkage ----------------------------------------------------
    join = merged.query(
        """
        SELECT e.id, s.name, s.suite
          FROM episodes e
          JOIN simulations s ON e.simulation_id = s.id
        """
    )
    assert set(join["name"]) == {"sim_alpha"}
    assert set(join["suite"]) == {"suite_nav"}
    assert set(join["id"]) == {ep_a, ep_b}

    # agent‑policy multiplex -----------------------------------------------
    ap = merged.query("SELECT * FROM agent_policies")
    assert ap.shape == (2, 4)  # 2 episodes × 1 agent
    assert set(ap["policy_key"]) == {"dummy_policy"}

    merged.close()


def test_merge_empty_shard_dir(tmp_path: Path):
    merged = StatsDB.from_shards_and_context(tmp_path, _DUMMY_AGENT_MAP, "sim_empty", "suite_empty", "env_a")
    assert merged.query("SELECT COUNT(*) AS n FROM episodes")["n"][0] == 0
    merged.close()


# --------------------------------------------------------------------------- #
# simulation‑id semantics                                                     #
# --------------------------------------------------------------------------- #


def test_simulation_id_uniqueness(tmp_path: Path):
    shards = tmp_path / "shards"
    _create_worker_db(shards / "a.duckdb")
    merged1 = StatsDB.from_shards_and_context(shards, _DUMMY_AGENT_MAP, "sim1", "suite", "env_a")
    sim_id_1 = merged1.query("SELECT DISTINCT simulation_id FROM episodes")["simulation_id"][0]
    merged1.close()

    merged2 = StatsDB.from_shards_and_context(shards, _DUMMY_AGENT_MAP, "sim2", "suite", "env_a")
    sim_id_2 = merged2.query("SELECT DISTINCT simulation_id FROM episodes")["simulation_id"][0]
    merged2.close()

    assert sim_id_1 != sim_id_2


def test_simulation_id_deduplication(tmp_path: Path):
    """Merging twice with the *same* (suite,name) pair must reuse the UUID."""
    shards = tmp_path / "shards"
    _create_worker_db(shards / "a.duckdb")
    merged1 = StatsDB.from_shards_and_context(shards, _DUMMY_AGENT_MAP, "sim_fixed", "suite", "env_a")
    sim_id_1 = merged1.query("SELECT DISTINCT simulation_id FROM episodes")["simulation_id"][0]
    merged1.close()

    merged2 = StatsDB.from_shards_and_context(shards, _DUMMY_AGENT_MAP, "sim_fixed", "suite", "env_a")
    sim_id_2 = merged2.query("SELECT DISTINCT simulation_id FROM episodes")["simulation_id"][0]
    merged2.close()

    assert sim_id_1 == sim_id_2


# --------------------------------------------------------------------------- #
# _insert_agent_policies                                                       #
# --------------------------------------------------------------------------- #


def test_insert_agent_policies(tmp_path: Path):
    db_path = tmp_path / "policies.duckdb"
    db = StatsDB(db_path, mode="rwc")

    episodes = [uuid.uuid4().hex for _ in range(3)]
    agent_map = {0: ("policy_a", 1), 1: ("policy_b", 0)}
    db._insert_agent_policies(episodes, agent_map)

    pol = db.query("SELECT * FROM policies")
    assert pol.shape[0] == 2

    ap = db.query("SELECT * FROM agent_policies")
    assert ap.shape[0] == len(episodes) * len(agent_map)
    db.close()


def test_insert_agent_policies_empty_inputs(tmp_path: Path):
    db = StatsDB(tmp_path / "x.duckdb", mode="rwc")
    db._insert_agent_policies([], _DUMMY_AGENT_MAP)
    db._insert_agent_policies([uuid.uuid4().hex], {})
    assert db.query("SELECT COUNT(*) AS n FROM agent_policies")["n"][0] == 0
    db.close()


# --------------------------------------------------------------------------- #
# utilities                                                                   #
# --------------------------------------------------------------------------- #


def test_query_method(tmp_path: Path):
    db = StatsDB(tmp_path / "q.duckdb", mode="rwc")
    ep = db.get_next_episode_id()
    db.con.execute(
        """
        INSERT INTO episodes
            (id, seed, map_w, map_h,
             step_count, started_at, finished_at, metadata)
        VALUES (?, 0, 1, 1, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, NULL)
        """,
        (ep,),
    )
    df = db.query("SELECT * FROM episodes")
    assert df.loc[0, "id"] == ep
    db.close()


def test_sequential_policy_simulations_and_merging(tmp_path: Path):
    """Evaluate multiple policies sequentially and merge their episodes."""

    with tempfile.TemporaryDirectory() as temp_dir:
        # directories for individual simulations
        policy_dirs = [Path(temp_dir) / f"policy_{i}" for i in range(3)]
        for p in policy_dirs:
            p.mkdir(parents=True)

        merged_db_path = Path(temp_dir) / "merged.duckdb"
        merged_db = StatsDB(merged_db_path)

        total_episodes = 0
        for i, policy_dir in enumerate(policy_dirs):
            shard_path = policy_dir / f"stats_{i}.duckdb"
            shard_db = StatsDB(shard_path)

            ep_id = shard_db.create_episode(seed=i, map_w=10, map_h=10)
            shard_db.add_agent_metrics(ep_id, 0, {"reward": float(i)})
            shard_db.finish_episode(ep_id, 100)

            agent_map = {0: (f"policy_{i}", i)}  # integer version

            policy_db = StatsDB.from_shards_and_context(
                policy_dir, agent_map, f"sim_{i}", "test_suite", f"env/test_{i}"
            )

            merged_db.merge_in(policy_db)
            policy_db.close()

            total_episodes += 1
            count = merged_db.con.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            assert count == total_episodes

            metrics = merged_db.con.execute(
                "SELECT e.id, a.value FROM episodes e JOIN agent_metrics a ON e.id = a.episode_id"
            ).fetchall()
            assert len(metrics) == total_episodes

            pol_count = merged_db.con.execute("SELECT COUNT(*) FROM policies").fetchone()[0]
            assert pol_count == total_episodes

        assert merged_db.con.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 3
        merged_db.close()


# --------------------------------------------------------------------------- #
# fixture for materialised‑view tests                                         #
# --------------------------------------------------------------------------- #


@pytest.fixture
def test_db():
    """Create a temporary StatsDB with rich test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / f"{uuid.uuid4().hex}.duckdb"
        db = StatsDB(db_path, mode="rwc")

        sim_id = db.ensure_simulation_id("test_sim", "test_suite", "env_test")

        episodes = []
        for i in range(3):
            ep_id = db.get_next_episode_id()
            episodes.append(ep_id)
            db.con.execute(
                """
                INSERT INTO episodes
                    (id, seed, map_w, map_h, step_count, started_at, finished_at, simulation_id)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
                """,
                (ep_id, i, 10, 10, 100, sim_id),
            )

        # two policies controlling different agents
        agent_policies = {
            "policy_a": (1, [0]),
            "policy_b": (2, [1]),
        }
        for key, (ver, aids) in agent_policies.items():
            for ep in episodes:
                for aid in aids:
                    db.con.execute(
                        """
                        INSERT INTO agent_policies
                            (episode_id, agent_id, policy_key, policy_version)
                        VALUES (?, ?, ?, ?)
                        """,
                        (ep, aid, key, ver),
                    )

        metrics = {"reward": [1.0, 2.0, 3.0], "steps": [100, 200, 300], "efficiency": [0.5, 0.7, 0.9]}
        for idx, ep in enumerate(episodes):
            for aid in range(2):
                for m, vals in metrics.items():
                    db.con.execute(
                        """
                        INSERT INTO agent_metrics
                            (episode_id, agent_id, metric, value)
                        VALUES (?, ?, ?, ?)
                        """,
                        (ep, aid, m, vals[idx] + np.random.normal(0, 0.1)),
                    )
        yield db
        db.close()


# --------------------------------------------------------------------------- #
# materialised‑view tests                                                     #
# --------------------------------------------------------------------------- #


def test_materialize_policy_simulations_view_creates_table(test_db):
    metric = "reward"
    expected = f"policy_simulations_{metric}"
    test_db.materialize_policy_simulations_view(metric)
    tables = test_db.con.execute("SHOW TABLES").fetchall()
    assert (expected,) in tables


def test_materialize_policy_simulations_view_with_nonexistent_metric(test_db):
    metric = "nonexistent_metric"
    with pytest.raises(ValueError):
        test_db.materialize_policy_simulations_view(metric)


def test_materialize_policy_simulations_view_aggregates_correctly(test_db):
    metric = "reward"
    test_db.materialize_policy_simulations_view(metric)

    rows = test_db.con.execute(f"SELECT COUNT(*) FROM policy_simulations_{metric}").fetchone()[0]
    assert rows == 2  # 2 policies × 1 simulation

    result = test_db.con.execute(f"SELECT * FROM policy_simulations_{metric} LIMIT 1").fetchone()
    assert result is not None


def test_materialize_policy_simulations_view_primary_key(test_db):
    metric = "reward"
    test_db.materialize_policy_simulations_view(metric)

    pk_info = test_db.con.execute(f"PRAGMA table_info(policy_simulations_{metric})").fetchall()
    pk_cols = {col[1] for col in pk_info if col[5] > 0}
    assert pk_cols == {"policy_key", "policy_version", "sim_suite", "sim_env"}


def test_materialize_policy_simulations_view_multiple_metrics(test_db):
    for metric in ("reward", "steps", "efficiency"):
        test_db.materialize_policy_simulations_view(metric)
    tables = {t[0] for t in test_db.con.execute("SHOW TABLES").fetchall()}
    for metric in ("reward", "steps", "efficiency"):
        assert f"policy_simulations_{metric}" in tables


def test_materialize_policy_simulations_view_invalid_metric_name(test_db):
    with pytest.raises(ValueError):
        test_db.materialize_policy_simulations_view("invalid; DROP TABLE episodes; --")


def test_view_query_results(test_db):
    metric = "reward"
    test_db.materialize_policy_simulations_view(metric)
    result = test_db.con.execute(
        f"SELECT * FROM policy_simulations_{metric} WHERE sim_suite = 'test_suite' AND sim_name = 'test_sim'"
    ).fetchall()
    assert len(result) > 0


def test_get_average_metric_by_filter(tmp_path: Path):
    """Test the get_average_metric_by_filter function with various filters."""
    # Set up a test database with multiple policies, simulations, and metrics
    db_path = tmp_path / "metric_filter_test.duckdb"
    db = StatsDB(db_path, mode="rwc")

    # Create test simulations with different environments
    sim_envs = {
        "navigation": db.ensure_simulation_id("sim_nav", "test_suite", "env_navigation"),
        "object_use": db.ensure_simulation_id("sim_obj", "test_suite", "env_object_use"),
        "multiagent": db.ensure_simulation_id("sim_multi", "test_suite", "env_multiagent"),
    }

    # Add policy entries
    policies = [
        ("test_policy", 1),
        ("test_policy", 2),
        ("other_policy", 1),
    ]

    db.con.executemany(
        """
        INSERT INTO policies (policy_key, policy_version)
        VALUES (?, ?)
        """,
        policies,
    )

    # Create episodes for each simulation
    episodes = {}
    for env_name, sim_id in sim_envs.items():
        episodes[env_name] = []
        for i in range(3):  # 3 episodes per simulation
            ep_id = db.get_next_episode_id()
            episodes[env_name].append(ep_id)
            db.con.execute(
                """
                INSERT INTO episodes 
                (id, seed, map_w, map_h, step_count, started_at, finished_at, simulation_id)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
                """,
                (ep_id, i, 10, 10, 100, sim_id),
            )

    # Add agent policies for each episode
    agent_policies = []
    for env_name, eps in episodes.items():
        for ep_id in eps:
            # Assign test_policy:1 to navigation and object_use, test_policy:2 to multiagent
            if env_name in ["navigation", "object_use"]:
                policy_key, policy_version = "test_policy", 1
            else:
                policy_key, policy_version = "test_policy", 2

            # Agent 0 gets test_policy, Agent 1 gets other_policy
            agent_policies.append((ep_id, 0, policy_key, policy_version))
            agent_policies.append((ep_id, 1, "other_policy", 1))

    db.con.executemany(
        """
        INSERT INTO agent_policies
        (episode_id, agent_id, policy_key, policy_version)
        VALUES (?, ?, ?, ?)
        """,
        agent_policies,
    )

    # Add metrics with predefined values
    # Navigation: 10, 12, 14 for test_policy:1
    # Object use: 20, 22, 24 for test_policy:1
    # Multiagent: 30, 32, 34 for test_policy:2
    # And some values for other_policy:1
    metrics_data = []

    for env_name, eps in episodes.items():
        for i, ep_id in enumerate(eps):
            if env_name == "navigation":
                metrics_data.append((ep_id, 0, "reward", 10 + i * 2))  # test_policy:1
                metrics_data.append((ep_id, 1, "reward", 5 + i))  # other_policy:1
            elif env_name == "object_use":
                metrics_data.append((ep_id, 0, "reward", 20 + i * 2))  # test_policy:1
                metrics_data.append((ep_id, 1, "reward", 15 + i))  # other_policy:1
            else:  # multiagent
                metrics_data.append((ep_id, 0, "reward", 30 + i * 2))  # test_policy:2
                metrics_data.append((ep_id, 1, "reward", 25 + i))  # other_policy:1

    db.con.executemany(
        """
        INSERT INTO agent_metrics
        (episode_id, agent_id, metric, value)
        VALUES (?, ?, ?, ?)
        """,
        metrics_data,
    )

    # Materialize the reward view
    db.materialize_policy_simulations_view("reward")

    # Test 1: Get average reward for test_policy:1 with navigation filter
    nav_avg = db.get_average_metric_by_filter("reward", "test_policy", 1, "sim_env LIKE '%navigation%'")
    assert nav_avg is not None
    assert abs(nav_avg - 12.0) < 0.01  # Should be (10 + 12 + 14) / 3 = 12

    # Test 2: Get average reward for test_policy:1 with object_use filter
    obj_avg = db.get_average_metric_by_filter("reward", "test_policy", 1, "sim_env LIKE '%object_use%'")
    assert obj_avg is not None
    assert abs(obj_avg - 22.0) < 0.01  # Should be (20 + 22 + 24) / 3 = 22

    # Test 3: Get average reward for test_policy:2 with multiagent filter
    multi_avg = db.get_average_metric_by_filter("reward", "test_policy", 2, "sim_env LIKE '%multiagent%'")
    assert multi_avg is not None
    assert abs(multi_avg - 32.0) < 0.01  # Should be (30 + 32 + 34) / 3 = 32

    # Test 4: Get overall average for test_policy:1 (navigation + object_use)
    overall_v1 = db.get_average_metric_by_filter("reward", "test_policy", 1)
    assert overall_v1 is not None
    assert abs(overall_v1 - 17.0) < 0.01  # Should be (10+12+14+20+22+24)/6 = 17

    # Test 5: Filter that doesn't match any simulations
    no_match = db.get_average_metric_by_filter("reward", "test_policy", 1, "sim_env LIKE '%nonexistent%'")
    assert no_match is None

    # Test 6: Policy that doesn't exist
    no_policy = db.get_average_metric_by_filter("reward", "nonexistent_policy", 1)
    assert no_policy is None

    # Test 7: Metric that doesn't exist (didn't materialize view)
    with pytest.raises(CatalogException):
        _ = db.get_average_metric_by_filter("nonexistent_metric", "test_policy", 1)

    db.close()


def test_deterministic_simulation_ids_across_policies():
    """
    Test that simulation IDs are consistent when the same environments
    are used across multiple policy runs.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create two separate databases to simulate different policy runs
        db1_path = Path(temp_dir) / "policy1.duckdb"
        db2_path = Path(temp_dir) / "policy2.duckdb"

        # Define the same simulation details for both runs
        sim_suite = "test_suite"
        sim_name = "test_sim"
        env_path = "env/test_env"

        # Simulate first policy run
        db1 = StatsDB(db1_path, mode="rwc")
        sim_id1 = db1.ensure_simulation_id(sim_name, sim_suite, env_path)

        # Create an episode linked to this simulation
        ep_id1 = db1.get_next_episode_id()
        db1.con.execute(
            """
            INSERT INTO episodes 
            (id, seed, map_w, map_h, step_count, started_at, finished_at, simulation_id)
            VALUES (?, 0, 10, 10, 100, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
            """,
            (ep_id1, sim_id1),
        )

        # Add agent metrics
        db1.con.execute(
            """
            INSERT INTO agent_metrics
            (episode_id, agent_id, metric, value)
            VALUES (?, 0, 'reward', 1.0)
            """,
            (ep_id1,),
        )

        # Add agent policies - THIS WAS MISSING
        db1.con.execute(
            """
            INSERT INTO policies (policy_key, policy_version)
            VALUES ('policy1', 1)
            """
        )

        db1.con.execute(
            """
            INSERT INTO agent_policies
            (episode_id, agent_id, policy_key, policy_version)
            VALUES (?, 0, 'policy1', 1)
            """,
            (ep_id1,),
        )

        # Simulate second policy run with the same environment
        db2 = StatsDB(db2_path, mode="rwc")
        sim_id2 = db2.ensure_simulation_id(sim_name, sim_suite, env_path)

        # Create an episode linked to this simulation
        ep_id2 = db2.get_next_episode_id()
        db2.con.execute(
            """
            INSERT INTO episodes
            (id, seed, map_w, map_h, step_count, started_at, finished_at, simulation_id)
            VALUES (?, 0, 10, 10, 100, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
            """,
            (ep_id2, sim_id2),
        )

        # Add agent metrics
        db2.con.execute(
            """
            INSERT INTO agent_metrics
            (episode_id, agent_id, metric, value)
            VALUES (?, 0, 'reward', 2.0)
            """,
            (ep_id2,),
        )

        # Add agent policies - THIS WAS MISSING
        db2.con.execute(
            """
            INSERT INTO policies (policy_key, policy_version)
            VALUES ('policy2', 1)
            """
        )

        db2.con.execute(
            """
            INSERT INTO agent_policies
            (episode_id, agent_id, policy_key, policy_version)
            VALUES (?, 0, 'policy2', 1)
            """,
            (ep_id2,),
        )

        # Verify that both runs created the same simulation ID
        assert sim_id1 == sim_id2
        assert sim_id1 == f"{sim_suite}:{sim_name}:{env_path}"

        # Now test merging these databases
        merged_path = Path(temp_dir) / "merged.duckdb"
        merged_db = StatsDB(merged_path, mode="rwc")

        # Merge both databases
        merged_db.merge_in(db1)
        merged_db.merge_in(db2)

        # Materialize the view
        merged_db.materialize_policy_simulations_view("reward")

        # Verify that the view has the data from both policies
        view_data = merged_db.con.execute("SELECT * FROM policy_simulations_reward").fetchall()

        # The view should have rows, and none should have 'unknown' for sim_env
        assert len(view_data) > 0

        # Check that no rows have 'unknown' for sim_suite or sim_env
        unknown_rows = merged_db.con.execute(
            """
            SELECT COUNT(*) FROM policy_simulations_reward 
            WHERE sim_suite = 'unknown' OR sim_env = 'unknown'
            """
        ).fetchone()[0]

        assert unknown_rows == 0, "Found rows with 'unknown' sim_suite or sim_env"

        # Verify simulation IDs match in the merged database
        sim_ids = merged_db.con.execute("SELECT DISTINCT simulation_id FROM episodes").fetchall()

        assert len(sim_ids) == 1, "Expected exactly one simulation ID in merged database"
        assert sim_ids[0][0] == sim_id1, "Simulation ID changed during merge"

        # Close all databases
        db1.close()
        db2.close()
        merged_db.close()


def test_simulation_scores():
    """Test the simulation_scores function."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / f"{uuid.uuid4().hex}.duckdb"
        db = StatsDB(db_path, mode="rwc")

        # Create a test simulation
        sim_id = db.ensure_simulation_id("test_sim", "test_suite", "env/test")

        # Create test episodes
        episodes = []
        for i in range(3):
            ep_id = db.get_next_episode_id()
            episodes.append(ep_id)
            db.con.execute(
                """
                INSERT INTO episodes 
                (id, seed, map_w, map_h, step_count, started_at, finished_at, simulation_id)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
                """,
                (ep_id, i, 10, 10, 100, sim_id),
            )

        # Add agent policies
        policy_key = "test_policy"
        policy_version = 1
        agent_map = {0: (policy_key, policy_version)}
        db._insert_agent_policies(episodes, agent_map)

        # Add metrics
        for i, ep_id in enumerate(episodes):
            db.con.execute(
                """
                INSERT INTO agent_metrics
                (episode_id, agent_id, metric, value)
                VALUES (?, ?, ?, ?)
                """,
                (ep_id, 0, "reward", 10.0 + i),
            )

        # Create the materialized view
        db.materialize_policy_simulations_view("reward")

        # Get scores
        scores = db.simulation_scores(policy_key, policy_version, "reward")

        # Verify results
        assert len(scores) == 1
        key = ("test_suite", "test_sim", "env/test")  # Use full environment path
        assert key in scores
        assert scores[key] == pytest.approx(11.0)  # Average of 10, 11, 12

        db.close()
