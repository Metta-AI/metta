from __future__ import annotations

import datetime
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from metta.eval.dashboard_data import PolicyEvalMetric, get_policy_eval_metrics
from metta.sim.simulation_stats_db import SimulationStatsDB


def _create_test_db_with_metrics(db_path: Path) -> Tuple[SimulationStatsDB, List[str], str]:
    db = SimulationStatsDB(db_path)

    sim_id = str(uuid.uuid4())
    policy_key = "test_policy"
    policy_version = 1

    # Create simulation
    db.con.execute(
        """
        INSERT INTO simulations (id, name, suite, env, policy_key, policy_version)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (sim_id, "test_sim", "test_suite", "test_env", policy_key, policy_version),
    )

    episodes: List[str] = []
    for i in range(3):  # Create 3 episodes
        ep_id = str(uuid.uuid4())
        episodes.append(ep_id)

        # Create episode
        db.con.execute(
            """
            INSERT INTO episodes (id, simulation_id, created_at, replay_url)
            VALUES (?, ?, ?, ?)
            """,
            (ep_id, sim_id, datetime.datetime.now(), f"http://replay/{ep_id}"),
        )

        # Create agent groups
        db.con.execute(
            """
            INSERT INTO agent_groups (episode_id, agent_id, group_id)
            VALUES (?, ?, ?), (?, ?, ?)
            """,
            (ep_id, 0, 1, ep_id, 1, 2),  # Two agents in different groups
        )

        # Create agent metrics
        metrics = [
            (ep_id, 0, "reward", 1.0 + i),
            (ep_id, 0, "score", 2.0 + i),
            (ep_id, 1, "reward", 1.5 + i),
            (ep_id, 1, "score", 2.5 + i),
        ]
        db.con.executemany(
            """
            INSERT INTO agent_metrics (episode_id, agent_id, metric, value)
            VALUES (?, ?, ?, ?)
            """,
            metrics,
        )

    db.con.commit()
    return db, episodes, sim_id


@pytest.fixture
def test_db():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / f"{uuid.uuid4().hex}.duckdb"
        db, eps, sid = _create_test_db_with_metrics(p)
        yield db, eps, sid
        db.close()


def test_get_policy_eval_metrics(test_db):
    db, _, _ = test_db
    policy_evals = get_policy_eval_metrics(db)

    # Should have one policy eval since we only created one simulation
    assert len(policy_evals) == 1
    policy_eval = policy_evals[0]

    # Check policy eval fields
    assert policy_eval.policy_key == "test_policy"
    assert policy_eval.policy_version == 1
    assert policy_eval.eval_name == "test_sim"
    assert policy_eval.suite == "test_suite"
    assert policy_eval.replay_url is not None
    assert "http://replay/" in policy_eval.replay_url

    assert policy_eval.group_num_agents["1"] == 3  # One agent per episode * 3 episodes
    assert policy_eval.group_num_agents["2"] == 3

    # Check metrics
    metrics_by_group: Dict[str, Dict[str, PolicyEvalMetric]] = {}
    for m in policy_eval.policy_eval_metrics:
        metrics_by_group[m.group_id] = metrics_by_group.get(m.group_id, {})
        metrics_by_group[m.group_id][m.metric] = m

    assert len(metrics_by_group) == 2  # Two agent groups

    # Group 1 metrics (agent 0)
    group1_metrics = {m.metric: m for m in metrics_by_group["1"].values()}
    assert len(group1_metrics) == 2  # reward and score
    assert group1_metrics["reward"].sum_value == pytest.approx(6)  # 1 + 2 + 3
    assert group1_metrics["score"].sum_value == pytest.approx(9)  # 2 + 3 + 4

    # Group 2 metrics (agent 1)
    group2_metrics = {m.metric: m for m in metrics_by_group["2"].values()}
    assert len(group2_metrics) == 2  # reward and score
    assert group2_metrics["reward"].sum_value == pytest.approx(7.5)  # 1.5 + 2.5 + 3.5
    assert group2_metrics["score"].sum_value == pytest.approx(10.5)  # 2.5 + 3.5 + 4.5
