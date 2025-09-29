import pytest
from fastapi.testclient import TestClient

from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.routes.score_routes import PolicyScoresRequest


@pytest.mark.slow
class TestScoreRoutes:
    def test_get_policy_scores_basic(
        self, test_client: TestClient, stats_client: StatsClient, create_test_data
    ) -> None:
        """Test aggregated metric stats (min, max, avg) for policies across evals/metrics."""
        # Create two policies in a single training run
        test_data = create_test_data("policy_scores_basic", num_policies=2)
        policies = test_data["policies"]
        epochs = test_data["epochs"]

        sim_suite = "agg_suite"
        env_name = "test_env"
        eval_name = f"{sim_suite}/{env_name}"

        # Policy 0: reward [10.0, 20.0], score [30.0, 50.0]
        stats_client.record_episode(
            agent_policies={0: policies[0].id},
            agent_metrics={0: {"reward": 10.0, "score": 30.0}},
            primary_policy_id=policies[0].id,
            stats_epoch=epochs[0].id,
            sim_suite=sim_suite,
            env_name=env_name,
            replay_url="https://example.com/replay/p0_e1",
        )
        stats_client.record_episode(
            agent_policies={0: policies[0].id},
            agent_metrics={0: {"reward": 20.0, "score": 50.0}},
            primary_policy_id=policies[0].id,
            stats_epoch=epochs[0].id,
            sim_suite=sim_suite,
            env_name=env_name,
            replay_url="https://example.com/replay/p0_e2",
        )

        # Policy 1: reward [40.0, 60.0], score [100.0, 200.0]
        stats_client.record_episode(
            agent_policies={0: policies[1].id},
            agent_metrics={0: {"reward": 40.0, "score": 100.0}},
            primary_policy_id=policies[1].id,
            stats_epoch=epochs[1].id,
            sim_suite=sim_suite,
            env_name=env_name,
            replay_url="https://example.com/replay/p1_e1",
        )
        stats_client.record_episode(
            agent_policies={0: policies[1].id},
            agent_metrics={0: {"reward": 60.0, "score": 200.0}},
            primary_policy_id=policies[1].id,
            stats_epoch=epochs[1].id,
            sim_suite=sim_suite,
            env_name=env_name,
            replay_url="https://example.com/replay/p1_e2",
        )

        request = PolicyScoresRequest(
            policy_ids=[p.id for p in policies],
            eval_names=[eval_name],
            metrics=["reward", "score"],
        )
        resp = stats_client.get_policy_scores(request)
        scores = resp.scores

        # Policy 0 expectations
        p0 = policies[0].id
        assert p0 in scores
        assert eval_name in scores[p0]
        assert "reward" in scores[p0][eval_name]
        assert "score" in scores[p0][eval_name]
        r0 = scores[p0][eval_name]["reward"]
        s0 = scores[p0][eval_name]["score"]
        assert r0.min == 10.0
        assert r0.max == 20.0
        assert abs(r0.avg - 15.0) < 1e-6
        assert s0.min == 30.0
        assert s0.max == 50.0
        assert abs(s0.avg - 40.0) < 1e-6

        # Policy 1 expectations
        p1 = policies[1].id
        assert p1 in scores
        assert eval_name in scores[p1]
        assert "reward" in scores[p1][eval_name]
        assert "score" in scores[p1][eval_name]
        r1 = scores[p1][eval_name]["reward"]
        s1 = scores[p1][eval_name]["score"]
        assert r1.min == 40.0
        assert r1.max == 60.0
        assert abs(r1.avg - 50.0) < 1e-6
        assert s1.min == 100.0
        assert s1.max == 200.0
        assert abs(s1.avg - 150.0) < 1e-6
