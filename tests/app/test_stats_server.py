import pytest
from fastapi.testclient import TestClient
from testcontainers.postgres import PostgresContainer

from metta.app.metta_repo import MettaRepo
from metta.app.server import create_app
from metta.app.stats_client import StatsClient


class TestStatsServerSimple:
    """Simplified end-to-end tests for the stats server."""

    @pytest.fixture(scope="class")
    def postgres_container(self):
        """Create a PostgreSQL container for testing."""
        container = PostgresContainer(
            image="postgres:17",
            username="test_user",
            password="test_password",
            dbname="test_db",
            driver=None,
        )
        container.start()
        yield container
        container.stop()

    @pytest.fixture(scope="class")
    def db_uri(self, postgres_container):
        """Get the database URI for the test container."""
        return postgres_container.get_connection_url()

    @pytest.fixture(scope="class")
    def stats_repo(self, db_uri):
        """Create a StatsRepo instance with the test database."""
        return MettaRepo(db_uri)

    @pytest.fixture(scope="class")
    def test_app(self, stats_repo):
        """Create a test FastAPI app with dependency injection."""
        return create_app(stats_repo)

    @pytest.fixture(scope="class")
    def test_client(self, test_app):
        """Create a test client."""
        return TestClient(test_app)

    @pytest.fixture(scope="class")
    def stats_client(self, test_client):
        """Create a stats client for testing."""
        return StatsClient(test_client)

    def test_complete_workflow(self, stats_client: StatsClient):
        """Test the complete end-to-end workflow."""

        # 1. Create a training run
        training_run = stats_client.create_training_run(
            name="test_training_run",
            user_id="test_user",
            attributes={"environment": "test_env", "algorithm": "test_alg"},
            url="https://example.com/run",
        )
        assert training_run.id is not None

        # 2. Create an epoch
        epoch = stats_client.create_epoch(
            run_id=training_run.id,
            start_training_epoch=0,
            end_training_epoch=100,
            attributes={"learning_rate": "0.001", "batch_size": "32"},
        )
        assert epoch.id is not None

        # 3. Create a policy
        policy = stats_client.create_policy(
            name="test_policy_v1",
            description="Test policy for end-to-end testing",
            url="https://example.com/policy",
            epoch_id=epoch.id,
        )
        assert policy.id is not None

        # 4. Create another policy for agent diversity
        policy2 = stats_client.create_policy(name="test_policy_v2", description="Second test policy", epoch_id=epoch.id)
        assert policy2.id is not None

        # 5. Record an episode
        episode = stats_client.record_episode(
            agent_policies={0: policy.id, 1: policy2.id},
            agent_metrics={
                0: {"reward": 100.5, "steps": 50.0, "success_rate": 0.8},
                1: {"reward": 85.2, "steps": 45.0, "success_rate": 0.7},
            },
            primary_policy_id=policy.id,
            stats_epoch=epoch.id,
            eval_name="test_evaluation",
            simulation_suite="test_suite",
            replay_url="https://example.com/replay",
            attributes={"episode_length": 100, "difficulty": "medium"},
        )
        assert episode.id is not None

        # 6. Test policy ID lookup
        policy_ids = stats_client.get_policy_ids(["test_policy_v1", "test_policy_v2"])
        assert policy_ids.policy_ids["test_policy_v1"] == policy.id
        assert policy_ids.policy_ids["test_policy_v2"] == policy2.id

    def test_multiple_episodes(self, stats_client):
        """Test recording multiple episodes."""

        # Create a training run
        training_run = stats_client.create_training_run(name="multi_episode_test", user_id="test_user")

        # Create an epoch
        epoch = stats_client.create_epoch(run_id=training_run.id, start_training_epoch=0, end_training_epoch=10)

        # Create a policy
        policy = stats_client.create_policy(name="multi_episode_policy", epoch_id=epoch.id)

        # Record multiple episodes
        episode_ids = []
        for i in range(5):
            episode = stats_client.record_episode(
                agent_policies={0: policy.id},
                agent_metrics={0: {"reward": float(i * 10), "steps": float(i * 5)}},
                primary_policy_id=policy.id,
                stats_epoch=epoch.id,
                eval_name=f"episode_{i}",
            )
            episode_ids.append(episode.id)
            assert episode.id is not None

        # Verify all episodes have different IDs
        assert len(set(episode_ids)) == 5

    def test_policy_id_lookup_empty(self, stats_client):
        """Test policy ID lookup with empty list."""
        policy_ids = stats_client.get_policy_ids([])
        assert policy_ids.policy_ids == {}

    def test_policy_id_lookup_nonexistent(self, stats_client):
        """Test policy ID lookup for non-existent policies."""
        policy_ids = stats_client.get_policy_ids(["nonexistent_policy"])
        assert policy_ids.policy_ids == {}


if __name__ == "__main__":
    # Simple test runner for debugging
    pytest.main([__file__, "-v", "-s"])
