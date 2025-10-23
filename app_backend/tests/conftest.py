import time
from typing import Any, Dict
from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from testcontainers.postgres import PostgresContainer

from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.metta_repo import MettaRepo
from metta.app_backend.server import create_app
from metta.app_backend.test_support import create_test_stats_client
from metta.common.test_support import docker_client_fixture, isolated_test_schema_uri

# Register the docker_client fixture
docker_client = docker_client_fixture()


@pytest.fixture(scope="session", autouse=True)
def mock_debug_user_email():
    """Mock debug_user_email for all tests to prevent local env interference."""
    with mock.patch("metta.app_backend.config.debug_user_email", None):
        yield


# Mark all tests that use postgres_container as flaky
def pytest_collection_modifyitems(config, items):
    skip_pg = pytest.mark.skip(reason="postgres_container flaky")
    for item in items:
        if "postgres_container" in item.fixturenames:
            item.add_marker(skip_pg)


@pytest.fixture(scope="class")
def postgres_container():
    """Create a PostgreSQL container for testing."""
    try:
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
    except Exception as e:
        pytest.skip(f"Failed to start PostgreSQL container: {e}")


@pytest.fixture(scope="class")
def db_uri(postgres_container: PostgresContainer) -> str:
    """Get the database URI for the test container."""
    return postgres_container.get_connection_url()


@pytest.fixture(scope="class")
def stats_repo(db_uri: str) -> MettaRepo:
    """Create a MettaRepo instance with the test database."""
    return MettaRepo(db_uri)


@pytest.fixture(scope="class")
def test_app(stats_repo: MettaRepo) -> FastAPI:
    """Create a test FastAPI app with dependency injection."""
    return create_app(stats_repo)


@pytest.fixture(scope="class")
def test_client(test_app: FastAPI) -> TestClient:
    """Create a test client."""
    return TestClient(test_app)


@pytest.fixture(scope="class")
def test_user_headers() -> Dict[str, str]:
    """Headers for authenticated requests."""
    return {"X-Auth-Request-Email": "test_user@example.com"}


@pytest.fixture(scope="class")
def auth_headers() -> Dict[str, str]:
    """Authentication headers for requests (alias for test_user_headers)."""
    return {"X-Auth-Request-Email": "test@example.com"}


@pytest.fixture(scope="class")
def stats_client(test_client: TestClient) -> StatsClient:
    """Create a stats client for testing."""
    # First create a machine token
    token_response = test_client.post(
        "/tokens",
        json={"name": "test_token", "permissions": ["read", "write"]},
        headers={"X-Auth-Request-Email": "test_user@example.com"},
    )
    assert token_response.status_code == 200, f"Failed to create token: {token_response.text}"
    token = token_response.json()["token"]

    # Create stats client that works with TestClient
    return create_test_stats_client(test_client, machine_token=token)


# Isolated fixtures for function-scoped testing
@pytest.fixture(scope="function")
def isolated_db_context(db_uri: str) -> str:
    """Create an isolated schema context for a single test."""
    schema_uri = isolated_test_schema_uri(db_uri)
    return schema_uri


@pytest.fixture(scope="function")
def isolated_stats_repo(isolated_db_context: str) -> MettaRepo:
    """Create a MettaRepo instance with an isolated schema."""
    return MettaRepo(isolated_db_context)


@pytest.fixture(scope="function")
def isolated_test_app(isolated_stats_repo: MettaRepo) -> FastAPI:
    """Create a test FastAPI app with isolated database."""
    return create_app(isolated_stats_repo)


@pytest.fixture(scope="function")
def isolated_test_client(isolated_test_app: FastAPI) -> TestClient:
    """Create a test client with isolated database."""
    return TestClient(isolated_test_app)


@pytest.fixture(scope="function")
def isolated_stats_client(isolated_test_client: TestClient) -> StatsClient:
    """Create a stats client with isolated database for testing."""
    # First create a machine token
    token_response = isolated_test_client.post(
        "/tokens",
        json={"name": "test_token", "permissions": ["read", "write"]},
        headers={"X-Auth-Request-Email": "test_user@example.com"},
    )
    assert token_response.status_code == 200, f"Failed to create token: {token_response.text}"
    token = token_response.json()["token"]

    # Create stats client that works with TestClient
    return create_test_stats_client(isolated_test_client, machine_token=token)


@pytest.fixture
def create_test_data(stats_client: StatsClient):
    def _create(
        run_name: str,
        num_policies: int = 2,
        create_run_free_policies: int = 0,
        overriding_stats_client: StatsClient | None = None,
    ) -> dict[str, Any]:
        use_stats_client = overriding_stats_client or stats_client
        data: dict[str, Any] = {"policies": [], "policy_names": [], "policy_epoch_ids": []}

        if num_policies > 0:
            timestamp = int(time.time() * 1_000_000)
            training_run = use_stats_client.create_training_run(
                name=f"{run_name}_{timestamp}",
                attributes={"environment": "test_env", "algorithm": "test_alg"},
                url="https://example.com/run",
                tags=["test_tag", "scorecard_test"],
            )

            epoch1 = use_stats_client.create_epoch(
                run_id=training_run.id,
                start_training_epoch=0,
                end_training_epoch=100,
                attributes={"learning_rate": "0.001"},
            )
            epoch2 = use_stats_client.create_epoch(
                run_id=training_run.id,
                start_training_epoch=100,
                end_training_epoch=200,
                attributes={"learning_rate": "0.0005"},
            )

            data["training_run"] = training_run
            data["epochs"] = [epoch1, epoch2]

            timestamp = int(time.time() * 1_000_000)
            for i in range(num_policies):
                epoch = epoch1 if i == 0 else epoch2
                policy_name = f"policy_{run_name}_{i}_{timestamp}"
                policy = use_stats_client.create_policy(
                    name=policy_name,
                    description=f"Test policy {i} for {run_name}",
                    epoch_id=epoch.id,
                )
                data["policies"].append(policy)
                data["policy_names"].append(policy_name)
                data["policy_epoch_ids"].append(epoch.id)

        timestamp = int(time.time() * 1_000_000)
        for i in range(create_run_free_policies):
            policy_name = f"runfree_policy_{run_name}_{i}_{timestamp}"
            policy = use_stats_client.create_policy(
                name=policy_name,
                description=f"Run-free test policy {i} for {run_name}",
                epoch_id=None,
            )
            data["policies"].append(policy)
            data["policy_names"].append(policy_name)
            data["policy_epoch_ids"].append(None)

        return data

    return _create


@pytest.fixture
def record_episodes(stats_client: StatsClient):
    def _record(
        test_data: dict,
        eval_category: str,
        env_names: list[str],
        metric_values: dict[str, float],
        overriding_stats_client: StatsClient | None = None,
    ) -> None:
        use_stats_client = overriding_stats_client or stats_client
        policy_epoch_ids: list[Any] = test_data.get("policy_epoch_ids", [])
        epochs = test_data.get("epochs", [])
        for i, policy in enumerate(test_data["policies"]):
            epoch_id = None
            if i < len(policy_epoch_ids):
                epoch_id = policy_epoch_ids[i]
            elif epochs:
                epoch_id = epochs[i % len(epochs)].id
            for env_name in env_names:
                metric_key = f"policy_{i}_{env_name}"
                metric_value = metric_values.get(metric_key, 50.0)
                use_stats_client.record_episode(
                    agent_policies={0: policy.id},
                    agent_metrics={0: {"reward": metric_value}},
                    primary_policy_id=policy.id,
                    stats_epoch=epoch_id,
                    sim_suite=eval_category,
                    env_name=env_name,
                    replay_url=f"https://example.com/replay/{policy.id}/{env_name}",
                )

    return _record
