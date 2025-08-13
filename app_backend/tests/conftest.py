from typing import Dict
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
