# ruff: noqa: E402
# need this to import and call suppress_noisy_logs first
from metta.common.util.log_config import suppress_noisy_logs

suppress_noisy_logs()
from typing import Dict
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from testcontainers.postgres import PostgresContainer

from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.metta_repo import MettaRepo
from metta.app_backend.server import create_app
from metta.app_backend.test_support.client_adapter import create_test_stats_client
from metta.common.tests_support import docker_client_fixture, isolated_test_schema_uri

# Register the docker_client fixture
docker_client = docker_client_fixture()


def pytest_configure(config):
    """Configure test settings before any tests run (works with xdist workers)."""
    from metta.app_backend import config as app_config

    app_config.settings.RUN_MIGRATIONS = True
    app_config.settings.DEBUG_USER_EMAIL = "test@example.com"


# Skip all tests that use postgres_container; it is flaky
# def pytest_collection_modifyitems(config, items):
#     skip_pg = pytest.mark.skip(reason="postgres_container flaky")
#     for item in items:
#         if "postgres_container" in item.fixturenames:
#             item.add_marker(skip_pg)


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
    from metta.app_backend import config as app_config
    from metta.app_backend import database

    # Reset the engine singleton and point it at the test database
    database._engine = None
    database._session_factory = None
    app_config.settings.STATS_DB_URI = db_uri

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
    """Headers for authenticated requests (empty since auth is via debug_user_email)."""
    return {}


@pytest.fixture(scope="class")
def auth_headers() -> Dict[str, str]:
    """Authentication headers for requests (empty since auth is via debug_user_email)."""
    return {}


@pytest.fixture(scope="class")
def stats_client(test_client: TestClient) -> StatsClient:
    """Create a stats client for testing."""
    # Auth is handled via debug_user_email, no need for headers
    return create_test_stats_client(test_client, machine_token="dummy_token")


@pytest.fixture(autouse=True)
def mock_k8s_client(monkeypatch):
    """Prevent any accidental k8s API calls in tests."""
    from metta.app_backend.job_runner import dispatcher

    mock_client = MagicMock()
    monkeypatch.setattr(dispatcher, "get_k8s_client", lambda: mock_client)
    yield mock_client


@pytest.fixture(autouse=True)
def mock_dispatch_job(monkeypatch):
    def stub_dispatch(job):
        return f"mock-k8s-job-{job.id.hex[:8]}"

    monkeypatch.setattr("metta.app_backend.routes.job_routes.dispatch_job", stub_dispatch)


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
    # Auth is handled via debug_user_email, no need for headers
    return create_test_stats_client(isolated_test_client, machine_token="dummy_token")
