import asyncio
import socket
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict
from unittest import mock

import pytest
import pytest_asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient
from testcontainers.postgres import PostgresContainer

from app_backend.tests.http_env import HttpAsyncStatsClientEnv, HttpEvalTaskClientEnv, TestClientStatsEnv
from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.metta_repo import MettaRepo
from metta.app_backend.server import create_app
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


# http_stats_client fixture removed due to sync/async compatibility issues


@pytest.fixture
def stats_client(test_client_stats_env: TestClientStatsEnv) -> StatsClient:
    """Create a stats client for testing using TestClient environment."""
    return test_client_stats_env.make_client()


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
def isolated_stats_client(isolated_test_client_stats_env: TestClientStatsEnv) -> StatsClient:
    """Create a stats client with isolated database for testing."""
    return isolated_test_client_stats_env.make_client()


@pytest.fixture
def create_test_data(stats_client: StatsClient):
    def _create(
        run_name: str,
        num_policies: int = 2,
        create_run_free_policies: int = 0,
        overriding_stats_client: StatsClient | None = None,
    ) -> dict[str, Any]:
        use_stats_client = overriding_stats_client or stats_client
        data: dict[str, Any] = {"policies": [], "policy_names": []}

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

        return data

    return _create


@pytest.fixture
def async_create_test_data(http_async_stats_env: HttpAsyncStatsClientEnv):
    async def _create(
        run_name: str,
        num_policies: int = 2,
        create_run_free_policies: int = 0,
        overriding_async_stats_client=None,
    ) -> dict[str, Any]:
        use_stats_client = overriding_async_stats_client or http_async_stats_env.make_client()
        data: dict[str, Any] = {"policies": [], "policy_names": []}

        if num_policies > 0:
            timestamp = int(time.time() * 1_000_000)
            training_run = await use_stats_client.create_training_run(
                name=f"{run_name}_{timestamp}",
                attributes={"environment": "test_env", "algorithm": "test_alg"},
                url="https://example.com/run",
                tags=["test_tag", "scorecard_test"],
            )

            epoch1 = await use_stats_client.create_epoch(
                run_id=training_run.id,
                start_training_epoch=0,
                end_training_epoch=100,
                attributes={"learning_rate": "0.001"},
            )
            epoch2 = await use_stats_client.create_epoch(
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
                policy = await use_stats_client.create_policy(
                    name=policy_name,
                    description=f"Test policy {i} for {run_name}",
                    epoch_id=epoch.id,
                )
                data["policies"].append(policy)
                data["policy_names"].append(policy_name)

        timestamp = int(time.time() * 1_000_000)
        for i in range(create_run_free_policies):
            policy_name = f"runfree_policy_{run_name}_{i}_{timestamp}"
            policy = await use_stats_client.create_policy(
                name=policy_name,
                description=f"Run-free test policy {i} for {run_name}",
                epoch_id=None,
            )
            data["policies"].append(policy)
            data["policy_names"].append(policy_name)

        return data

    return _create


@pytest.fixture
def isolated_async_create_test_data(isolated_http_async_stats_env: HttpAsyncStatsClientEnv):
    async def _create(
        run_name: str,
        num_policies: int = 2,
        create_run_free_policies: int = 0,
        overriding_async_stats_client=None,
    ) -> dict[str, Any]:
        use_stats_client = overriding_async_stats_client or isolated_http_async_stats_env.make_client()
        data: dict[str, Any] = {"policies": [], "policy_names": []}

        if num_policies > 0:
            timestamp = int(time.time() * 1_000_000)
            training_run = await use_stats_client.create_training_run(
                name=f"{run_name}_{timestamp}",
                attributes={"environment": "test_env", "algorithm": "test_alg"},
                url="https://example.com/run",
                tags=["test_tag", "scorecard_test"],
            )

            epoch1 = await use_stats_client.create_epoch(
                run_id=training_run.id,
                start_training_epoch=0,
                end_training_epoch=100,
                attributes={"learning_rate": "0.001"},
            )
            epoch2 = await use_stats_client.create_epoch(
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
                policy = await use_stats_client.create_policy(
                    name=policy_name,
                    description=f"Test policy {i} for {run_name}",
                    epoch_id=epoch.id,
                )
                data["policies"].append(policy)
                data["policy_names"].append(policy_name)

        timestamp = int(time.time() * 1_000_000)
        for i in range(create_run_free_policies):
            policy_name = f"runfree_policy_{run_name}_{i}_{timestamp}"
            policy = await use_stats_client.create_policy(
                name=policy_name,
                description=f"Run-free test policy {i} for {run_name}",
                epoch_id=None,
            )
            data["policies"].append(policy)
            data["policy_names"].append(policy_name)

        return data

    return _create


@pytest.fixture
def isolated_async_record_episodes(isolated_http_async_stats_env: HttpAsyncStatsClientEnv):
    async def _record(
        test_data: dict,
        eval_category: str,
        env_names: list[str],
        metric_values: dict[str, float],
        overriding_async_stats_client=None,
    ) -> None:
        use_stats_client = overriding_async_stats_client or isolated_http_async_stats_env.make_client()
        epochs = test_data.get("epochs", [])
        for i, policy in enumerate(test_data["policies"]):
            epoch_id = epochs[i % len(epochs)].id if epochs and i < len(epochs) else None
            for env_name in env_names:
                sim_name = f"{eval_category}/{env_name}"
                metric_key = f"policy_{i}_{env_name}"
                metric_value = metric_values.get(metric_key, 50.0)
                await use_stats_client.record_episode(
                    agent_policies={0: policy.id},
                    agent_metrics={0: {"reward": metric_value}},
                    primary_policy_id=policy.id,
                    stats_epoch=epoch_id,
                    sim_name=sim_name,
                    env_label=env_name,
                    replay_url=f"https://example.com/replay/{policy.id}/{sim_name}",
                )

    return _record


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
        epochs = test_data.get("epochs", [])
        for i, policy in enumerate(test_data["policies"]):
            epoch_id = epochs[i % len(epochs)].id if epochs and i < len(epochs) else None
            for env_name in env_names:
                sim_name = f"{eval_category}/{env_name}"
                metric_key = f"policy_{i}_{env_name}"
                metric_value = metric_values.get(metric_key, 50.0)
                use_stats_client.record_episode(
                    agent_policies={0: policy.id},
                    agent_metrics={0: {"reward": metric_value}},
                    primary_policy_id=policy.id,
                    stats_epoch=epoch_id,
                    sim_name=sim_name,
                    env_label=env_name,
                    replay_url=f"https://example.com/replay/{policy.id}/{sim_name}",
                )

    return _record


@pytest.fixture
def async_record_episodes(http_async_stats_env: HttpAsyncStatsClientEnv):
    async def _record(
        test_data: dict,
        eval_category: str,
        env_names: list[str],
        metric_values: dict[str, float],
        overriding_async_stats_client=None,
    ) -> None:
        use_stats_client = overriding_async_stats_client or http_async_stats_env.make_client()
        epochs = test_data.get("epochs", [])
        for i, policy in enumerate(test_data["policies"]):
            epoch_id = epochs[i % len(epochs)].id if epochs and i < len(epochs) else None
            for env_name in env_names:
                sim_name = f"{eval_category}/{env_name}"
                metric_key = f"policy_{i}_{env_name}"
                metric_value = metric_values.get(metric_key, 50.0)
                await use_stats_client.record_episode(
                    agent_policies={0: policy.id},
                    agent_metrics={0: {"reward": metric_value}},
                    primary_policy_id=policy.id,
                    stats_epoch=epoch_id,
                    sim_name=sim_name,
                    env_label=env_name,
                    replay_url=f"https://example.com/replay/{policy.id}/{sim_name}",
                )

    return _record


def _find_free_port() -> int:
    """Find a free port for the HTTP server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


@asynccontextmanager
async def _http_server(test_app: FastAPI):
    """Start a real HTTP server for testing."""
    port = _find_free_port()

    config = uvicorn.Config(test_app, host="127.0.0.1", port=port, log_level="critical")
    server = uvicorn.Server(config)

    # Start server in background
    task = asyncio.create_task(server.serve())

    # Wait for server to start
    await asyncio.sleep(0.2)

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


@pytest_asyncio.fixture
async def http_eval_task_env(test_app: FastAPI) -> AsyncGenerator[HttpEvalTaskClientEnv, Any]:
    """Create an HTTP environment for eval task client tests."""
    async with _http_server(test_app) as base_url:
        async with AsyncClient(base_url=base_url) as tmp:
            r = await tmp.post(
                "/tokens",
                json={"name": "orchestrator_test_token", "permissions": ["read", "write"]},
                headers={"X-Auth-Request-Email": "test_user@example.com"},
            )
            r.raise_for_status()
            token = r.json()["token"]
        env = HttpEvalTaskClientEnv(base_url=base_url, token=token)
        try:
            yield env
        finally:
            await env.aclose_all()


@pytest_asyncio.fixture
async def isolated_http_eval_task_env(isolated_test_app: FastAPI) -> AsyncGenerator[HttpEvalTaskClientEnv, Any]:
    """Create an HTTP environment for isolated database tests."""
    async with _http_server(isolated_test_app) as base_url:
        async with AsyncClient(base_url=base_url) as tmp:
            r = await tmp.post(
                "/tokens",
                json={"name": "isolated_test_token", "permissions": ["read", "write"]},
                headers={"X-Auth-Request-Email": "test_user@example.com"},
            )
            r.raise_for_status()
            token = r.json()["token"]
        env = HttpEvalTaskClientEnv(base_url=base_url, token=token)
        try:
            yield env
        finally:
            await env.aclose_all()


@pytest_asyncio.fixture
async def http_async_stats_env(test_app: FastAPI) -> AsyncGenerator[HttpAsyncStatsClientEnv, Any]:
    """Create an HTTP environment for async stats client tests."""
    async with _http_server(test_app) as base_url:
        async with AsyncClient(base_url=base_url) as tmp:
            r = await tmp.post(
                "/tokens",
                json={"name": "http_async_stats_token", "permissions": ["read", "write"]},
                headers={"X-Auth-Request-Email": "test_user@example.com"},
            )
            r.raise_for_status()
            token = r.json()["token"]
        env = HttpAsyncStatsClientEnv(base_url=base_url, token=token)
        try:
            yield env
        finally:
            await env.aclose_all()


@pytest_asyncio.fixture
async def isolated_http_async_stats_env(isolated_test_app: FastAPI) -> AsyncGenerator[HttpAsyncStatsClientEnv, Any]:
    """Create an HTTP environment for isolated async stats client tests."""
    async with _http_server(isolated_test_app) as base_url:
        async with AsyncClient(base_url=base_url) as tmp:
            r = await tmp.post(
                "/tokens",
                json={"name": "isolated_http_async_stats_token", "permissions": ["read", "write"]},
                headers={"X-Auth-Request-Email": "test_user@example.com"},
            )
            r.raise_for_status()
            token = r.json()["token"]
        env = HttpAsyncStatsClientEnv(base_url=base_url, token=token)
        try:
            yield env
        finally:
            await env.aclose_all()


@pytest.fixture
def test_client_stats_env(test_client: TestClient) -> TestClientStatsEnv:
    """Create a TestClient environment for stats client tests."""
    # Create a token first using TestClient
    token_response = test_client.post(
        "/tokens",
        json={"name": "test_client_stats_token", "permissions": ["read", "write"]},
        headers={"X-Auth-Request-Email": "test_user@example.com"},
    )
    assert token_response.status_code == 200, f"Failed to create token: {token_response.text}"
    token = token_response.json()["token"]

    return TestClientStatsEnv(test_client, token)


@pytest.fixture(scope="function")
def isolated_test_client_stats_env(isolated_test_client: TestClient) -> TestClientStatsEnv:
    """Create a TestClient environment for isolated stats client tests."""
    # Create a token first using TestClient
    token_response = isolated_test_client.post(
        "/tokens",
        json={"name": "isolated_test_client_stats_token", "permissions": ["read", "write"]},
        headers={"X-Auth-Request-Email": "test_user@example.com"},
    )
    assert token_response.status_code == 200, f"Failed to create token: {token_response.text}"
    token = token_response.json()["token"]

    return TestClientStatsEnv(isolated_test_client, token)


@pytest.fixture
def orchestrator_test_policy_id(stats_client: StatsClient) -> uuid.UUID:
    """Create a test policy specifically for orchestrator tests."""
    training_run = stats_client.create_training_run(
        name=f"test_orchestrator_run_{uuid.uuid4().hex[:8]}",
        attributes={"test": "orchestrator"},
    )

    epoch = stats_client.create_epoch(
        run_id=training_run.id,
        start_training_epoch=0,
        end_training_epoch=100,
    )

    policy = stats_client.create_policy(
        name=f"test_orchestrator_policy_{uuid.uuid4().hex[:8]}",
        description="Test policy for orchestrator tests",
        epoch_id=epoch.id,
    )

    return policy.id
