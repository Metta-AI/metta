import asyncio

from fastapi.testclient import TestClient
from httpx import AsyncClient

from metta.app_backend.clients.eval_task_client import EvalTaskClient
from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.test_support.client_adapter import create_test_stats_client


class HttpEvalTaskClientEnv:
    """Environment for HTTP-based eval task client tests."""

    def __init__(self, base_url: str, token: str):
        self.base_url = base_url
        self.token = token
        self._httpx_clients = []

    def make_client(self) -> EvalTaskClient:
        """Create a new EvalTaskClient instance."""
        client = EvalTaskClient.__new__(EvalTaskClient)
        httpx_client = AsyncClient(base_url=self.base_url)
        client._http_client = httpx_client
        client._machine_token = self.token
        self._httpx_clients.append(httpx_client)
        return client

    async def aclose_all(self):
        await asyncio.gather(*(cl.aclose() for cl in self._httpx_clients), return_exceptions=True)


class TestClientStatsEnv:
    """Environment for TestClient-based stats client tests."""

    def __init__(self, test_client: TestClient, token: str):
        self.test_client = test_client
        self.token = token
        self._clients = []

    def make_client(self) -> StatsClient:
        """Create a new StatsClient instance using TestClient."""
        client = create_test_stats_client(self.test_client, self.token)
        self._clients.append(client)
        return client

    def close_all(self):
        """Clean up all clients if needed."""
        # TestClient cleanup happens automatically
        pass


class HttpAsyncStatsClientEnv:
    """Environment for HTTP-based async stats client tests."""

    def __init__(self, base_url: str, token: str):
        self.base_url = base_url
        self.token = token
        self._httpx_clients = []

    def make_client(self):
        """Create a new AsyncStatsClient instance."""
        from metta.app_backend.clients.stats_client import AsyncStatsClient

        client = AsyncStatsClient.__new__(AsyncStatsClient)
        httpx_client = AsyncClient(base_url=self.base_url)
        client._http_client = httpx_client
        client._machine_token = self.token
        self._httpx_clients.append(httpx_client)
        return client

    async def aclose_all(self):
        """Clean up all HTTP clients."""
        await asyncio.gather(*(cl.aclose() for cl in self._httpx_clients), return_exceptions=True)
