"""Adapter to make StatsClient work with TestClient for tests."""

import httpx
from fastapi.testclient import TestClient


class TestClientAdapter:
    """Adapter that makes TestClient work like httpx.AsyncClient for StatsClient."""

    def __init__(self, test_client: TestClient):
        self.test_client = test_client
        self.base_url = test_client.base_url

    async def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Make a request using the TestClient synchronously."""
        # Remove headers we'll handle separately
        headers = kwargs.pop("headers", {})

        # Make the sync request
        response = self.test_client.request(method, url, headers=headers, **kwargs)

        # Convert to httpx.Response-like object
        return response

    async def aclose(self):
        """No-op for test client."""
        pass


def create_test_stats_client(test_client: TestClient, machine_token: str):
    """Create a StatsClient that works with TestClient."""
    from metta.app_backend.clients.stats_client import AsyncStatsClient, StatsClient

    # Create the async client with proper URL
    async_client = AsyncStatsClient(backend_url=str(test_client.base_url), machine_token=machine_token)

    # Replace the http client with our adapter
    async_client._http_client = TestClientAdapter(test_client)

    # Create the sync wrapper
    stats_client = StatsClient(backend_url=str(test_client.base_url), machine_token=machine_token)
    stats_client._async_client = async_client

    return stats_client
