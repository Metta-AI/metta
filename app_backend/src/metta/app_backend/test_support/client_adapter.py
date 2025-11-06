"""Adapter to make StatsClient work with TestClient for tests."""

import fastapi.testclient
import httpx


class TestClientAdapter:
    """Adapter that makes TestClient work like httpx.AsyncClient for StatsClient."""

    def __init__(self, test_client: fastapi.testclient.TestClient):
        self.test_client = test_client
        self.base_url = test_client.base_url

    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Make a request using the TestClient synchronously."""
        # Remove headers we'll handle separately
        headers = kwargs.pop("headers", {})

        # Make the sync request
        response = self.test_client.request(method, url, headers=headers, **kwargs)

        # Convert to httpx.Response-like object
        return response


def create_test_stats_client(test_client: fastapi.testclient.TestClient, machine_token: str):
    """Create a StatsClient that works with TestClient."""
    import metta.app_backend.clients.stats_client

    stats_client = metta.app_backend.clients.stats_client.HttpStatsClient(
        backend_url=str(test_client.base_url), machine_token=machine_token
    )
    stats_client._http_client = TestClientAdapter(test_client)  # type: ignore

    return stats_client
