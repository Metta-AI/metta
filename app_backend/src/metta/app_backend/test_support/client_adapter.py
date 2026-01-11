"""Adapter to make StatsClient work with TestClient for tests."""

from typing import cast

import httpx
from fastapi.testclient import TestClient

import metta.app_backend.config
from metta.app_backend.auth import User


def get_fake_softmax_user() -> User:
    return User(id="debug_user_id", email="softmax_user@example.com", is_softmax_team_member=True)


def get_user_headers(user: User) -> dict[str, str]:
    return {
        "X-User-Id": user.id,
        "X-User-Email": user.email,
        "X-User-Is-Softmax-Team-Member": str(user.is_softmax_team_member),
        "X-Auth-Secret": metta.app_backend.config.settings.OBSERVATORY_AUTH_SECRET or "",
    }


class TestClientAdapter:
    """Adapter that makes TestClient work like httpx.AsyncClient for StatsClient."""

    def __init__(self, test_client: TestClient, user: User | None = None):
        self.test_client = test_client
        self.base_url = test_client.base_url
        self.user = user

    @classmethod
    def with_user(cls, test_client: TestClient, user: User | None) -> httpx.Client:
        return cast(httpx.Client, TestClientAdapter(test_client, user=user))

    @classmethod
    def with_softmax_user(cls, test_client: TestClient) -> httpx.Client:
        return cls.with_user(test_client, get_fake_softmax_user())

    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Make a request using the TestClient synchronously."""
        # Remove headers we'll handle separately
        headers = kwargs.pop("headers", {})

        if self.user:
            headers.update(get_user_headers(self.user))

        # Make the sync request
        response = self.test_client.request(method, url, headers=headers, **kwargs)

        return response

    def post(self, url: str, **kwargs) -> httpx.Response:
        """Make a POST request using the TestClient synchronously."""
        return self.request("POST", url, **kwargs)

    def get(self, url: str, **kwargs) -> httpx.Response:
        """Make a GET request using the TestClient synchronously."""
        return self.request("GET", url, **kwargs)


def create_test_stats_client(test_client: TestClient, user: User | None = None):
    """Create a StatsClient that works with TestClient."""
    from metta.app_backend.clients.stats_client import StatsClient

    stats_client = StatsClient(backend_url=str(test_client.base_url), machine_token="dummy")
    stats_client._http_client = TestClientAdapter.with_user(test_client, user)

    return stats_client
