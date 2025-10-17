"""Unit tests for the Cogweb client helper."""

from __future__ import annotations

from typing import Any

from cogweb.cogweb_client import AgentBucketInfo, CogwebClient


class _FakeResponse:
    def __init__(self, payload: dict[str, str]) -> None:
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self) -> None:  # pragma: no cover - nothing to do
        return

    def json(self) -> dict[str, str]:
        return self._payload


def test_get_agent_bucket(monkeypatch):
    CogwebClient.clear_cache()

    captured: dict[str, Any] = {}

    def fake_get(url: str, headers: dict[str, str] | None = None):
        captured["url"] = url
        captured["headers"] = headers
        return _FakeResponse(
            {
                "bucket": "example-bucket",
                "prefix": "agents",
                "uri": "s3://example-bucket/agents",
                "region": "us-west-2",
            }
        )

    monkeypatch.setattr("cogweb.cogweb_client.httpx.get", fake_get)

    client = CogwebClient.get_client(base_url="https://api.example.com", auth_token="token-123")
    bucket = client.get_agent_bucket()

    assert isinstance(bucket, AgentBucketInfo)
    assert bucket.bucket == "example-bucket"
    assert bucket.prefix == "agents"
    assert bucket.region == "us-west-2"
    assert captured["url"] == "https://api.example.com/agents/bucket"
    assert captured["headers"] == {"X-Auth-Token": "token-123"}
