"""Tests for Cogweb agent configuration routes."""


def test_get_agent_bucket_returns_config(monkeypatch, isolated_test_client):
    monkeypatch.setattr(
        "metta.app_backend.routes.agent_routes.cogweb_agent_bucket_uri",
        "s3://example-bucket/agent-prefix",
        raising=False,
    )
    monkeypatch.setattr(
        "metta.app_backend.routes.agent_routes.cogweb_agent_bucket_region",
        "us-west-2",
        raising=False,
    )

    response = isolated_test_client.get("/agents/bucket", headers={"X-Auth-Request-Email": "tester@example.com"})

    assert response.status_code == 200
    assert response.json() == {
        "bucket": "example-bucket",
        "prefix": "agent-prefix",
        "uri": "s3://example-bucket/agent-prefix",
        "region": "us-west-2",
    }


def test_get_agent_bucket_invalid_config(monkeypatch, isolated_test_client):
    monkeypatch.setattr(
        "metta.app_backend.routes.agent_routes.cogweb_agent_bucket_uri",
        "https://not-an-s3-uri",
        raising=False,
    )

    response = isolated_test_client.get("/agents/bucket", headers={"X-Auth-Request-Email": "tester@example.com"})

    assert response.status_code == 500
    assert "Invalid S3 URI" in response.json()["detail"]


def test_get_agent_bucket_requires_auth(monkeypatch, isolated_test_client):
    monkeypatch.setattr(
        "metta.app_backend.routes.agent_routes.cogweb_agent_bucket_uri",
        "s3://example-bucket/agent-prefix",
        raising=False,
    )

    response = isolated_test_client.get("/agents/bucket")

    assert response.status_code == 401
