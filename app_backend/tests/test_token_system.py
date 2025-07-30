import pytest
from fastapi.testclient import TestClient

from tests.base_async_test import BaseAsyncTest


class TestTokenSystem(BaseAsyncTest):
    """Tests for the machine token system."""

    def test_create_token(self, test_client: TestClient) -> None:
        """Test creating a machine token."""
        response = test_client.post(
            "/tokens",
            json={"name": "test_token"},
            headers={"X-Auth-Request-Email": "test@example.com"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "token" in data
        assert len(data["token"]) > 0

    def test_list_tokens(self, test_client: TestClient) -> None:
        """Test listing machine tokens."""
        # First create a token
        create_response = test_client.post(
            "/tokens",
            json={"name": "list_test_token"},
            headers={"X-Auth-Request-Email": "test@example.com"},
        )
        assert create_response.status_code == 200

        # Then list tokens
        response = test_client.get(
            "/tokens",
            headers={"X-Auth-Request-Email": "test@example.com"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "tokens" in data
        assert len(data["tokens"]) >= 1

        # Check that our token is in the list
        token_names = [token["name"] for token in data["tokens"]]
        assert "list_test_token" in token_names

        # Check that expiration_time is present
        for token in data["tokens"]:
            assert "expiration_time" in token

    def test_delete_token(self, test_client: TestClient) -> None:
        """Test deleting a machine token."""
        # First create a token
        create_response = test_client.post(
            "/tokens",
            json={"name": "delete_test_token"},
            headers={"X-Auth-Request-Email": "test@example.com"},
        )
        assert create_response.status_code == 200

        # List tokens to get the token ID
        list_response = test_client.get(
            "/tokens",
            headers={"X-Auth-Request-Email": "test@example.com"},
        )
        assert list_response.status_code == 200
        tokens = list_response.json()["tokens"]

        # Find our token
        token_id = None
        for token in tokens:
            if token["name"] == "delete_test_token":
                token_id = token["id"]
                break
        assert token_id is not None

        # Delete the token
        delete_response = test_client.delete(
            f"/tokens/{token_id}",
            headers={"X-Auth-Request-Email": "test@example.com"},
        )
        assert delete_response.status_code == 200

        # Verify it's deleted
        list_response_after = test_client.get(
            "/tokens",
            headers={"X-Auth-Request-Email": "test@example.com"},
        )
        assert list_response_after.status_code == 200
        tokens_after = list_response_after.json()["tokens"]
        token_names_after = [token["name"] for token in tokens_after]
        assert "delete_test_token" not in token_names_after

    def test_token_authentication(self, test_client: TestClient) -> None:
        """Test using a machine token for authentication."""
        # Create a token
        create_response = test_client.post(
            "/tokens",
            json={"name": "auth_test_token"},
            headers={"X-Auth-Request-Email": "test@example.com"},
        )
        assert create_response.status_code == 200
        token = create_response.json()["token"]

        # Use the token to access a protected endpoint
        response = test_client.get(
            "/training-runs",
            headers={"X-Auth-Token": token},
        )
        assert response.status_code == 200

    def test_user_email_authentication(self, test_client: TestClient) -> None:
        """Test using user email for authentication."""
        response = test_client.get(
            "/training-runs",
            headers={"X-Auth-Request-Email": "test@example.com"},
        )
        assert response.status_code == 200

    def test_no_authentication_fails(self, test_client: TestClient) -> None:
        """Test that write requests without authentication fail."""
        # Test a write operation (POST) without authentication
        response = test_client.post(
            "/tokens",
            json={"name": "test_token"},
        )
        assert response.status_code == 401

    def test_invalid_token_fails(self, test_client: TestClient) -> None:
        """Test that invalid tokens fail authentication for write operations."""
        # Test a write operation (POST) with invalid token
        response = test_client.post(
            "/tokens",
            json={"name": "test_token"},
            headers={"X-Auth-Token": "invalid_token"},
        )
        assert response.status_code == 401

    def test_whoami_with_email(self, test_client: TestClient) -> None:
        """Test whoami endpoint with email authentication."""
        response = test_client.get(
            "/whoami",
            headers={"X-Auth-Request-Email": "test@example.com"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["user_email"] == "test@example.com"

    def test_whoami_with_token(self, test_client: TestClient) -> None:
        """Test whoami endpoint with token authentication."""
        # Create a token
        create_response = test_client.post(
            "/tokens",
            json={"name": "whoami_test_token"},
            headers={"X-Auth-Request-Email": "test@example.com"},
        )
        assert create_response.status_code == 200
        token = create_response.json()["token"]

        # Use the token with whoami
        response = test_client.get(
            "/whoami",
            headers={"X-Auth-Token": token},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["user_email"] == "test@example.com"

    def test_whoami_no_auth(self, test_client: TestClient) -> None:
        """Test whoami endpoint without authentication."""
        response = test_client.get("/whoami")
        assert response.status_code == 200
        data = response.json()
        assert data["user_email"] == "unknown"


if __name__ == "__main__":
    # Simple test runner for debugging
    pytest.main([__file__, "-v", "-s"])
