# ruff: noqa: E402
# need this to import and call suppress_noisy_logs first
from metta.common.util.log_config import suppress_noisy_logs

suppress_noisy_logs()

"""Pytest configuration for gitta tests."""

from unittest.mock import MagicMock, Mock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_github_api():
    """Auto-mock all GitHub API calls to prevent rate limiting in tests.

    This fixture automatically applies to all gitta tests to ensure
    no real HTTP calls are made to GitHub's API.
    """
    with (
        patch("httpx.get") as mock_get,
        patch("httpx.post") as mock_post,
        patch("httpx.AsyncClient") as mock_async_client,
    ):
        # Mock successful GET response (empty PR list by default)
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # Mock successful POST response
        mock_post_response = Mock()
        mock_post_response.json.return_value = {"state": "success"}
        mock_post_response.raise_for_status = Mock()
        mock_post.return_value = mock_post_response

        # Mock async client
        mock_async_instance = MagicMock()
        mock_async_client.return_value.__aenter__.return_value = mock_async_instance

        yield {
            "get": mock_get,
            "post": mock_post,
            "async_client": mock_async_client,
        }


@pytest.fixture
def clear_pr_cache():
    """Clear the PR cache before and after each test."""
    from gitta.github import _clear_matched_pr_cache

    _clear_matched_pr_cache()
    yield
    _clear_matched_pr_cache()
