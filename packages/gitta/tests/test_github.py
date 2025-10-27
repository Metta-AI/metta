"""Tests for GitHub API and CLI functionality."""

import os
import subprocess
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest

from gitta import (
    GitError,
    create_pr,
    get_matched_pr,
    post_commit_status,
    run_gh,
)


class TestGitHubCLI:
    """Test GitHub CLI integration."""

    @patch("subprocess.run")
    def test_run_gh_success(self, mock_run):
        """Test successful gh command execution."""
        mock_run.return_value = subprocess.CompletedProcess(
            args=["gh", "pr", "list"], returncode=0, stdout="PR #1: Title\n", stderr=""
        )

        result = run_gh("pr", "list")
        assert "PR #1" in result

    @patch("subprocess.run")
    def test_run_gh_failure(self, mock_run):
        """Test gh command failure."""
        # Simulate CalledProcessError which run_gh catches and converts to GitError
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=["gh", "pr", "create"], stderr="error: authentication required"
        )

        with pytest.raises(GitError) as exc_info:
            run_gh("pr", "create")
        assert "authentication required" in str(exc_info.value)

    @patch("subprocess.run")
    def test_run_gh_not_installed(self, mock_run):
        """Test error when gh is not installed."""
        mock_run.side_effect = FileNotFoundError()

        with pytest.raises(GitError) as exc_info:
            run_gh("pr", "list")
        assert "not installed" in str(exc_info.value).lower()


class TestGitHubAPI:
    """Test GitHub API functions."""

    @patch("httpx.get")
    def test_get_matched_pr_success(self, mock_get):
        """Test PR matching functionality with successful response."""
        # Mock successful response with PR data
        mock_response = MagicMock()
        mock_response.json.return_value = [{"number": 123, "title": "Test PR"}]
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = get_matched_pr("abc123", "Metta-AI/metta")
        assert result is not None
        assert result[0] == 123
        assert result[1] == "Test PR"

    @patch("httpx.get")
    def test_get_matched_pr_no_matches(self, mock_get):
        """Test PR matching when no PRs found."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        result = get_matched_pr("abc123", "Metta-AI/metta")
        assert result is None

    @patch("httpx.get")
    def test_get_matched_pr_404_error(self, mock_get):
        """Test PR matching with 404 error (commit not found)."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not found"

        # Use real httpx.HTTPStatusError
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            message="Not found", request=Mock(), response=mock_response
        )
        mock_get.return_value = mock_response

        # Should return None for 404 (not found)
        result = get_matched_pr("abc123", "Metta-AI/metta")
        assert result is None

    @patch("httpx.post")
    def test_post_commit_status_success(self, mock_post):
        """Test posting commit status to GitHub."""
        mock_response = Mock()
        mock_response.json.return_value = {"state": "success", "context": "CI", "description": "Tests passed"}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        # Set environment variable for token
        with patch.dict(os.environ, {"GITHUB_TOKEN": "test_token"}):
            result = post_commit_status(
                commit_sha="abc123", state="success", repo="Metta-AI/metta", description="Tests passed"
            )

        assert result["state"] == "success"
        assert result["description"] == "Tests passed"
        mock_post.assert_called_once()

        # Check the call was made with correct headers
        call_args = mock_post.call_args
        assert "Authorization" in call_args[1]["headers"]
        assert "token test_token" in call_args[1]["headers"]["Authorization"]

    def test_post_commit_status_no_token(self):
        """Test that post_commit_status fails without token."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                post_commit_status("abc123", "success", "Metta-AI/metta")
            assert "token not provided" in str(exc_info.value).lower()

    def test_post_commit_status_no_repo(self):
        """Test that post_commit_status fails without repo."""
        with patch.dict(os.environ, {"GITHUB_TOKEN": "test_token"}):
            with pytest.raises(ValueError) as exc_info:
                post_commit_status("abc123", "success", "")
            assert "repository must be provided" in str(exc_info.value).lower()

    @patch("httpx.post")
    def test_create_pr_success(self, mock_post):
        """Test creating a pull request."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "number": 456,
            "html_url": "https://github.com/Metta-AI/metta/pull/456",
            "title": "Test PR",
            "state": "open",
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {"GITHUB_TOKEN": "test_token"}):
            result = create_pr(
                repo="Metta-AI/metta", title="Test PR", body="Test description", head="feature-branch", base="main"
            )

        assert result["number"] == 456
        assert "pull/456" in result["html_url"]
        mock_post.assert_called_once()

    @patch("httpx.post")
    def test_create_pr_with_draft(self, mock_post):
        """Test creating a draft pull request."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "number": 789,
            "draft": True,
            "html_url": "https://github.com/Metta-AI/metta/pull/789",
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {"GITHUB_TOKEN": "test_token"}):
            result = create_pr(
                repo="Metta-AI/metta", title="Draft PR", body="WIP", head="feature", base="main", draft=True
            )

        assert result["draft"] is True
        # Check that draft parameter was sent
        call_args = mock_post.call_args
        assert call_args[1]["json"]["draft"] is True

    def test_create_pr_no_token(self):
        """Test that create_pr fails without token."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                create_pr(repo="Metta-AI/metta", title="Test", body="Test", head="feature", base="main")
            assert "token not provided" in str(exc_info.value).lower()

    @patch("httpx.post")
    def test_create_pr_api_error(self, mock_post):
        """Test create_pr with API error."""
        mock_response = Mock()
        mock_response.text = "Bad request: invalid base branch"

        # Use real httpx.HTTPError
        mock_response.raise_for_status.side_effect = httpx.HTTPError("Bad request: invalid base branch")
        mock_post.return_value = mock_response

        with patch.dict(os.environ, {"GITHUB_TOKEN": "test_token"}):
            with pytest.raises(GitError) as exc_info:
                create_pr(repo="Metta-AI/metta", title="Test", body="Test", head="feature", base="invalid")
            assert "Failed to create PR" in str(exc_info.value)
            assert "Bad request" in str(exc_info.value)
