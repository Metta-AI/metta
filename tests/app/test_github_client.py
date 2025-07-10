from unittest.mock import MagicMock, patch

import httpx
import pytest

from app_backend.git_client import GitCommit, GitError
from app_backend.github_client import GitHubClient, run_gh


class TestGithubUtilities:
    def test_run_gh_not_installed(self):
        """Test GitHub CLI not installed scenario."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            with pytest.raises(GitError) as exc_info:
                run_gh("pr", "list")
            assert "GitHub CLI (gh) is not installed" in str(exc_info.value)


class TestGitHubClient:
    """Tests for GitHubClient."""

    def test_github_client_creation(self):
        """Test GitHubClient creation with httpx dependency."""
        with httpx.Client() as http_client:
            github_client = GitHubClient(http_client)
            assert github_client.http_client == http_client

    def test_get_merge_base_success(self):
        """Test successful merge base retrieval."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"merge_base_commit": {"sha": "abc123"}}

        with httpx.Client() as http_client:
            github_client = GitHubClient(http_client)

            with patch.object(http_client, "get", return_value=mock_response):
                result = github_client.get_merge_base("commit-hash")
                assert result == "abc123"

    def test_get_merge_base_failure(self):
        """Test merge base retrieval failure."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not found", request=None, response=mock_response
        )

        with httpx.Client() as http_client:
            github_client = GitHubClient(http_client)

            with patch.object(http_client, "get", return_value=mock_response):
                with pytest.raises(httpx.HTTPStatusError):
                    github_client.get_merge_base("invalid-commit-hash")

    def test_get_commit_range_success(self):
        """Test successful commit range retrieval."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "commits": [
                {
                    "sha": "abc123",
                    "commit": {
                        "message": "Test commit",
                        "author": {"name": "Test Author", "date": "2024-01-01T00:00:00Z"},
                    },
                }
            ]
        }

        with httpx.Client() as http_client:
            github_client = GitHubClient(http_client)

            with patch.object(http_client, "get", return_value=mock_response):
                commits = github_client.get_commit_range("merge-base", "commit-hash")
                assert len(commits) == 1
                assert isinstance(commits[0], GitCommit)
                assert commits[0].hash == "abc123"
                assert commits[0].message == "Test commit"
                assert commits[0].author == "Test Author"
                assert commits[0].date == "2024-01-01"

    def test_get_commit_range_failure(self):
        """Test commit range retrieval failure."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not found", request=None, response=mock_response
        )

        with httpx.Client() as http_client:
            github_client = GitHubClient(http_client)

            with patch.object(http_client, "get", return_value=mock_response):
                with pytest.raises(httpx.HTTPStatusError):
                    github_client.get_commit_range("merge-base", "invalid-commit-hash")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
