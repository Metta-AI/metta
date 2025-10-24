"""Unit tests for GitHub collector."""

from unittest.mock import MagicMock, patch

import pytest

from devops.datadog.collectors.github import GitHubCollector


class TestGitHubCollector:
    """Test GitHubCollector functionality."""

    @patch("devops.datadog.collectors.github.collector.Github")
    @patch("devops.datadog.collectors.github.collector.get_secretsmanager_secret")
    def test_initialization_success(self, mock_get_secret, mock_github):
        """Test that collector initializes with GitHub token."""
        mock_get_secret.return_value = "test_github_token"
        mock_github_instance = MagicMock()
        mock_github.return_value = mock_github_instance

        collector = GitHubCollector()

        assert collector.name == "github"
        mock_get_secret.assert_called_once_with("github/dashboard-token")
        mock_github.assert_called_once_with("test_github_token")

    @patch("devops.datadog.collectors.github.collector.Github")
    @patch("devops.datadog.collectors.github.collector.get_secretsmanager_secret")
    def test_initialization_with_explicit_token(self, mock_get_secret, mock_github):
        """Test that explicit token overrides secrets manager."""
        mock_github_instance = MagicMock()
        mock_github.return_value = mock_github_instance

        GitHubCollector(github_token="explicit_token")

        # Should not call secrets manager
        mock_get_secret.assert_not_called()
        mock_github.assert_called_once_with("explicit_token")

    @patch("devops.datadog.collectors.github.collector.Github")
    @patch("devops.datadog.collectors.github.collector.get_secretsmanager_secret")
    def test_secrets_manager_failure_raises_error(self, mock_get_secret, mock_github):
        """Test that Secrets Manager failure raises ValueError."""
        mock_get_secret.side_effect = Exception("AWS Error")

        with pytest.raises(ValueError, match="Failed to get GitHub token"):
            GitHubCollector()

    @patch("devops.datadog.collectors.github.collector.Github")
    @patch("devops.datadog.collectors.github.collector.get_secretsmanager_secret")
    def test_collect_metrics_returns_dict(self, mock_get_secret, mock_github):
        """Test that collect_metrics returns a dictionary."""
        mock_get_secret.return_value = "token"
        mock_github_instance = MagicMock()
        mock_repo = MagicMock()

        # Mock repo methods
        mock_repo.get_pulls.return_value.totalCount = 5
        mock_repo.get_commits.return_value.totalCount = 100

        mock_github_instance.get_repo.return_value = mock_repo
        mock_github.return_value = mock_github_instance

        collector = GitHubCollector()
        metrics = collector.collect_metrics()

        assert isinstance(metrics, dict)
        # Should have some metrics
        assert len(metrics) > 0

    @patch("devops.datadog.collectors.github.collector.Github")
    @patch("devops.datadog.collectors.github.collector.get_secretsmanager_secret")
    def test_collect_safe_handles_api_errors(self, mock_get_secret, mock_github):
        """Test that collect_safe handles GitHub API errors gracefully."""
        mock_get_secret.return_value = "token"
        mock_github_instance = MagicMock()

        # Mock API error
        mock_github_instance.get_repo.side_effect = Exception("API Rate Limit")
        mock_github.return_value = mock_github_instance

        collector = GitHubCollector()
        metrics = collector.collect_safe()

        # Should return error metrics instead of raising
        assert "github.collection_success" in metrics
        assert metrics["github.collection_success"] == 0.0
        assert "github.error_count" in metrics
        assert metrics["github.error_count"] == 1.0

    @patch("devops.datadog.collectors.github.collector.Github")
    @patch("devops.datadog.collectors.github.collector.get_secretsmanager_secret")
    def test_metric_names_have_correct_prefix(self, mock_get_secret, mock_github):
        """Test that all metrics start with 'github.' prefix."""
        mock_get_secret.return_value = "token"
        mock_github_instance = MagicMock()
        mock_repo = MagicMock()

        # Setup minimal mocks to avoid errors
        mock_repo.get_pulls.return_value.totalCount = 1
        mock_github_instance.get_repo.return_value = mock_repo
        mock_github.return_value = mock_github_instance

        collector = GitHubCollector()
        metrics = collector.collect_safe()

        # All metric names should start with 'github.'
        for metric_name in metrics.keys():
            assert metric_name.startswith("github."), f"Metric {metric_name} doesn't start with 'github.'"
