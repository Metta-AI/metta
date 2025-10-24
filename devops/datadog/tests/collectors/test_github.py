"""Unit tests for GitHub collector."""

from unittest.mock import MagicMock, patch

from devops.datadog.collectors.github import GitHubCollector


class TestGitHubCollector:
    """Test GitHubCollector functionality."""

    def test_initialization_success(self):
        """Test that collector initializes with required parameters."""
        collector = GitHubCollector(
            organization="test-org",
            repository="test-repo",
            github_token="test_token_123",
        )

        assert collector.name == "github"
        assert collector.organization == "test-org"
        assert collector.repository == "test-repo"
        assert collector.github_token == "test_token_123"
        assert collector.repo == "test-org/test-repo"

    def test_get_auth_header(self):
        """Test that auth header is formatted correctly."""
        collector = GitHubCollector(
            organization="org",
            repository="repo",
            github_token="my_secret_token",
        )

        auth_header = collector._get_auth_header()

        assert auth_header == "token my_secret_token"

    @patch("devops.datadog.collectors.github.collector.get_pull_requests")
    @patch("devops.datadog.collectors.github.collector.get_commits")
    @patch("devops.datadog.collectors.github.collector.get_branches")
    @patch("devops.datadog.collectors.github.collector.list_all_workflow_runs")
    def test_collect_metrics_returns_dict(self, mock_workflows, mock_branches, mock_commits, mock_prs):
        """Test that collect_metrics returns a dictionary."""
        # Mock the gitta API responses
        mock_prs.return_value = [MagicMock(state="open", number=1)]
        mock_commits.return_value = [MagicMock()]
        mock_branches.return_value = [MagicMock()]
        mock_workflows.return_value = []

        collector = GitHubCollector(
            organization="test-org",
            repository="test-repo",
            github_token="token",
        )

        metrics = collector.collect_metrics()

        assert isinstance(metrics, dict)
        # Should have some metrics
        assert len(metrics) > 0

    @patch("devops.datadog.collectors.github.collector.get_pull_requests")
    def test_collect_safe_handles_api_errors(self, mock_prs):
        """Test that collect_safe handles GitHub API errors gracefully."""
        # Mock API error
        mock_prs.side_effect = Exception("API Rate Limit")

        collector = GitHubCollector(
            organization="test-org",
            repository="test-repo",
            github_token="token",
        )

        metrics = collector.collect_safe()

        # Should return empty dict on error (BaseCollector behavior)
        assert metrics == {}

    @patch("devops.datadog.collectors.github.collector.get_pull_requests")
    @patch("devops.datadog.collectors.github.collector.get_commits")
    @patch("devops.datadog.collectors.github.collector.get_branches")
    @patch("devops.datadog.collectors.github.collector.list_all_workflow_runs")
    def test_metric_names_have_correct_prefix(self, mock_workflows, mock_branches, mock_commits, mock_prs):
        """Test that all metrics start with 'github.' prefix."""
        # Mock minimal responses to avoid errors
        mock_prs.return_value = []
        mock_commits.return_value = []
        mock_branches.return_value = []
        mock_workflows.return_value = []

        collector = GitHubCollector(
            organization="test-org",
            repository="test-repo",
            github_token="token",
        )

        metrics = collector.collect_safe()

        # All metric names should start with 'github.'
        for metric_name in metrics.keys():
            assert metric_name.startswith("github."), f"Metric {metric_name} doesn't start with 'github.'"
