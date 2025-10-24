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
    @patch("devops.datadog.collectors.github.collector.get_commits")
    @patch("devops.datadog.collectors.github.collector.get_branches")
    @patch("devops.datadog.collectors.github.collector.list_all_workflow_runs")
    def test_collect_safe_handles_api_errors(self, mock_workflows, mock_branches, mock_commits, mock_prs):
        """Test that collect_safe handles GitHub API errors gracefully."""
        # Mock API error on PR collection
        mock_prs.side_effect = Exception("API Rate Limit")
        # Other mocks return empty so their collection succeeds
        mock_commits.return_value = []
        mock_branches.return_value = []
        mock_workflows.return_value = []

        collector = GitHubCollector(
            organization="test-org",
            repository="test-repo",
            github_token="token",
        )

        metrics = collector.collect_safe()

        # Should not raise exception and should return a dict with defaults
        # When PR collection fails, it sets default values for PR metrics
        # Other categories still collect successfully
        assert isinstance(metrics, dict)
        # Should have PR metrics with default values
        assert "prs.open" in metrics
        # Should also have metrics from other categories that didn't fail
        assert len(metrics) > 0

    @patch("devops.datadog.collectors.github.collector.get_pull_requests")
    @patch("devops.datadog.collectors.github.collector.get_commits")
    @patch("devops.datadog.collectors.github.collector.get_branches")
    @patch("devops.datadog.collectors.github.collector.list_all_workflow_runs")
    def test_metric_names_use_dot_notation(self, mock_workflows, mock_branches, mock_commits, mock_prs):
        """Test that all metrics use proper dot notation (category.metric_name)."""
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

        # All metric names should use dot notation (e.g., "prs.open", "commits.total_7d")
        # Just verify they use dot notation, don't be overly specific about prefixes
        for metric_name in metrics.keys():
            assert "." in metric_name, f"Metric {metric_name} doesn't use dot notation"
            # Verify it's properly namespaced (has at least category.name structure)
            parts = metric_name.split(".")
            assert len(parts) >= 2, f"Metric {metric_name} should have at least category.name structure"
