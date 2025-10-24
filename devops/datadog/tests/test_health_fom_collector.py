"""Unit tests for Health FoM collector.

Tests the Figure of Merit (FoM) calculation formulas and collector logic.
"""

from unittest.mock import MagicMock, patch

import pytest

from devops.datadog.collectors.health_fom import HealthFomCollector


class TestFomFormulas:
    """Test FoM calculation formulas for correctness."""

    @pytest.fixture
    def collector(self):
        """Create a mock FoM collector for testing formulas."""
        with patch.dict("os.environ", {"DD_API_KEY": "test_key", "DD_APP_KEY": "test_app_key"}):
            collector = HealthFomCollector()
            # Mock the Datadog client to avoid API calls
            collector._dd_client = MagicMock()
            return collector

    def test_tests_passing_fom_binary(self, collector):
        """Test that tests passing FoM is binary (1.0 or 0.0)."""
        # Mock query to return tests passing = 1
        collector._query_metric = MagicMock(return_value=1.0)

        foms = collector._ci_foms()

        assert "health.ci.tests_passing.fom" in foms
        assert foms["health.ci.tests_passing.fom"] == 1.0

        # Mock query to return tests passing = 0
        collector._query_metric = MagicMock(return_value=0.0)

        foms = collector._ci_foms()

        assert foms["health.ci.tests_passing.fom"] == 0.0

    def test_failing_workflows_fom_scaling(self, collector):
        """Test that failing workflows FoM scales correctly (fewer is better)."""
        test_cases = [
            (0, 1.0),  # 0 failures → perfect (1.0)
            (1, 0.8),  # 1 failure → 0.8
            (2.5, 0.5),  # 2.5 failures → 0.5
            (5, 0.0),  # 5+ failures → 0.0
            (10, 0.0),  # Way over → still 0.0
        ]

        for failures, expected_fom in test_cases:
            collector._query_metric = MagicMock(
                side_effect=lambda m, val=failures: val if "failed_workflows" in m else None
            )

            foms = collector._ci_foms()

            actual_fom = foms.get("health.ci.failing_workflows.fom")
            assert actual_fom is not None, f"FoM not calculated for {failures} failures"
            assert abs(actual_fom - expected_fom) < 0.01, (
                f"For {failures} failures, expected {expected_fom}, got {actual_fom}"
            )

    def test_hotfix_count_fom_scaling(self, collector):
        """Test that hotfix count FoM scales correctly (fewer is better)."""
        test_cases = [
            (0, 1.0),  # 0 hotfixes → perfect (1.0)
            (5, 0.5),  # 5 hotfixes → 0.5
            (10, 0.0),  # 10+ hotfixes → 0.0
            (20, 0.0),  # Way over → still 0.0
        ]

        for hotfixes, expected_fom in test_cases:
            collector._query_metric = MagicMock(side_effect=lambda m, val=hotfixes: val if "hotfix" in m else None)

            foms = collector._ci_foms()

            actual_fom = foms.get("health.ci.hotfix_count.fom")
            assert actual_fom is not None
            assert abs(actual_fom - expected_fom) < 0.01

    def test_revert_count_fom_scaling(self, collector):
        """Test that revert count FoM scales correctly (fewer is better)."""
        test_cases = [
            (0, 1.0),  # 0 reverts → perfect (1.0)
            (1, 0.5),  # 1 revert → 0.5
            (2, 0.0),  # 2+ reverts → 0.0
            (5, 0.0),  # Way over → still 0.0
        ]

        for reverts, expected_fom in test_cases:
            collector._query_metric = MagicMock(side_effect=lambda m, val=reverts: val if "reverts" in m else None)

            foms = collector._ci_foms()

            actual_fom = foms.get("health.ci.revert_count.fom")
            assert actual_fom is not None
            assert abs(actual_fom - expected_fom) < 0.01

    def test_ci_duration_fom_scaling(self, collector):
        """Test that CI duration P90 FoM scales correctly (faster is better)."""
        test_cases = [
            (3.0, 1.0),  # 3 min → perfect (1.0)
            (5.0, 0.71),  # 5 min → ~0.71 (within spec)
            (6.5, 0.5),  # 6.5 min → 0.5
            (10.0, 0.0),  # 10+ min → 0.0
            (20.0, 0.0),  # Way over → still 0.0
        ]

        for duration, expected_fom in test_cases:
            collector._query_metric = MagicMock(
                side_effect=lambda m, val=duration: val if "duration_p90" in m else None
            )

            foms = collector._ci_foms()

            actual_fom = foms.get("health.ci.duration_p90.fom")
            assert actual_fom is not None
            assert abs(actual_fom - expected_fom) < 0.02, (
                f"For {duration} min, expected {expected_fom}, got {actual_fom}"
            )

    def test_stale_prs_fom_scaling(self, collector):
        """Test that stale PRs FoM scales correctly (fewer is better)."""
        test_cases = [
            (0, 1.0),  # 0 stale → perfect (1.0)
            (20, 0.6),  # 20 stale → 0.6
            (25, 0.5),  # 25 stale → 0.5
            (50, 0.0),  # 50+ stale → 0.0
            (100, 0.0),  # Way over → still 0.0
        ]

        for stale_prs, expected_fom in test_cases:
            collector._query_metric = MagicMock(
                side_effect=lambda m, val=stale_prs: val if "stale_count" in m else None
            )

            foms = collector._ci_foms()

            actual_fom = foms.get("health.ci.stale_prs.fom")
            assert actual_fom is not None
            assert abs(actual_fom - expected_fom) < 0.01

    def test_pr_cycle_time_fom_scaling(self, collector):
        """Test that PR cycle time FoM scales correctly (faster is better)."""
        test_cases = [
            (24.0, 1.0),  # 24 hours → perfect (1.0)
            (48.0, 0.5),  # 48 hours → 0.5 (at target)
            (60.0, 0.25),  # 60 hours → 0.25
            (72.0, 0.0),  # 72+ hours → 0.0
            (100.0, 0.0),  # Way over → still 0.0
        ]

        for cycle_time, expected_fom in test_cases:
            collector._query_metric = MagicMock(
                side_effect=lambda m, val=cycle_time: val if "cycle_time" in m else None
            )

            foms = collector._ci_foms()

            actual_fom = foms.get("health.ci.pr_cycle_time.fom")
            assert actual_fom is not None
            assert abs(actual_fom - expected_fom) < 0.01

    def test_all_metrics_none_returns_empty(self, collector):
        """Test that if all metric queries return None, we get empty dict."""
        # Mock all queries to return None (no data)
        collector._query_metric = MagicMock(return_value=None)

        foms = collector._ci_foms()

        # Should return empty dict when no data available
        assert foms == {}

    def test_partial_metrics_available(self, collector):
        """Test that collector handles partial metric availability gracefully."""

        # Mock some metrics available, some not
        def mock_query(metric_name):
            if "tests_passing" in metric_name:
                return 1.0
            elif "hotfix" in metric_name:
                return 3
            else:
                return None  # Other metrics not available

        collector._query_metric = MagicMock(side_effect=mock_query)

        foms = collector._ci_foms()

        # Should have 2 FoMs
        assert len(foms) == 2
        assert "health.ci.tests_passing.fom" in foms
        assert "health.ci.hotfix_count.fom" in foms
        assert foms["health.ci.tests_passing.fom"] == 1.0
        assert abs(foms["health.ci.hotfix_count.fom"] - 0.7) < 0.01  # 3 hotfixes → 0.7


class TestHealthFomCollector:
    """Test HealthFomCollector integration."""

    @patch("softmax.aws.secrets_manager.get_secretsmanager_secret")
    def test_collector_initialization_requires_env_vars(self, mock_get_secret):
        """Test that collector requires DD_API_KEY and DD_APP_KEY."""
        # Mock secrets manager to raise error (no credentials available)
        mock_get_secret.side_effect = Exception("No credentials")

        # Clear env vars
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="DD_API_KEY not found"):
                HealthFomCollector()

    def test_collector_initialization_with_env_vars(self):
        """Test that collector initializes successfully with env vars."""
        with patch.dict("os.environ", {"DD_API_KEY": "test_key", "DD_APP_KEY": "test_app_key"}):
            collector = HealthFomCollector()

            assert collector.name == "health_fom"
            assert collector._dd_client is not None

    def test_collect_metrics_returns_dict(self):
        """Test that collect_metrics returns a dictionary."""
        with patch.dict("os.environ", {"DD_API_KEY": "test_key", "DD_APP_KEY": "test_app_key"}):
            collector = HealthFomCollector()

            # Mock _query_metric to return test data
            collector._query_metric = MagicMock(side_effect=lambda m: 1.0 if "tests_passing" in m else None)

            metrics = collector.collect_metrics()

            assert isinstance(metrics, dict)
            # Should have at least the tests passing FoM
            assert "health.ci.tests_passing.fom" in metrics

    def test_fom_values_are_clamped_to_0_1(
        self,
    ):
        """Test that all FoM values are in [0.0, 1.0] range."""
        with patch.dict("os.environ", {"DD_API_KEY": "test_key", "DD_APP_KEY": "test_app_key"}):
            collector = HealthFomCollector()

            # Mock extreme values
            def mock_query(metric_name):
                if "tests_passing" in metric_name:
                    return 1.0
                elif "hotfix" in metric_name:
                    return 1000  # Way over threshold
                elif "duration" in metric_name:
                    return 0.5  # Way under threshold
                else:
                    return 50

            collector._query_metric = MagicMock(side_effect=mock_query)

            foms = collector.collect_metrics()

            # All values should be in [0.0, 1.0]
            for metric_name, value in foms.items():
                assert 0.0 <= value <= 1.0, f"{metric_name} = {value} is out of range [0.0, 1.0]"


class TestDatadogClientQueryMethod:
    """Test DatadogClient query_metric method."""

    @patch("devops.datadog.utils.datadog_client.ApiClient")
    def test_query_metric_returns_value(self, mock_api_client):
        """Test that query_metric extracts value from API response."""
        from devops.datadog.utils.datadog_client import DatadogClient

        # Mock API response
        mock_response = MagicMock()
        mock_response.series = [
            MagicMock(pointlist=[[1234567890, 42.0]])  # [timestamp, value]
        ]

        mock_metrics_api = MagicMock()
        mock_metrics_api.query_metrics.return_value = mock_response

        mock_api_client.return_value.__enter__.return_value = MagicMock()

        with patch("devops.datadog.utils.datadog_client.MetricsApiV1") as mock_v1:
            mock_v1.return_value = mock_metrics_api

            client = DatadogClient(api_key="test", app_key="test")
            value = client.query_metric("test.metric")

            assert value == 42.0

    @patch("devops.datadog.utils.datadog_client.ApiClient")
    def test_query_metric_returns_none_when_no_data(self, mock_api_client):
        """Test that query_metric returns None when no data available."""
        from devops.datadog.utils.datadog_client import DatadogClient

        # Mock empty response
        mock_response = MagicMock()
        mock_response.series = []

        mock_metrics_api = MagicMock()
        mock_metrics_api.query_metrics.return_value = mock_response

        mock_api_client.return_value.__enter__.return_value = MagicMock()

        with patch("devops.datadog.utils.datadog_client.MetricsApiV1") as mock_v1:
            mock_v1.return_value = mock_metrics_api

            client = DatadogClient(api_key="test", app_key="test")
            value = client.query_metric("test.metric")

            assert value is None

    @patch("devops.datadog.utils.datadog_client.ApiClient")
    def test_query_metric_handles_api_error(self, mock_api_client):
        """Test that query_metric returns None on API error."""
        from devops.datadog.utils.datadog_client import DatadogClient

        mock_api_client.return_value.__enter__.side_effect = Exception("API Error")

        client = DatadogClient(api_key="test", app_key="test")
        value = client.query_metric("test.metric")

        assert value is None
