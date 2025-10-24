"""Unit tests for DatadogDashboardClient."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from devops.datadog.utils.dashboard_client import DatadogDashboardClient, get_datadog_credentials


class TestGetDatadogCredentials:
    """Test get_datadog_credentials helper function."""

    def test_credentials_from_environment(self):
        """Test that credentials are read from environment variables."""
        with patch.dict(
            "os.environ",
            {
                "DD_API_KEY": "test_api_key",
                "DD_APP_KEY": "test_app_key",
                "DD_SITE": "datadoghq.eu",
            },
        ):
            api_key, app_key, site = get_datadog_credentials()

            assert api_key == "test_api_key"
            assert app_key == "test_app_key"
            assert site == "datadoghq.eu"

    def test_default_site_when_not_set(self):
        """Test that DD_SITE defaults to datadoghq.com."""
        with patch.dict("os.environ", {"DD_API_KEY": "key1", "DD_APP_KEY": "key2"}):
            api_key, app_key, site = get_datadog_credentials()

            assert site == "datadoghq.com"

    @patch("softmax.aws.secrets_manager.get_secretsmanager_secret")
    def test_api_key_from_secrets_manager(self, mock_get_secret):
        """Test that API key falls back to AWS Secrets Manager."""
        mock_get_secret.return_value = "secret_api_key"

        with patch.dict("os.environ", {"DD_APP_KEY": "app_key"}, clear=True):
            api_key, app_key, site = get_datadog_credentials()

            assert api_key == "secret_api_key"
            mock_get_secret.assert_called_with("datadog/api-key")

    @patch("softmax.aws.secrets_manager.get_secretsmanager_secret")
    def test_app_key_from_secrets_manager(self, mock_get_secret):
        """Test that APP key falls back to AWS Secrets Manager."""
        mock_get_secret.return_value = "secret_app_key"

        with patch.dict("os.environ", {"DD_API_KEY": "api_key"}, clear=True):
            api_key, app_key, site = get_datadog_credentials()

            assert app_key == "secret_app_key"
            mock_get_secret.assert_called_with("datadog/app-key")

    @patch("softmax.aws.secrets_manager.get_secretsmanager_secret")
    def test_secrets_manager_failure_raises_error(self, mock_get_secret):
        """Test that Secrets Manager failure raises ValueError."""
        mock_get_secret.side_effect = Exception("AWS error")

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="DD_API_KEY not found"):
                get_datadog_credentials()


class TestDatadogDashboardClient:
    """Test DatadogDashboardClient class."""

    def test_initialization_with_explicit_credentials(self):
        """Test that client initializes with explicit credentials."""
        client = DatadogDashboardClient(api_key="key1", app_key="key2", site="datadoghq.com")

        assert client.api_key == "key1"
        assert client.app_key == "key2"
        assert client.site == "datadoghq.com"
        assert client.base_url == "https://api.datadoghq.com/api"

    @patch("devops.datadog.utils.dashboard_client.get_datadog_credentials")
    def test_initialization_without_credentials_calls_helper(self, mock_get_creds):
        """Test that client fetches credentials if not provided."""
        mock_get_creds.return_value = ("api", "app", "datadoghq.eu")

        client = DatadogDashboardClient()

        mock_get_creds.assert_called_once()
        assert client.api_key == "api"
        assert client.app_key == "app"
        assert client.site == "datadoghq.eu"

    def test_session_has_correct_headers(self):
        """Test that session is configured with authentication headers."""
        client = DatadogDashboardClient(api_key="test_key", app_key="test_app")

        assert client._session.headers["DD-API-KEY"] == "test_key"
        assert client._session.headers["DD-APPLICATION-KEY"] == "test_app"
        assert client._session.headers["Content-Type"] == "application/json"

    def test_list_dashboards_success(self):
        """Test that list_dashboards returns dashboard list."""
        client = DatadogDashboardClient(api_key="key", app_key="app")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "dashboards": [
                {"id": "abc-123", "title": "Dashboard 1"},
                {"id": "def-456", "title": "Dashboard 2"},
            ]
        }

        with patch.object(client._session, "get", return_value=mock_response):
            dashboards = client.list_dashboards()

            assert len(dashboards) == 2
            assert dashboards[0]["id"] == "abc-123"
            assert dashboards[1]["title"] == "Dashboard 2"

    def test_list_dashboards_http_error(self):
        """Test that list_dashboards raises on HTTP error."""
        client = DatadogDashboardClient(api_key="key", app_key="app")

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")

        with patch.object(client._session, "get", return_value=mock_response):
            with pytest.raises(requests.HTTPError):
                client.list_dashboards()

    def test_get_dashboard_success(self):
        """Test that get_dashboard fetches full dashboard JSON."""
        client = DatadogDashboardClient(api_key="key", app_key="app")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "abc-123",
            "title": "Test Dashboard",
            "widgets": [],
        }

        with patch.object(client._session, "get", return_value=mock_response):
            dashboard = client.get_dashboard("abc-123")

            assert dashboard["id"] == "abc-123"
            assert dashboard["title"] == "Test Dashboard"
            assert "widgets" in dashboard

    def test_create_dashboard_success(self):
        """Test that create_dashboard POSTs new dashboard."""
        client = DatadogDashboardClient(api_key="key", app_key="app")

        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "new-123", "title": "New Dashboard"}

        dashboard_data = {"title": "New Dashboard", "widgets": []}

        with patch.object(client._session, "post", return_value=mock_response) as mock_post:
            result = client.create_dashboard(dashboard_data)

            assert result["id"] == "new-123"
            mock_post.assert_called_once()
            # Verify URL is correct
            call_args = mock_post.call_args
            assert "/v1/dashboard" in call_args[0][0]

    def test_update_dashboard_success(self):
        """Test that update_dashboard PUTs updated dashboard."""
        client = DatadogDashboardClient(api_key="key", app_key="app")

        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "abc-123", "title": "Updated"}

        dashboard_data = {"title": "Updated", "widgets": []}

        with patch.object(client._session, "put", return_value=mock_response) as mock_put:
            result = client.update_dashboard("abc-123", dashboard_data)

            assert result["title"] == "Updated"
            mock_put.assert_called_once()
            # Verify dashboard ID in URL
            call_args = mock_put.call_args
            assert "abc-123" in call_args[0][0]

    def test_delete_dashboard_success(self):
        """Test that delete_dashboard sends DELETE request."""
        client = DatadogDashboardClient(api_key="key", app_key="app")

        mock_response = MagicMock()
        mock_response.json.return_value = {"deleted_dashboard_id": "abc-123"}

        with patch.object(client._session, "delete", return_value=mock_response) as mock_delete:
            result = client.delete_dashboard("abc-123")

            assert result["deleted_dashboard_id"] == "abc-123"
            mock_delete.assert_called_once()

    def test_list_metrics_success(self):
        """Test that list_metrics returns metric names."""
        client = DatadogDashboardClient(api_key="key", app_key="app")

        mock_response = MagicMock()
        mock_response.json.return_value = {"metrics": ["metric.one", "metric.two", "metric.three"]}

        with patch.object(client._session, "get", return_value=mock_response):
            metrics = client.list_metrics()

            assert len(metrics) == 3
            assert "metric.one" in metrics

    def test_list_metrics_with_filter(self):
        """Test that list_metrics passes filter parameter."""
        client = DatadogDashboardClient(api_key="key", app_key="app")

        mock_response = MagicMock()
        mock_response.json.return_value = {"metrics": ["github.prs.open"]}

        with patch.object(client._session, "get", return_value=mock_response) as mock_get:
            client.list_metrics(filter_query="github*")

            # Verify filter was passed
            call_args = mock_get.call_args
            assert "filter" in call_args[1]["params"]
            assert call_args[1]["params"]["filter"] == "github*"

    def test_get_metric_metadata_success(self):
        """Test that get_metric_metadata returns metadata."""
        client = DatadogDashboardClient(api_key="key", app_key="app")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "description": "Number of open PRs",
            "type": "gauge",
            "unit": "pr",
        }

        with patch.object(client._session, "get", return_value=mock_response):
            metadata = client.get_metric_metadata("github.prs.open")

            assert metadata["description"] == "Number of open PRs"
            assert metadata["type"] == "gauge"

    def test_list_tags_success(self):
        """Test that list_tags returns tag information."""
        client = DatadogDashboardClient(api_key="key", app_key="app")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "tags": {
                "env": ["prod", "staging"],
                "service": ["api", "worker"],
            }
        }

        with patch.object(client._session, "get", return_value=mock_response):
            tags = client.list_tags()

            assert "env" in tags
            assert "prod" in tags["env"]
