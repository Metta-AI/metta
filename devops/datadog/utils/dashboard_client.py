"""Datadog Dashboard API client for dashboard CRUD operations."""

import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)


def get_datadog_credentials() -> tuple[str, str, str]:
    """Get Datadog credentials from environment or AWS Secrets Manager.

    Returns:
        Tuple of (api_key, app_key, site)

    Raises:
        ValueError: If credentials cannot be found
    """
    api_key = os.getenv("DD_API_KEY")
    app_key = os.getenv("DD_APP_KEY")
    site = os.getenv("DD_SITE", "datadoghq.com")

    # Fetch from AWS Secrets Manager if not in environment
    if not api_key:
        try:
            from devops.datadog.utils.secrets import get_secretsmanager_secret

            api_key = get_secretsmanager_secret("datadog/api-key")
        except Exception as e:
            raise ValueError(f"DD_API_KEY not found in environment or AWS Secrets Manager: {e}") from e

    if not app_key:
        try:
            from devops.datadog.utils.secrets import get_secretsmanager_secret

            app_key = get_secretsmanager_secret("datadog/app-key")
        except Exception as e:
            raise ValueError(f"DD_APP_KEY not found in environment or AWS Secrets Manager: {e}") from e

    if not api_key or not app_key:
        raise ValueError("Missing required credentials: DD_API_KEY and DD_APP_KEY")

    return api_key, app_key, site


class DatadogDashboardClient:
    """Client for Datadog Dashboard API operations.

    Handles authentication and provides methods for dashboard CRUD operations.
    Uses the Datadog HTTP API v1 for dashboards.
    """

    def __init__(self, api_key: str | None = None, app_key: str | None = None, site: str | None = None):
        """Initialize Datadog Dashboard API client.

        Args:
            api_key: Datadog API key (if None, fetches from env/secrets)
            app_key: Datadog application key (if None, fetches from env/secrets)
            site: Datadog site (default: datadoghq.com)
        """
        if api_key is None or app_key is None:
            api_key, app_key, site = get_datadog_credentials()
        elif site is None:
            site = os.getenv("DD_SITE", "datadoghq.com")

        self.api_key = api_key
        self.app_key = app_key
        self.site = site
        self.base_url = f"https://api.{site}/api"
        self._session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create authenticated session for Datadog API."""
        session = requests.Session()
        session.headers.update(
            {
                "DD-API-KEY": self.api_key,
                "DD-APPLICATION-KEY": self.app_key,
                "Content-Type": "application/json",
            }
        )
        return session

    def list_dashboards(self) -> list[dict[str, Any]]:
        """Fetch list of all dashboards.

        Returns:
            List of dashboard summaries with id, title, url, etc.
        """
        url = f"{self.base_url}/v1/dashboard"
        response = self._session.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get("dashboards", [])

    def get_dashboard(self, dashboard_id: str) -> dict[str, Any]:
        """Fetch full dashboard JSON by ID.

        Args:
            dashboard_id: Dashboard ID

        Returns:
            Full dashboard JSON definition
        """
        url = f"{self.base_url}/v1/dashboard/{dashboard_id}"
        response = self._session.get(url)
        response.raise_for_status()
        return response.json()

    def dashboard_exists(self, dashboard_id: str) -> bool:
        """Check if a dashboard exists.

        Args:
            dashboard_id: Dashboard ID

        Returns:
            True if dashboard exists, False otherwise
        """
        url = f"{self.base_url}/v1/dashboard/{dashboard_id}"
        try:
            response = self._session.get(url)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def create_dashboard(self, dashboard_json: dict[str, Any]) -> dict[str, Any]:
        """Create a new dashboard.

        Args:
            dashboard_json: Dashboard JSON definition

        Returns:
            Created dashboard JSON with ID
        """
        url = f"{self.base_url}/v1/dashboard"

        # Remove fields that shouldn't be in create request
        payload = dashboard_json.copy()
        for field in ["id", "url", "created_at", "modified_at", "author_handle", "author_name"]:
            payload.pop(field, None)

        response = self._session.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def update_dashboard(self, dashboard_id: str, dashboard_json: dict[str, Any]) -> dict[str, Any]:
        """Update an existing dashboard.

        Args:
            dashboard_id: Dashboard ID
            dashboard_json: Dashboard JSON definition

        Returns:
            Updated dashboard JSON
        """
        url = f"{self.base_url}/v1/dashboard/{dashboard_id}"

        # Remove fields that shouldn't be in update request
        payload = dashboard_json.copy()
        for field in ["id", "url", "created_at", "modified_at", "author_handle", "author_name"]:
            payload.pop(field, None)

        response = self._session.put(url, json=payload)
        response.raise_for_status()
        return response.json()

    def delete_dashboard(self, dashboard_id: str) -> dict[str, Any]:
        """Delete a dashboard.

        Args:
            dashboard_id: Dashboard ID

        Returns:
            Response from Datadog API
        """
        url = f"{self.base_url}/v1/dashboard/{dashboard_id}"
        response = self._session.delete(url)
        response.raise_for_status()
        return response.json()

    def list_metrics(self, search: str | None = None, from_seconds: int = 86400) -> list[str]:
        """List active metrics.

        Args:
            search: Optional search filter
            from_seconds: Lookback period in seconds (default: 24 hours)

        Returns:
            List of metric names
        """
        url = f"{self.base_url}/v1/metrics"
        params = {"from": f"-{from_seconds}"}

        response = self._session.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        metrics = data.get("metrics", [])

        if search:
            search_lower = search.lower()
            metrics = [m for m in metrics if search_lower in m.lower()]

        return sorted(metrics)

    def get_metric_metadata(self, metric_name: str) -> dict[str, Any]:
        """Get metadata for a specific metric.

        Args:
            metric_name: Metric name

        Returns:
            Metric metadata
        """
        url = f"{self.base_url}/v1/metrics/{metric_name}"
        response = self._session.get(url)
        response.raise_for_status()
        return response.json()

    def list_tags(self) -> dict[str, list[str]]:
        """List all available tags.

        Returns:
            Dict of tag categories and values
        """
        url = f"{self.base_url}/v1/tags/hosts"
        response = self._session.get(url)
        response.raise_for_status()
        return response.json()
