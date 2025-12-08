"""Service configurations and helper functions for external integrations.

This module centralizes all service endpoints, buckets, and helper functions
for interacting with external services like Observatory, WandB, S3, SkyPilot, etc.
"""

import json
import logging
import urllib.request
from typing import Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)


# ==============================================================================
# Service Endpoints
# ==============================================================================


class ServiceEndpoints:
    """Centralized service endpoints configuration."""

    # Observatory
    OBSERVATORY_API = "https://api.observatory.softmax-research.net"
    OBSERVATORY_WEB = "https://observatory.softmax-research.net"

    # SkyPilot
    SKYPILOT_API = "https://skypilot-api.softmax-research.net"

    # WandB
    WANDB_BASE = "https://wandb.ai"
    WANDB_PROJECT = "metta-research/metta"

    # Datadog
    DATADOG_LOGS = "https://app.datadoghq.com/logs"

    # S3
    S3_BUCKET = "softmax-public"
    S3_POLICIES_PREFIX = "policies"


# ==============================================================================
# Observatory Service
# ==============================================================================


class PolicyInfo:
    """Policy information from Observatory."""

    def __init__(
        self,
        policy_id: Optional[str] = None,
        policy_version_id: Optional[str] = None,
        version: Optional[int] = None,
        name: Optional[str] = None,
    ):
        self.policy_id = policy_id
        self.policy_version_id = policy_version_id
        self.version = version
        self.name = name


class ObservatoryService:
    """Helper functions for Observatory API integration."""

    @staticmethod
    def get_observatory_web_base() -> str:
        """Get Observatory web base URL."""
        return ServiceEndpoints.OBSERVATORY_WEB

    @staticmethod
    def get_policy_versions_url(policy_name: str, limit: int = 1) -> str:
        """Get Observatory API URL for fetching policy versions by name.

        Args:
            policy_name: Policy name to search for (exact match)
            limit: Maximum number of results to return

        Returns:
            Full API URL
        """
        return f"{ServiceEndpoints.OBSERVATORY_API}/stats/policy-versions?name_exact={quote(policy_name, safe='')}&limit={limit}"

    @staticmethod
    def get_policy_web_url(policy_version_id: str) -> str:
        """Get Observatory web URL for viewing a policy version.

        Args:
            policy_version_id: Policy version UUID

        Returns:
            Full web URL
        """
        return f"{ServiceEndpoints.OBSERVATORY_WEB}/policy-version/{policy_version_id}"

    @staticmethod
    def get_policy_web_url_by_name(policy_name: str) -> str:
        """Get Observatory web URL for viewing a policy by name (fallback).

        Args:
            policy_name: Policy name (will be URL-encoded)

        Returns:
            Full web URL
        """
        return f"{ServiceEndpoints.OBSERVATORY_WEB}/policy/{quote(policy_name, safe='')}"

    @staticmethod
    def fetch_policy_info(policy_name: str, timeout: int = 5) -> Optional[PolicyInfo]:
        """Fetch policy information from Observatory API.

        Args:
            policy_name: Policy name to search for (exact match)
            timeout: Request timeout in seconds

        Returns:
            PolicyInfo if found, None otherwise
        """
        try:
            url = ObservatoryService.get_policy_versions_url(policy_name, limit=1)
            req = urllib.request.Request(url)
            req.add_header("Accept", "application/json")

            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
                # Response format: {"data": [...], "total": N}
                items = data.get("data", data) if isinstance(data, dict) else data
                if items and len(items) > 0:
                    item = items[0]
                    return PolicyInfo(
                        policy_id=item.get("policy_id"),
                        policy_version_id=item.get("id"),
                        version=item.get("version"),
                        name=item.get("name"),
                    )
        except Exception as e:
            logger.debug(f"Could not fetch policy info for {policy_name}: {e}")

        return None

    @staticmethod
    def fetch_policy_version(experiment_id: str, timeout: int = 5) -> Optional[str]:
        """Fetch policy version from Observatory API (legacy compatibility).

        Args:
            experiment_id: Experiment ID to search for
            timeout: Request timeout in seconds

        Returns:
            Policy version string if found, None otherwise
        """
        info = ObservatoryService.fetch_policy_info(experiment_id, timeout)
        return str(info.version) if info and info.version is not None else None


# ==============================================================================
# WandB Service
# ==============================================================================


class WandbService:
    """Helper functions for WandB integration."""

    @staticmethod
    def get_run_url(experiment_id: str) -> str:
        """Get WandB run URL for an experiment.

        Args:
            experiment_id: Experiment/run ID

        Returns:
            Full WandB URL
        """
        return f"{ServiceEndpoints.WANDB_BASE}/{ServiceEndpoints.WANDB_PROJECT}/runs/{experiment_id}"


# ==============================================================================
# Datadog Service
# ==============================================================================


class DatadogService:
    """Helper functions for Datadog integration."""

    @staticmethod
    def get_logs_url(query: str) -> str:
        """Get Datadog logs URL with query.

        Args:
            query: Datadog query string

        Returns:
            Full Datadog logs URL
        """
        return f"{ServiceEndpoints.DATADOG_LOGS}?query={query}"

    @staticmethod
    def get_experiment_logs_url(experiment_id: str) -> str:
        """Get Datadog logs URL for an experiment.

        Args:
            experiment_id: Experiment/run ID

        Returns:
            Full Datadog logs URL filtered by experiment ID
        """
        query = f"metta_run_id%3A%22{experiment_id}%22"
        return DatadogService.get_logs_url(query)

    @staticmethod
    def get_job_logs_url(job_id: str, experiment_id: Optional[str] = None) -> str:
        """Get Datadog logs URL for a job.

        Args:
            job_id: SkyPilot job ID
            experiment_id: Optional experiment ID for additional filtering

        Returns:
            Full Datadog logs URL filtered by job and optionally experiment
        """
        if experiment_id:
            query = f"skypilot_task_id%3A%2A{job_id}%2A%20metta_run_id%3A%22{experiment_id}%22"
        else:
            query = f"skypilot_task_id%3A%2A{job_id}%2A"
        return DatadogService.get_logs_url(query)


# ==============================================================================
# SkyPilot Service
# ==============================================================================


class SkyPilotService:
    """Helper functions for SkyPilot integration."""

    @staticmethod
    def get_dashboard_url() -> str:
        """Get SkyPilot dashboard base URL.

        Returns:
            Base dashboard URL
        """
        return f"{ServiceEndpoints.SKYPILOT_API}/dashboard"

    @staticmethod
    def get_job_dashboard_url(job_id: str) -> str:
        """Get SkyPilot dashboard URL for a specific job.

        Args:
            job_id: Job ID

        Returns:
            Full job dashboard URL
        """
        return f"{SkyPilotService.get_dashboard_url()}/jobs/{job_id}"

    @staticmethod
    def fetch_job_entrypoint(job_id: str) -> Optional[str]:
        """Fetch job entrypoint command from SkyPilot dashboard.

        NOTE: The SkyPilot dashboard (https://skypilot-api.softmax-research.net/dashboard/jobs/{job_id})
        displays job entrypoint commands, but it's a Next.js single-page app that loads data
        dynamically via JavaScript. There's no REST API endpoint available for fetching this data.

        To implement this function properly, we would need to:
        1. Use a headless browser (Playwright/Selenium) to render the JavaScript
        2. Extract the entrypoint from the rendered page
        3. See test_links.py for an example of using Playwright with the dashboard

        Args:
            job_id: Job ID

        Returns:
            None (not yet implemented)
        """
        # TODO: Implement using Playwright to render the dashboard page and extract entrypoint
        logger.debug(f"fetch_job_entrypoint not yet implemented for job {job_id}")
        return None


# ==============================================================================
# S3 Service
# ==============================================================================


class S3Service:
    """Helper functions for S3 integration."""

    @staticmethod
    def get_policies_path(experiment_id: str) -> str:
        """Get S3 path for experiment policies.

        Args:
            experiment_id: Experiment ID

        Returns:
            Full S3 path (s3://bucket/prefix/experiment_id/)
        """
        return f"s3://{ServiceEndpoints.S3_BUCKET}/{ServiceEndpoints.S3_POLICIES_PREFIX}/{experiment_id}/"

    @staticmethod
    def get_model_path(experiment_id: str, filename: str) -> str:
        """Get S3 path for a model file.

        Args:
            experiment_id: Experiment ID
            filename: Model filename

        Returns:
            Full S3 path to model file
        """
        return f"{S3Service.get_policies_path(experiment_id)}{filename}"


# ==============================================================================
# Convenience Functions
# ==============================================================================


def get_all_experiment_links(experiment_id: str) -> dict[str, str]:
    """Get all external service links for an experiment.

    Args:
        experiment_id: Experiment ID

    Returns:
        Dictionary mapping service name to URL
    """
    return {
        "wandb": WandbService.get_run_url(experiment_id),
        "datadog": DatadogService.get_experiment_logs_url(experiment_id),
        "observatory_api": ObservatoryService.get_policy_api_url(experiment_id, limit=500),
        "observatory_web": ObservatoryService.get_policy_web_url(experiment_id),
        "s3": S3Service.get_policies_path(experiment_id),
    }


def get_all_job_links(job_id: str, experiment_id: Optional[str] = None) -> dict[str, str]:
    """Get all external service links for a job.

    Args:
        job_id: Job ID
        experiment_id: Optional experiment ID for additional context

    Returns:
        Dictionary mapping service name to URL
    """
    links = {
        "skypilot": SkyPilotService.get_job_dashboard_url(job_id),
        "datadog": DatadogService.get_job_logs_url(job_id, experiment_id),
    }

    if experiment_id:
        links["wandb"] = WandbService.get_run_url(experiment_id)

    return links
