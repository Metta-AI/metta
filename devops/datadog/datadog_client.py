from __future__ import annotations

import logging
import os
from typing import Iterable

from datadog_api_client import ApiClient, Configuration
from datadog_api_client.v2.api.metrics_api import MetricsApi
from datadog_api_client.v2.model.metric_payload import MetricPayload

from metta.common.datadog.config import datadog_config
from softmax.aws.secrets_manager import get_secretsmanager_secret

from devops.datadog.models import MetricSample

logger = logging.getLogger(__name__)


class DatadogMetricsClient:
    """Thin wrapper that submits MetricSample batches to Datadog."""

    def __init__(self) -> None:
        self._configuration = self._build_configuration()

    def submit(self, samples: Iterable[MetricSample]) -> None:
        series = [sample.to_series() for sample in samples]
        if not series:
            logger.info("No metrics to submit.")
            return

        # Log metric names for debugging
        metric_names = [s.metric for s in series]
        logger.info("Submitting metrics to Datadog (site=%s): %s",
                   self._configuration.server_variables.get("site", "unknown"),
                   ", ".join(metric_names))

        payload = MetricPayload(series=series)
        try:
            with ApiClient(self._configuration) as api_client:
                response = MetricsApi(api_client).submit_metrics(body=payload)
                logger.info("Submitted %s metrics to Datadog successfully", len(series))
                if hasattr(response, 'data') and response.data:
                    logger.debug("Datadog response: %s", response.data)
        except Exception as e:
            logger.error("Failed to submit metrics to Datadog: %s", e, exc_info=True)
            raise

    def _build_configuration(self) -> Configuration:
        configuration = Configuration()
        configuration.server_variables["site"] = os.environ.get("DD_SITE", datadog_config.DD_SITE)

        api_key = self._get_api_key()
        app_key = self._get_app_key()

        if not api_key:
            error_msg = (
                "Missing Datadog API key. "
                "Set DD_API_KEY environment variable or ensure 'datadog/api-key' exists in AWS Secrets Manager. "
                "The service account needs permissions to read from Secrets Manager."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Log API key status (masked for security)
        api_key_preview = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
        logger.info("Using Datadog API key: %s", api_key_preview)

        configuration.api_key["apiKeyAuth"] = api_key
        if app_key:
            configuration.api_key["appKeyAuth"] = app_key

        return configuration

    @staticmethod
    def _get_env_key(candidates: list[str]) -> str | None:
        for key in candidates:
            if value := os.environ.get(key):
                return value
        return None

    def _get_api_key(self) -> str | None:
        # Try environment variables first
        env_key = self._get_env_key(["DD_API_KEY", "DATADOG_API_KEY"])
        if env_key:
            logger.debug("Using API key from environment variable")
            return env_key
        
        # Fall back to AWS Secrets Manager
        logger.debug("Fetching API key from AWS Secrets Manager: datadog/api-key")
        try:
            secret_key = get_secretsmanager_secret("datadog/api-key", require_exists=False)
            if secret_key:
                logger.debug("Successfully loaded API key from AWS Secrets Manager")
            else:
                logger.warning("API key not found in AWS Secrets Manager (datadog/api-key)")
            return secret_key
        except Exception as e:
            logger.error("Failed to fetch API key from AWS Secrets Manager: %s", e, exc_info=True)
            return None

    def _get_app_key(self) -> str | None:
        # Try environment variables first
        env_key = self._get_env_key(["DD_APP_KEY", "DATADOG_APP_KEY"])
        if env_key:
            logger.debug("Using app key from environment variable")
            return env_key
        
        # Fall back to AWS Secrets Manager
        logger.debug("Fetching app key from AWS Secrets Manager: datadog/app-key")
        try:
            secret_key = get_secretsmanager_secret("datadog/app-key", require_exists=False)
            if secret_key:
                logger.debug("Successfully loaded app key from AWS Secrets Manager")
            else:
                logger.debug("App key not found in AWS Secrets Manager (optional)")
            return secret_key
        except Exception as e:
            logger.warning("Failed to fetch app key from AWS Secrets Manager: %s", e)
            return None
