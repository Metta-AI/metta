from __future__ import annotations

import logging
import os
from typing import Iterable

from datadog_api_client import ApiClient, Configuration
from datadog_api_client.v2.api.metrics_api import MetricsApi
from datadog_api_client.v2.model.metric_payload import MetricPayload

from metta.common.datadog.config import datadog_config
from softmax.aws.secrets_manager import get_secretsmanager_secret

from .models import MetricSample

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

        payload = MetricPayload(series=series)
        with ApiClient(self._configuration) as api_client:
            MetricsApi(api_client).submit_metrics(body=payload)
        logger.info("Submitted %s metrics to Datadog", len(series))

    def _build_configuration(self) -> Configuration:
        configuration = Configuration()
        configuration.server_variables["site"] = os.environ.get("DD_SITE", datadog_config.DD_SITE)

        api_key = self._get_api_key()
        app_key = self._get_app_key()

        if not api_key:
            raise RuntimeError("Missing Datadog API key. Set DD_API_KEY or DATADOG_API_KEY.")

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
        return self._get_env_key(["DD_API_KEY", "DATADOG_API_KEY"]) or get_secretsmanager_secret(
            "datadog/api-key", require_exists=False
        )

    def _get_app_key(self) -> str | None:
        return self._get_env_key(["DD_APP_KEY", "DATADOG_APP_KEY"]) or get_secretsmanager_secret(
            "datadog/app-key", require_exists=False
        )
