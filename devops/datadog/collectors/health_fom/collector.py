"""Health Figure of Merit (FoM) collector for Datadog monitoring.

This collector reads raw metrics from Datadog, applies normalization formulas,
and emits health.*.fom metrics (0.0-1.0 scale) for the System Health Dashboard.

Phase 1: CI/CD metrics (7 FoMs)
Future phases: Training metrics (WandB), Eval metrics
"""

import os
from typing import Any

from devops.datadog.common.base import BaseCollector
from devops.datadog.common.datadog_client import DatadogClient


class HealthFomCollector(BaseCollector):
    """Calculate Figure of Merit (FoM) values for system health metrics.

    Reads raw metrics from Datadog API, applies normalization formulas to
    convert them to a 0.0-1.0 scale, and emits health.*.fom metrics.

    FoM Scale:
    - 1.0 = Healthy (green)
    - 0.7-1.0 = Good (green)
    - 0.3-0.7 = Warning (yellow)
    - 0.0-0.3 = Critical (red)
    """

    def __init__(self):
        super().__init__(name="health_fom")

        # Initialize Datadog client for querying metrics
        api_key = os.getenv("DD_API_KEY")
        app_key = os.getenv("DD_APP_KEY")
        site = os.getenv("DD_SITE", "datadoghq.com")

        # Fetch from AWS Secrets Manager if not in environment
        if not api_key:
            try:
                from softmax.aws.secrets_manager import get_secretsmanager_secret

                api_key = get_secretsmanager_secret("datadog/api-key")
            except Exception as e:
                raise ValueError(f"DD_API_KEY not found in environment or AWS Secrets Manager: {e}") from e

        if not app_key:
            try:
                from softmax.aws.secrets_manager import get_secretsmanager_secret

                app_key = get_secretsmanager_secret("datadog/app-key")
            except Exception as e:
                raise ValueError(f"DD_APP_KEY not found in environment or AWS Secrets Manager: {e}") from e

        if not api_key or not app_key:
            raise ValueError("DD_API_KEY and DD_APP_KEY must be set for FoM collector")

        self._dd_client = DatadogClient(api_key=api_key, app_key=app_key, site=site)

    def collect_metrics(self) -> dict[str, Any]:
        """Calculate all FoM metrics from raw sources.

        Returns:
            Dict of health.*.fom metrics with values in [0.0, 1.0]
        """
        fom_metrics = {}

        # Phase 1: CI/CD FoMs (7 metrics, all raw sources available from GitHub collector)
        fom_metrics.update(self._ci_foms())

        # Phase 2: Training FoMs (9 metrics, requires WandB collector)
        # fom_metrics.update(self._training_foms())

        # Phase 3: Eval FoMs (5 metrics, requires Eval collector)
        # fom_metrics.update(self._eval_foms())

        return fom_metrics

    def _ci_foms(self) -> dict[str, float]:
        """Calculate CI/CD Figure of Merit values.

        All 7 metrics have raw sources available from GitHub collector.

        Returns:
            Dict of health.ci.*.fom metrics
        """
        foms = {}

        try:
            # 1. Tests Passing on Main (binary: pass=1.0, fail=0.0)
            tests_passing = self._query_metric("github.ci.tests_passing_on_main")
            if tests_passing is not None:
                foms["health.ci.tests_passing.fom"] = 1.0 if tests_passing > 0 else 0.0

            # 2. Failing Workflows (fewer is better: 0→1.0, 5+→0.0)
            failed_workflows = self._query_metric("github.ci.failed_workflows_7d")
            if failed_workflows is not None:
                foms["health.ci.failing_workflows.fom"] = max(1.0 - (failed_workflows / 5.0), 0.0)

            # 3. Hotfix Count (fewer is better: 0→1.0, 10+→0.0)
            hotfix_count = self._query_metric("github.commits.hotfix")
            if hotfix_count is not None:
                foms["health.ci.hotfix_count.fom"] = max(1.0 - (hotfix_count / 10.0), 0.0)

            # 4. Revert Count (fewer is better: 0→1.0, 2+→0.0)
            revert_count = self._query_metric("github.commits.reverts")
            if revert_count is not None:
                foms["health.ci.revert_count.fom"] = max(1.0 - (revert_count / 2.0), 0.0)

            # 5. CI Duration P90 (faster is better: 3min→1.0, 10min+→0.0)
            ci_duration = self._query_metric("github.ci.duration_p90_minutes")
            if ci_duration is not None:
                # Clamp to [0.0, 1.0] range
                fom_value = 1.0 - (ci_duration - 3.0) / (10.0 - 3.0)
                foms["health.ci.duration_p90.fom"] = max(0.0, min(1.0, fom_value))

            # 6. Stale PRs (fewer is better: 0→1.0, 50+→0.0)
            stale_prs = self._query_metric("github.prs.stale_count_14d")
            if stale_prs is not None:
                foms["health.ci.stale_prs.fom"] = max(1.0 - (stale_prs / 50.0), 0.0)

            # 7. PR Cycle Time (faster is better: 24h→1.0, 72h+→0.0)
            cycle_time = self._query_metric("github.prs.cycle_time_hours")
            if cycle_time is not None:
                # Clamp to [0.0, 1.0] range
                fom_value = 1.0 - (cycle_time - 24.0) / (72.0 - 24.0)
                foms["health.ci.pr_cycle_time.fom"] = max(0.0, min(1.0, fom_value))

        except Exception as e:
            self.logger.error(f"Failed to calculate CI FoMs: {e}")

        return foms

    def _query_metric(self, metric_name: str, aggregation: str = "last") -> float | None:
        """Query a metric value from Datadog.

        Args:
            metric_name: Datadog metric name (e.g., "github.ci.tests_passing_on_main")
            aggregation: Aggregation type ("last", "avg", "sum", "max", "min")

        Returns:
            Most recent metric value, or None if not available
        """
        try:
            # Query Datadog for metric value from last 1 hour
            # This uses the timeseries query API
            value = self._dd_client.query_metric(metric_name, aggregation=aggregation)
            return value
        except Exception as e:
            self.logger.warning(f"Failed to query metric {metric_name}: {e}")
            return None
