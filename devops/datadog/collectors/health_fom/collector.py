"""Health Figure of Merit (FoM) collector for Datadog monitoring.

This collector reads raw metrics from Datadog, applies normalization formulas,
and emits health.*.fom metrics (0.0-1.0 scale) for the System Health Dashboard.

Phase 1: CI/CD metrics (7 FoMs)
Future phases: Training metrics (WandB), Eval metrics
"""

import os
from typing import Any

from devops.datadog.utils.base import BaseCollector
from devops.datadog.utils.datadog_client import DatadogClient
from metta.common.util.aws_secrets import get_secretsmanager_secret


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
        api_key = os.getenv("DD_API_KEY") or get_secretsmanager_secret("datadog/api-key")
        app_key = os.getenv("DD_APP_KEY") or get_secretsmanager_secret("datadog/app-key")
        site = os.getenv("DD_SITE", "datadoghq.com")
        self._dd_client = DatadogClient(api_key=api_key, app_key=app_key, site=site)

    def collect_metrics(self) -> dict[str, Any]:
        """Calculate all FoM metrics from raw sources.

        Returns:
            Dict of health.*.fom metrics with values in [0.0, 1.0]
        """
        fom_metrics = {}

        try:
            # Phase 1: CI/CD FoMs (7 metrics, all raw sources available from GitHub collector)
            ci_foms = self._ci_foms()
            if ci_foms:
                fom_metrics.update(ci_foms)
                self.logger.info(f"Collected {len(ci_foms)} CI FoM metrics")
            else:
                self.logger.warning("No CI FoM metrics available")

            # Phase 2: Training FoMs (using available WandB metrics)
            training_foms = self._training_foms()
            if training_foms:
                fom_metrics.update(training_foms)
                self.logger.info(f"Collected {len(training_foms)} Training FoM metrics")
            else:
                self.logger.warning("No Training FoM metrics available")

            # Phase 3: Eval FoMs (5 metrics, requires Eval collector)
            # fom_metrics.update(self._eval_foms())

        except Exception as e:
            self.logger.error(f"Failed to collect FoM metrics: {e}")
            # Return empty dict on error - don't crash the entire collector job
            return {}

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

    def _training_foms(self) -> dict[str, float]:
        """Calculate Training Figure of Merit values.

        Uses available WandB metrics to assess training health.

        Returns:
            Dict of health.training.*.fom metrics
        """
        foms = {}

        try:
            # 1. Training Run Success (at least 1 completed run in 7 days)
            completed_runs = self._query_metric("wandb.runs.completed_7d")
            if completed_runs is not None:
                # Target: at least 7 runs per week (1 per day)
                foms["health.training.run_success.fom"] = min(completed_runs / 7.0, 1.0)

            # 2. Training Run Failures (fewer is better)
            failed_runs = self._query_metric("wandb.runs.failed_7d")
            if failed_runs is not None:
                # Target: 0 failures, tolerate up to 3
                foms["health.training.run_failures.fom"] = max(1.0 - (failed_runs / 3.0), 0.0)

            # 3. Model Performance - Best Accuracy (higher is better)
            best_accuracy = self._query_metric("wandb.metrics.best_accuracy")
            if best_accuracy is not None:
                # Normalize to 0-1 range (assuming accuracy is already a percentage 0-100 or ratio 0-1)
                # If accuracy is 0-100, divide by 100. If 0-1, use as-is
                if best_accuracy > 1.0:
                    # Assume percentage (0-100)
                    foms["health.training.best_accuracy.fom"] = min(best_accuracy / 100.0, 1.0)
                else:
                    # Already a ratio (0-1)
                    foms["health.training.best_accuracy.fom"] = best_accuracy

            # 4. Model Performance - Average Accuracy (7 days)
            avg_accuracy = self._query_metric("wandb.metrics.avg_accuracy_7d")
            if avg_accuracy is not None:
                if avg_accuracy > 1.0:
                    foms["health.training.avg_accuracy.fom"] = min(avg_accuracy / 100.0, 1.0)
                else:
                    foms["health.training.avg_accuracy.fom"] = avg_accuracy

            # 5. Training Loss (lower is better, inverse metric)
            latest_loss = self._query_metric("wandb.metrics.latest_loss")
            if latest_loss is not None and latest_loss > 0:
                # Normalize: loss of 0.1→1.0 (excellent), 1.0→0.5 (ok), 2.0+→0.0 (poor)
                fom_value = 1.0 - (latest_loss - 0.1) / (2.0 - 0.1)
                foms["health.training.latest_loss.fom"] = max(0.0, min(1.0, fom_value))

            # 6. GPU Utilization (higher is better)
            gpu_util = self._query_metric("wandb.training.gpu_utilization_avg")
            if gpu_util is not None:
                # Target: >80% utilization
                # 80%→1.0, 50%→0.63, 0%→0.0
                foms["health.training.gpu_utilization.fom"] = min(gpu_util / 80.0, 1.0)

            # 7. Training Duration Consistency (faster is better, but consistent)
            avg_duration = self._query_metric("wandb.training.avg_duration_hours")
            if avg_duration is not None:
                # Target: 2-8 hours per run
                # Too fast (<1h) might indicate incomplete runs
                # Too slow (>12h) might indicate issues
                if avg_duration < 1.0:
                    # Suspiciously fast
                    foms["health.training.duration.fom"] = 0.3
                elif avg_duration <= 8.0:
                    # Optimal range
                    foms["health.training.duration.fom"] = 1.0
                else:
                    # Too slow: 8h→1.0, 16h→0.0
                    fom_value = 1.0 - (avg_duration - 8.0) / (16.0 - 8.0)
                    foms["health.training.duration.fom"] = max(0.0, min(1.0, fom_value))

        except Exception as e:
            self.logger.error(f"Failed to calculate Training FoMs: {e}")

        return foms

    def _query_metric(self, metric_name: str, aggregation: str = "avg") -> float | None:
        """Query a metric value from Datadog.

        Args:
            metric_name: Datadog metric name (e.g., "github.ci.tests_passing_on_main")
            aggregation: Aggregation type ("last", "avg", "sum", "max", "min")

        Returns:
            Most recent metric value, or None if not available
        """
        lookback_seconds = 14400  # 4 hours
        self.logger.info(f"Querying metric: {metric_name} (aggregation={aggregation}, lookback={lookback_seconds}s)")

        try:
            value = self._dd_client.query_metric(
                metric_name,
                aggregation=aggregation,
                lookback_seconds=lookback_seconds,
            )

            if value is not None:
                self.logger.info(f"✓ Found {metric_name} = {value}")
            else:
                self.logger.warning(f"✗ No data returned for {metric_name}")

            return value
        except Exception as e:
            self.logger.error(f"✗ Failed to query {metric_name}: {type(e).__name__}: {e}")
            return None
