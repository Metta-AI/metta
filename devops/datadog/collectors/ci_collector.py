from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Set

from devops.datadog.collectors.base import BaseCollector
from devops.datadog.github_client import GitHubClient
from devops.datadog.models import MetricKind, MetricSample
from devops.datadog.utils import parse_github_timestamp, percentile, utcnow
from softmax.aws.secrets_manager import get_secretsmanager_secret


@dataclass
class CICollectorConfig:
    repo: str
    branch: str
    hotfix_labels: List[str]
    force_merge_labels: List[str]
    revert_labels: List[str]
    hours_to_consider: int
    weekly_window_days: int
    flaky_threshold: int
    duration_p90_minutes: float
    cancelled_threshold: int
    hotfix_threshold: int
    force_merge_threshold: int
    revert_threshold: int
    # Workflow names for specific checks (configurable via env vars)
    tests_blocking_merge_workflows: List[str]
    benchmarks_workflows: List[str]

    @staticmethod
    def from_env() -> "CICollectorConfig":
        return CICollectorConfig(
            repo=os.environ.get("METTA_GITHUB_REPO", "Metta-AI/metta"),
            branch=os.environ.get("METTA_GITHUB_BRANCH", "main"),
            hotfix_labels=_split_env("METTA_GITHUB_HOTFIX_LABELS", default="hotfix"),
            force_merge_labels=_split_env("METTA_GITHUB_FORCE_MERGE_LABELS", default="force-merge,force_merge"),
            revert_labels=_split_env("METTA_GITHUB_REVERT_LABELS", default="revert"),
            hours_to_consider=int(os.environ.get("CI_HISTORY_WINDOW_HOURS", "24")),
            weekly_window_days=int(os.environ.get("CI_WEEKLY_WINDOW_DAYS", "7")),
            flaky_threshold=int(os.environ.get("CI_FLAKY_TEST_THRESHOLD", "10")),
            duration_p90_minutes=float(os.environ.get("CI_P90_DURATION_MINUTES", "5")),
            cancelled_threshold=int(os.environ.get("CI_CANCELLED_THRESHOLD", "10")),
            hotfix_threshold=int(os.environ.get("CI_HOTFIX_THRESHOLD", "5")),
            force_merge_threshold=int(os.environ.get("CI_FORCE_MERGE_THRESHOLD", "7")),
            revert_threshold=int(os.environ.get("CI_REVERT_THRESHOLD", "1")),
            tests_blocking_merge_workflows=_split_env("CI_TESTS_BLOCKING_MERGE_WORKFLOWS", default=""),
            benchmarks_workflows=_split_env("CI_BENCHMARKS_WORKFLOWS", default=""),
        )


def _split_env(key: str, default: str = "") -> List[str]:
    value = os.environ.get(key, default)
    if not value:
        return []
    return [token.strip() for token in value.split(",") if token.strip()]


class CICollector(BaseCollector):
    slug = "ci"
    metric_namespace = "metta.infra.cron"
    workflow_category = "ci"

    def __init__(self) -> None:
        super().__init__()
        self.config = CICollectorConfig.from_env()
        token = (
            os.environ.get("GITHUB_DASHBOARD_TOKEN")
            or os.environ.get("GITHUB_TOKEN")
            or get_secretsmanager_secret("github/dashboard-token", require_exists=False)
        )
        if not token:
            self.logger.warning(
                "No GitHub token found in environment or AWS Secrets Manager. "
                "API calls will be rate-limited (60 req/hour)."
            )
        else:
            self.logger.debug("GitHub token loaded successfully")
        self.github = GitHubClient(token=token)

    def collect(self) -> list[MetricSample]:
        self.logger.info("Collecting CI metrics for repo=%s branch=%s", self.config.repo, self.config.branch)
        runs = self.github.list_workflow_runs(
            self.config.repo,
            per_page=100,
            branch=self.config.branch,
            status="completed",
        )
        return (
            self._build_run_metrics(runs)
            + self._build_workflow_status_metrics()
            + self._build_weekly_metrics(
                label_groups={
                    "hotfix": (self.config.hotfix_labels, self.config.hotfix_threshold, "hotfix_commits"),
                    "force_merge": (self.config.force_merge_labels, self.config.force_merge_threshold, "force_merges"),
                    "revert": (self.config.revert_labels, self.config.revert_threshold, "reverts"),
                }
            )
        )

    def _build_run_metrics(self, runs: List[Dict]) -> List[MetricSample]:
        cutoff = utcnow() - timedelta(hours=self.config.hours_to_consider)
        durations: List[float] = []
        flaky_runs = 0
        cancelled_runs = 0
        successful_runs = 0
        total_runs = 0

        for run in runs:
            run_started_at = run.get("run_started_at") or run.get("created_at")
            updated_at = run.get("updated_at")
            if not run_started_at or not updated_at:
                continue
            started = parse_github_timestamp(run_started_at)
            if started < cutoff:
                continue
            updated = parse_github_timestamp(updated_at)
            total_runs += 1
            if run.get("run_attempt", 1) > 1:
                flaky_runs += 1

            if run.get("conclusion") == "cancelled":
                cancelled_runs += 1

            if run.get("conclusion") == "success":
                successful_runs += 1

            durations.append((updated - started).total_seconds() / 60.0)

        p90_duration = percentile(sorted(durations), 90)
        success_ratio = (successful_runs / total_runs) if total_runs else 0.0

        samples: List[MetricSample] = []

        samples.append(
            self.build_sample(
                metric="ci.workflow.flaky_tests",
                value=flaky_runs,
                workflow_name="latest_state_of_main",
                task="tests_blocking_merge",
                check="flaky_tests",
                condition=f"< {self.config.flaky_threshold}",
                status="pass" if flaky_runs < self.config.flaky_threshold else "fail",
                metric_kind=MetricKind.COUNT,
            )
        )

        samples.append(
            self.build_sample(
                metric="ci.workflow.duration.p90",
                value=p90_duration,
                workflow_name="ci_smoothness",
                task="workflow_duration",
                check="p90_minutes",
                condition=f"< {self.config.duration_p90_minutes}",
                status="pass" if p90_duration < self.config.duration_p90_minutes else "fail",
            )
        )

        samples.append(
            self.build_sample(
                metric="ci.workflow.cancelled",
                value=cancelled_runs,
                workflow_name="ci_smoothness",
                task="jobs_cancelled",
                check="count",
                condition=f"< {self.config.cancelled_threshold}",
                status="pass" if cancelled_runs < self.config.cancelled_threshold else "fail",
                metric_kind=MetricKind.COUNT,
            )
        )

        samples.append(
            self.build_sample(
                metric="ci.workflow.success",
                value=1.0 if success_ratio == 1.0 else 0.0,
                workflow_name="latest_state_of_main",
                task="tests_blocking_merge",
                check="success_ratio",
                condition=">= 1",
                status="pass" if success_ratio == 1.0 else "fail",
            )
        )

        self.logger.info(
            "CI window stats: runs=%s flaky=%s cancelled=%s p90=%.2f success_ratio=%.2f",
            total_runs,
            flaky_runs,
            cancelled_runs,
            p90_duration,
            success_ratio,
        )
        return samples

    def _build_weekly_metrics(
        self,
        label_groups: Dict[str, tuple[List[str], int, str]],
    ) -> List[MetricSample]:
        since = (utcnow() - timedelta(days=self.config.weekly_window_days)).date().isoformat()
        samples: List[MetricSample] = []

        # Map task names to metric suffixes per the plan
        metric_suffix_map = {
            "hotfix_commits": "github.hotfix.count",
            "force_merges": "github.force_merge.count",
            "reverts": "github.reverts.count",
        }

        for _metric_name, (labels, threshold, task) in label_groups.items():
            count = self._count_prs_by_labels(labels, since)
            metric_suffix = metric_suffix_map.get(task, f"github.{task}.count")
            samples.append(
                self.build_sample(
                    metric=metric_suffix,
                    value=count,
                    workflow_name="commit_history",
                    task=task,
                    check="weekly_count",
                    condition=f"< {threshold}",
                    status="pass" if count < threshold else "fail",
                    metric_kind=MetricKind.COUNT,
                )
            )
            self.logger.info("Weekly %s count=%s since=%s labels=%s", task, count, since, labels)

        return samples

    def _build_workflow_status_metrics(self) -> List[MetricSample]:
        """Build metrics for specific workflow status checks per Nishad's plan."""
        samples: List[MetricSample] = []

        # 1. Tests that block merge passing
        if self.config.tests_blocking_merge_workflows:
            passing = self._check_workflows_passing(self.config.tests_blocking_merge_workflows)
            samples.append(
                self.build_sample(
                    metric="ci.workflow.tests_blocking_merge",
                    value=1.0 if passing else 0.0,
                    workflow_name="latest_state_of_main",
                    task="tests_blocking_merge",
                    check="workflow_passing",
                    condition="> 0",
                    status="pass" if passing else "fail",
                )
            )

        # 2. Benchmarks passing
        if self.config.benchmarks_workflows:
            passing = self._check_workflows_passing(self.config.benchmarks_workflows)
            samples.append(
                self.build_sample(
                    metric="ci.workflow.benchmarks",
                    value=1.0 if passing else 0.0,
                    workflow_name="latest_state_of_main",
                    task="benchmarks",
                    check="workflow_passing",
                    condition="> 0",
                    status="pass" if passing else "fail",
                )
            )

        # 3. Num other workflows whose latest run off main is failing
        failing_count = self._count_other_workflows_failing()
        samples.append(
            self.build_sample(
                metric="ci.workflow.other_failing",
                value=float(failing_count),
                workflow_name="latest_state_of_main",
                task="other_workflows",
                check="failing_count",
                condition="< 2",
                status="pass" if failing_count < 2 else "fail",
                metric_kind=MetricKind.COUNT,
            )
        )

        return samples

    def _check_workflows_passing(self, workflow_names: List[str]) -> bool:
        """Check if all specified workflows have passing latest runs."""
        if not workflow_names:
            return False

        for workflow_name in workflow_names:
            # Try by name first, then by ID if name looks like a number
            if workflow_name.isdigit():
                latest_run = self.github.get_latest_workflow_run(
                    self.config.repo,
                    workflow_id=int(workflow_name),
                    branch=self.config.branch,
                )
            else:
                latest_run = self.github.get_latest_workflow_run(
                    self.config.repo,
                    workflow_name=workflow_name,
                    branch=self.config.branch,
                )
            if not latest_run:
                self.logger.warning("No latest run found for workflow: %s", workflow_name)
                return False
            conclusion = latest_run.get("conclusion")
            if conclusion != "success":
                self.logger.info(
                    "Workflow %s latest run conclusion: %s (not passing)",
                    workflow_name,
                    conclusion,
                )
                return False

        return True

    def _count_other_workflows_failing(self) -> int:
        """Count workflows (excluding tests/benchmarks) whose latest run is failing."""
        all_workflows = self.github.list_workflows(self.config.repo)
        exclude_workflows = set(self.config.tests_blocking_merge_workflows) | set(self.config.benchmarks_workflows)

        failing_count = 0
        for workflow in all_workflows:
            workflow_name = workflow.get("name") or workflow.get("path", "").split("/")[-1]
            workflow_id = str(workflow.get("id", ""))

            # Skip if this workflow matches by name OR id
            if workflow_name in exclude_workflows or workflow_id in exclude_workflows:
                continue

            # Get latest run for this workflow
            latest_run = self.github.get_latest_workflow_run(
                self.config.repo,
                workflow_id=workflow_id,
                branch=self.config.branch,
            )
            if not latest_run:
                continue

            conclusion = latest_run.get("conclusion")
            if conclusion and conclusion not in ("success", "cancelled"):
                # Failed, skipped, or other non-success status
                failing_count += 1
                self.logger.debug("Workflow %s is failing (conclusion: %s)", workflow_name, conclusion)

        return failing_count

    def _count_prs_by_labels(self, labels: List[str], since_iso_date: str) -> int:
        if not labels:
            return 0
        found: Set[int] = set()
        for label in labels:
            query = f'repo:{self.config.repo} is:pr is:merged label:"{label}" merged:>={since_iso_date}'
            data = self.github.search_issues(query)
            for item in data.get("items", []):
                found.add(item["number"])
        return len(found)
