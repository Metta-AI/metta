"""GitHub metrics collector for Datadog."""

import logging
import statistics
from datetime import datetime, timedelta, timezone
from typing import Any

from devops.datadog.utils.base import BaseCollector
from gitta import (
    get_branches,
    get_commit_with_stats,
    get_commits,
    get_pull_requests,
    list_all_workflow_runs,
)

logger = logging.getLogger(__name__)


class GitHubCollector(BaseCollector):
    """Collector for GitHub repository metrics.

    Collects metrics about pull requests, commits, CI/CD workflows,
    branches, and developer activity.
    """

    def __init__(
        self,
        organization: str,
        repository: str,
        github_token: str,
    ):
        super().__init__(name="github")
        self.organization = organization
        self.repository = repository
        self.github_token = github_token
        self.repo = f"{organization}/{repository}"

    def _get_auth_header(self) -> str:
        return f"token {self.github_token}"

    def collect_metrics(self) -> dict[str, Any]:
        metrics = {}

        metrics.update(self._collect_pr_metrics())
        metrics.update(self._collect_branch_metrics())
        metrics.update(self._collect_commit_metrics())
        metrics.update(self._collect_ci_metrics())
        metrics.update(self._collect_developer_metrics())

        return metrics

    def _collect_pr_metrics(self) -> dict[str, Any]:
        metrics = {}

        try:
            open_prs = get_pull_requests(
                repo=self.repo,
                state="open",
                Authorization=self._get_auth_header(),
            )
            metrics["github.prs.open"] = len(open_prs)

            seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
            merged_prs = get_pull_requests(
                repo=self.repo,
                state="closed",
                since=seven_days_ago.isoformat(),
                Authorization=self._get_auth_header(),
            )
            merged_prs = [pr for pr in merged_prs if pr.get("merged_at")]
            metrics["github.prs.merged_7d"] = len(merged_prs)

            closed_prs = get_pull_requests(
                repo=self.repo,
                state="closed",
                since=seven_days_ago.isoformat(),
                Authorization=self._get_auth_header(),
            )
            closed_without_merge = [pr for pr in closed_prs if not pr.get("merged_at")]
            metrics["github.prs.closed_without_merge_7d"] = len(closed_without_merge)

            if merged_prs:
                merge_times = []
                for pr in merged_prs:
                    created = datetime.fromisoformat(pr["created_at"].replace("Z", "+00:00"))
                    merged = datetime.fromisoformat(pr["merged_at"].replace("Z", "+00:00"))
                    hours = (merged - created).total_seconds() / 3600
                    merge_times.append(hours)

                metrics["github.prs.avg_time_to_merge_hours"] = sum(merge_times) / len(merge_times)
                metrics["github.prs.cycle_time_hours"] = sum(merge_times) / len(merge_times)
            else:
                metrics["github.prs.avg_time_to_merge_hours"] = None
                metrics["github.prs.cycle_time_hours"] = None

            stale_threshold = datetime.now(timezone.utc) - timedelta(days=14)
            stale_prs = [
                pr
                for pr in open_prs
                if datetime.fromisoformat(pr["created_at"].replace("Z", "+00:00")) < stale_threshold
            ]
            metrics["github.prs.stale_count_14d"] = len(stale_prs)

            # Note: GitHub API limitation - 'comments' field returns issue comments, not review comments
            prs_with_comments = [pr for pr in merged_prs if pr.get("comments", 0) > 0]
            if merged_prs:
                metrics["github.prs.with_review_comments_pct"] = (len(prs_with_comments) / len(merged_prs)) * 100
                total_comments = sum(pr.get("comments", 0) for pr in merged_prs)
                metrics["github.prs.avg_comments_per_pr"] = total_comments / len(merged_prs)
            else:
                metrics["github.prs.with_review_comments_pct"] = None
                metrics["github.prs.avg_comments_per_pr"] = None

        except Exception as e:
            logger.error(f"Failed to collect PR metrics: {e}", exc_info=True)
            for key in [
                "github.prs.open",
                "github.prs.merged_7d",
                "github.prs.closed_without_merge_7d",
                "github.prs.avg_time_to_merge_hours",
                "github.prs.cycle_time_hours",
                "github.prs.stale_count_14d",
                "github.prs.with_review_comments_pct",
                "github.prs.avg_comments_per_pr",
            ]:
                metrics.setdefault(key, 0 if "count" in key or "pct" not in key else None)

        return metrics

    def _collect_branch_metrics(self) -> dict[str, Any]:
        metrics = {}

        try:
            branches = get_branches(
                repo=self.repo,
                Authorization=self._get_auth_header(),
            )
            active_branches = [b for b in branches if b.get("name") not in ("main", "master")]
            metrics["github.branches.active"] = len(active_branches)

        except Exception as e:
            logger.error(f"Failed to collect branch metrics: {e}", exc_info=True)
            metrics["github.branches.active"] = 0

        return metrics

    def _collect_commit_metrics(self) -> dict[str, Any]:
        metrics = {}

        try:
            seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)

            commits = get_commits(
                repo=self.repo,
                since=seven_days_ago.isoformat(),
                Authorization=self._get_auth_header(),
            )

            metrics["github.commits.total_7d"] = len(commits)

            hotfix_count = sum(
                1 for commit in commits if "hotfix" in commit.get("commit", {}).get("message", "").lower()
            )
            revert_count = sum(
                1 for commit in commits if "revert" in commit.get("commit", {}).get("message", "").lower()
            )

            metrics["github.commits.hotfix"] = hotfix_count
            metrics["github.commits.reverts"] = revert_count

            force_merge_count = sum(
                1
                for commit in commits
                if any(
                    keyword in commit.get("commit", {}).get("message", "").lower()
                    for keyword in ["force push", "force merge", "force-push", "force-merge"]
                )
            )
            metrics["github.commits.force_merge_7d"] = force_merge_count

            total_additions = 0
            total_deletions = 0
            files_changed = set()

            for commit in commits:
                sha = commit.get("sha")
                if not sha:
                    continue

                try:
                    commit_with_stats = get_commit_with_stats(
                        repo=self.repo,
                        sha=sha,
                        Authorization=self._get_auth_header(),
                    )
                    stats = commit_with_stats.get("stats", {})
                    total_additions += stats.get("additions", 0)
                    total_deletions += stats.get("deletions", 0)

                    for file in commit_with_stats.get("files", []):
                        files_changed.add(file.get("filename"))

                except Exception as e:
                    logger.warning(f"Failed to get stats for commit {sha}: {e}")
                    continue

            metrics["github.code.lines_added_7d"] = total_additions
            metrics["github.code.lines_deleted_7d"] = total_deletions
            metrics["github.code.files_changed_7d"] = len(files_changed)

        except Exception as e:
            logger.error(f"Failed to collect commit metrics: {e}", exc_info=True)
            for key in [
                "github.commits.total_7d",
                "github.commits.hotfix",
                "github.commits.reverts",
                "github.commits.force_merge_7d",
                "github.code.lines_added_7d",
                "github.code.lines_deleted_7d",
                "github.code.files_changed_7d",
            ]:
                metrics.setdefault(key, 0)

        return metrics

    def _collect_ci_metrics(self) -> dict[str, Any]:
        metrics = {}

        try:
            seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)

            workflow_runs = list_all_workflow_runs(
                repo=self.repo,
                created=seven_days_ago.isoformat(),
                Authorization=self._get_auth_header(),
            )

            metrics["github.ci.workflow_runs_7d"] = len(workflow_runs)

            failed_runs = [run for run in workflow_runs if run.get("conclusion") == "failure"]
            metrics["github.ci.failed_workflows_7d"] = len(failed_runs)

            cancelled_runs = [run for run in workflow_runs if run.get("conclusion") == "cancelled"]
            metrics["github.ci.timeout_cancellations_7d"] = len(cancelled_runs)

            # A workflow is considered flaky if it has run_attempt > 1
            flaky_runs = [run for run in workflow_runs if run.get("run_attempt", 1) > 1]
            metrics["github.ci.flaky_checks_7d"] = len(flaky_runs)

            benchmark_runs = [
                run
                for run in workflow_runs
                if "benchmark" in run.get("name", "").lower() and run.get("status") == "completed"
            ]
            if benchmark_runs:
                main_benchmark_runs = [run for run in benchmark_runs if run.get("head_branch") == "main"]
                if main_benchmark_runs:
                    latest_benchmark = max(main_benchmark_runs, key=lambda r: r.get("created_at", ""))
                    metrics["github.ci.benchmarks_passing"] = (
                        1 if latest_benchmark.get("conclusion") == "success" else 0
                    )
                else:
                    metrics["github.ci.benchmarks_passing"] = None
            else:
                metrics["github.ci.benchmarks_passing"] = None

            main_runs = [
                run for run in workflow_runs if run.get("head_branch") == "main" and run.get("status") == "completed"
            ]
            if main_runs:
                latest_main_run = max(main_runs, key=lambda r: r.get("created_at", ""))
                metrics["github.ci.tests_passing_on_main"] = 1 if latest_main_run.get("conclusion") == "success" else 0
            else:
                metrics["github.ci.tests_passing_on_main"] = None

            durations = []
            for run in workflow_runs:
                if run.get("status") != "completed":
                    continue

                created = datetime.fromisoformat(run["created_at"].replace("Z", "+00:00"))
                updated = datetime.fromisoformat(run["updated_at"].replace("Z", "+00:00"))
                duration_minutes = (updated - created).total_seconds() / 60
                durations.append(duration_minutes)

            if durations:
                metrics["github.ci.avg_workflow_duration_minutes"] = sum(durations) / len(durations)

                sorted_durations = sorted(durations)
                metrics["github.ci.duration_p50_minutes"] = statistics.median(sorted_durations)

                if len(sorted_durations) >= 10:
                    quantiles = statistics.quantiles(sorted_durations, n=100)
                    metrics["github.ci.duration_p90_minutes"] = quantiles[89]
                    metrics["github.ci.duration_p99_minutes"] = quantiles[98]
                else:
                    # Not enough data for accurate percentiles
                    metrics["github.ci.duration_p90_minutes"] = None
                    metrics["github.ci.duration_p99_minutes"] = None
            else:
                metrics["github.ci.avg_workflow_duration_minutes"] = None
                metrics["github.ci.duration_p50_minutes"] = None
                metrics["github.ci.duration_p90_minutes"] = None
                metrics["github.ci.duration_p99_minutes"] = None

        except Exception as e:
            logger.error(f"Failed to collect CI metrics: {e}", exc_info=True)
            for key in [
                "github.ci.workflow_runs_7d",
                "github.ci.failed_workflows_7d",
                "github.ci.timeout_cancellations_7d",
                "github.ci.flaky_checks_7d",
                "github.ci.tests_passing_on_main",
                "github.ci.benchmarks_passing",
                "github.ci.avg_workflow_duration_minutes",
                "github.ci.duration_p50_minutes",
                "github.ci.duration_p90_minutes",
                "github.ci.duration_p99_minutes",
            ]:
                if "count" in key or "runs" in key or "cancellations" in key or "checks" in key:
                    metrics.setdefault(key, 0)
                else:
                    metrics.setdefault(key, None)

        return metrics

    def _collect_developer_metrics(self) -> dict[str, Any]:
        metrics = {}

        try:
            seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)

            commits = get_commits(
                repo=self.repo,
                since=seven_days_ago.isoformat(),
                Authorization=self._get_auth_header(),
            )

            developers = set()
            for commit in commits:
                author = commit.get("commit", {}).get("author", {}).get("email")
                if author:
                    developers.add(author)

            metrics["github.developers.active_7d"] = len(developers)

            if developers:
                metrics["github.commits.per_developer_7d"] = len(commits) / len(developers)
            else:
                metrics["github.commits.per_developer_7d"] = 0

        except Exception as e:
            logger.error(f"Failed to collect developer metrics: {e}", exc_info=True)
            metrics["github.developers.active_7d"] = 0
            metrics["github.commits.per_developer_7d"] = 0

        return metrics
