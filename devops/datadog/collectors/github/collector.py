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

            seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
            merged_prs = get_pull_requests(
                repo=self.repo,
                state="closed",
                since=seven_days_ago.isoformat(),
                Authorization=self._get_auth_header(),
            )
            merged_prs = [pr for pr in merged_prs if pr.get("merged_at")]

            closed_prs = get_pull_requests(
                repo=self.repo,
                state="closed",
                since=seven_days_ago.isoformat(),
                Authorization=self._get_auth_header(),
            )
            closed_without_merge = [pr for pr in closed_prs if not pr.get("merged_at")]

            # PR count by status
            metrics["github.prs"] = [
                (len(open_prs), ["status:open"]),
                (len(merged_prs), ["status:merged", "timeframe:7d"]),
                (len(closed_without_merge), ["status:closed_without_merge", "timeframe:7d"]),
            ]

            # Per-PR time to merge
            stale_threshold = datetime.now(timezone.utc) - timedelta(days=14)
            stale_count = 0

            if "github.pr.time_to_merge_hours" not in metrics:
                metrics["github.pr.time_to_merge_hours"] = []

            for pr in merged_prs:
                created = datetime.fromisoformat(pr["created_at"].replace("Z", "+00:00"))
                merged = datetime.fromisoformat(pr["merged_at"].replace("Z", "+00:00"))
                hours = (merged - created).total_seconds() / 3600
                metrics["github.pr.time_to_merge_hours"].append((hours, [f"pr_id:{pr['number']}"]))

            for pr in open_prs:
                created = datetime.fromisoformat(pr["created_at"].replace("Z", "+00:00"))
                if created < stale_threshold:
                    stale_count += 1

            metrics["github.prs.stale"] = [(stale_count, ["threshold:14d"])]

            # Note: GitHub API limitation - 'comments' field returns issue comments, not review comments
            if "github.pr.comments" not in metrics:
                metrics["github.pr.comments"] = []

            for pr in merged_prs:
                comment_count = pr.get("comments", 0)
                if comment_count > 0:
                    metrics["github.pr.comments"].append((comment_count, [f"pr_id:{pr['number']}"]))

        except Exception as e:
            logger.error(f"Failed to collect PR metrics: {e}", exc_info=True)

        return metrics

    def _collect_branch_metrics(self) -> dict[str, Any]:
        metrics = {}

        try:
            branches = get_branches(
                repo=self.repo,
                Authorization=self._get_auth_header(),
            )
            active_branches = [b for b in branches if b.get("name") not in ("main", "master")]
            metrics["github.branches"] = [(len(active_branches), ["type:active"])]

        except Exception as e:
            logger.error(f"Failed to collect branch metrics: {e}", exc_info=True)

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

            metrics["github.commits"] = [
                (len(commits), ["timeframe:7d"]),
            ]

            hotfix_count = sum(
                1 for commit in commits if "hotfix" in commit.get("commit", {}).get("message", "").lower()
            )
            revert_count = sum(
                1 for commit in commits if "revert" in commit.get("commit", {}).get("message", "").lower()
            )
            force_merge_count = sum(
                1
                for commit in commits
                if any(
                    keyword in commit.get("commit", {}).get("message", "").lower()
                    for keyword in ["force push", "force merge", "force-push", "force-merge"]
                )
            )

            metrics["github.commits.special"] = [
                (hotfix_count, ["type:hotfix", "timeframe:7d"]),
                (revert_count, ["type:revert", "timeframe:7d"]),
                (force_merge_count, ["type:force_merge", "timeframe:7d"]),
            ]

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

            metrics["github.code.lines"] = [
                (total_additions, ["type:added", "timeframe:7d"]),
                (total_deletions, ["type:deleted", "timeframe:7d"]),
            ]
            metrics["github.code.files_changed"] = [(len(files_changed), ["timeframe:7d"])]

        except Exception as e:
            logger.error(f"Failed to collect commit metrics: {e}", exc_info=True)

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

            failed_runs = [run for run in workflow_runs if run.get("conclusion") == "failure"]
            cancelled_runs = [run for run in workflow_runs if run.get("conclusion") == "cancelled"]
            flaky_runs = [run for run in workflow_runs if run.get("run_attempt", 1) > 1]

            metrics["github.ci.runs"] = [
                (len(workflow_runs), ["timeframe:7d"]),
                (len(failed_runs), ["status:failed", "timeframe:7d"]),
                (len(cancelled_runs), ["status:cancelled", "timeframe:7d"]),
                (len(flaky_runs), ["status:flaky", "timeframe:7d"]),
            ]

            benchmark_runs = [
                run
                for run in workflow_runs
                if "benchmark" in run.get("name", "").lower() and run.get("status") == "completed"
            ]

            if benchmark_runs:
                main_benchmark_runs = [run for run in benchmark_runs if run.get("head_branch") == "main"]
                if main_benchmark_runs:
                    latest_benchmark = max(main_benchmark_runs, key=lambda r: r.get("created_at", ""))
                    passing = 1 if latest_benchmark.get("conclusion") == "success" else 0
                    metrics["github.ci.benchmarks"] = [(passing, ["branch:main", "status:latest"])]

            main_runs = [
                run for run in workflow_runs if run.get("head_branch") == "main" and run.get("status") == "completed"
            ]
            if main_runs:
                latest_main_run = max(main_runs, key=lambda r: r.get("created_at", ""))
                passing = 1 if latest_main_run.get("conclusion") == "success" else 0
                metrics["github.ci.tests"] = [(passing, ["branch:main", "status:latest"])]

            durations = []
            for run in workflow_runs:
                if run.get("status") != "completed":
                    continue

                created = datetime.fromisoformat(run["created_at"].replace("Z", "+00:00"))
                updated = datetime.fromisoformat(run["updated_at"].replace("Z", "+00:00"))
                duration_minutes = (updated - created).total_seconds() / 60
                durations.append(duration_minutes)

            if durations:
                sorted_durations = sorted(durations)

                metrics["github.ci.duration_minutes"] = [
                    (sum(durations) / len(durations), ["metric:mean"]),
                    (statistics.median(sorted_durations), ["metric:p50"]),
                ]

                if len(sorted_durations) >= 10:
                    quantiles = statistics.quantiles(sorted_durations, n=100)
                    metrics["github.ci.duration_minutes"].extend(
                        [
                            (quantiles[89], ["metric:p90"]),
                            (quantiles[98], ["metric:p99"]),
                        ]
                    )

        except Exception as e:
            logger.error(f"Failed to collect CI metrics: {e}", exc_info=True)

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

            metrics["github.developers"] = [(len(developers), ["status:active", "timeframe:7d"])]

            if developers:
                commits_per_dev = len(commits) / len(developers)
                metrics["github.commits_per_developer"] = [(commits_per_dev, ["timeframe:7d"])]

        except Exception as e:
            logger.error(f"Failed to collect developer metrics: {e}", exc_info=True)

        return metrics
