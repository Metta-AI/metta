"""GitHub metrics collector for Datadog."""

import logging
import statistics
from datetime import datetime, timedelta, timezone
from typing import Any

from devops.datadog.common.base import BaseCollector
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
        """Initialize GitHub collector.

        Args:
            organization: GitHub organization name
            repository: Repository name
            github_token: GitHub API authentication token
        """
        super().__init__(name="github")
        self.organization = organization
        self.repository = repository
        self.github_token = github_token
        self.repo = f"{organization}/{repository}"

    def _get_auth_header(self) -> str:
        """Get GitHub authorization header."""
        return f"token {self.github_token}"

    def collect_metrics(self) -> dict[str, Any]:
        """Collect all GitHub metrics.

        Returns:
            Dictionary mapping metric keys to values
        """
        metrics = {}

        # Pull request metrics
        metrics.update(self._collect_pr_metrics())

        # Branch metrics
        metrics.update(self._collect_branch_metrics())

        # Commit and code metrics
        metrics.update(self._collect_commit_metrics())

        # CI/CD metrics
        metrics.update(self._collect_ci_metrics())

        # Developer metrics
        metrics.update(self._collect_developer_metrics())

        return metrics

    def _collect_pr_metrics(self) -> dict[str, Any]:
        """Collect pull request metrics."""
        metrics = {}

        try:
            # Get all open PRs
            open_prs = get_pull_requests(
                repo=self.repo,
                state="open",
                Authorization=self._get_auth_header(),
            )
            metrics["prs.open"] = len(open_prs)

            # Get PRs merged in last 7 days
            seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
            merged_prs = get_pull_requests(
                repo=self.repo,
                state="closed",
                since=seven_days_ago.isoformat(),
                Authorization=self._get_auth_header(),
            )
            merged_prs = [pr for pr in merged_prs if pr.get("merged_at")]
            metrics["prs.merged_7d"] = len(merged_prs)

            # Get PRs closed without merge in last 7 days
            closed_prs = get_pull_requests(
                repo=self.repo,
                state="closed",
                since=seven_days_ago.isoformat(),
                Authorization=self._get_auth_header(),
            )
            closed_without_merge = [pr for pr in closed_prs if not pr.get("merged_at")]
            metrics["prs.closed_without_merge_7d"] = len(closed_without_merge)

            # Calculate average time to merge
            if merged_prs:
                merge_times = []
                for pr in merged_prs:
                    created = datetime.fromisoformat(pr["created_at"].replace("Z", "+00:00"))
                    merged = datetime.fromisoformat(pr["merged_at"].replace("Z", "+00:00"))
                    hours = (merged - created).total_seconds() / 3600
                    merge_times.append(hours)

                metrics["prs.avg_time_to_merge_hours"] = sum(merge_times) / len(merge_times)
                metrics["prs.cycle_time_hours"] = sum(merge_times) / len(merge_times)
            else:
                metrics["prs.avg_time_to_merge_hours"] = None
                metrics["prs.cycle_time_hours"] = None

            # Count stale PRs (open > 14 days)
            stale_threshold = datetime.now(timezone.utc) - timedelta(days=14)
            stale_prs = [
                pr
                for pr in open_prs
                if datetime.fromisoformat(pr["created_at"].replace("Z", "+00:00")) < stale_threshold
            ]
            metrics["prs.stale_count_14d"] = len(stale_prs)

            # Review comment metrics (Note: GitHub API limitation)
            # The 'comments' field returns issue comments, not review comments
            prs_with_comments = [pr for pr in merged_prs if pr.get("comments", 0) > 0]
            if merged_prs:
                metrics["prs.with_review_comments_pct"] = (len(prs_with_comments) / len(merged_prs)) * 100
                total_comments = sum(pr.get("comments", 0) for pr in merged_prs)
                metrics["prs.avg_comments_per_pr"] = total_comments / len(merged_prs)
            else:
                metrics["prs.with_review_comments_pct"] = None
                metrics["prs.avg_comments_per_pr"] = None

        except Exception as e:
            logger.error(f"Failed to collect PR metrics: {e}", exc_info=True)
            # Return partial metrics collected so far
            for key in [
                "prs.open",
                "prs.merged_7d",
                "prs.closed_without_merge_7d",
                "prs.avg_time_to_merge_hours",
                "prs.cycle_time_hours",
                "prs.stale_count_14d",
                "prs.with_review_comments_pct",
                "prs.avg_comments_per_pr",
            ]:
                metrics.setdefault(key, 0 if "count" in key or "pct" not in key else None)

        return metrics

    def _collect_branch_metrics(self) -> dict[str, Any]:
        """Collect branch metrics."""
        metrics = {}

        try:
            branches = get_branches(
                repo=self.repo,
                Authorization=self._get_auth_header(),
            )
            # Exclude main/master from count
            active_branches = [b for b in branches if b.get("name") not in ("main", "master")]
            metrics["branches.active"] = len(active_branches)

        except Exception as e:
            logger.error(f"Failed to collect branch metrics: {e}", exc_info=True)
            metrics["branches.active"] = 0

        return metrics

    def _collect_commit_metrics(self) -> dict[str, Any]:
        """Collect commit and code change metrics."""
        metrics = {}

        try:
            seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)

            # Get commits from last 7 days
            commits = get_commits(
                repo=self.repo,
                since=seven_days_ago.isoformat(),
                Authorization=self._get_auth_header(),
            )

            metrics["commits.total_7d"] = len(commits)

            # Count hotfix and revert commits
            hotfix_count = sum(
                1 for commit in commits if "hotfix" in commit.get("commit", {}).get("message", "").lower()
            )
            revert_count = sum(
                1 for commit in commits if "revert" in commit.get("commit", {}).get("message", "").lower()
            )

            metrics["commits.hotfix"] = hotfix_count
            metrics["commits.reverts"] = revert_count

            # Collect code statistics
            total_additions = 0
            total_deletions = 0
            files_changed = set()

            # Fetch individual commits to get stats
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

                    # Track unique files changed
                    for file in commit_with_stats.get("files", []):
                        files_changed.add(file.get("filename"))

                except Exception as e:
                    logger.warning(f"Failed to get stats for commit {sha}: {e}")
                    continue

            metrics["code.lines_added_7d"] = total_additions
            metrics["code.lines_deleted_7d"] = total_deletions
            metrics["code.files_changed_7d"] = len(files_changed)

        except Exception as e:
            logger.error(f"Failed to collect commit metrics: {e}", exc_info=True)
            for key in [
                "commits.total_7d",
                "commits.hotfix",
                "commits.reverts",
                "code.lines_added_7d",
                "code.lines_deleted_7d",
                "code.files_changed_7d",
            ]:
                metrics.setdefault(key, 0)

        return metrics

    def _collect_ci_metrics(self) -> dict[str, Any]:
        """Collect CI/CD workflow metrics."""
        metrics = {}

        try:
            seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)

            # Get workflow runs from last 7 days
            workflow_runs = list_all_workflow_runs(
                repo=self.repo,
                created_after=seven_days_ago.isoformat(),
                Authorization=self._get_auth_header(),
            )

            metrics["ci.workflow_runs_7d"] = len(workflow_runs)

            # Count failed workflows
            failed_runs = [run for run in workflow_runs if run.get("conclusion") == "failure"]
            metrics["ci.failed_workflows_7d"] = len(failed_runs)

            # Check if main branch tests are passing
            # Get most recent workflow run on main branch
            main_runs = [
                run for run in workflow_runs if run.get("head_branch") == "main" and run.get("status") == "completed"
            ]
            if main_runs:
                latest_main_run = max(main_runs, key=lambda r: r.get("created_at", ""))
                metrics["ci.tests_passing_on_main"] = 1 if latest_main_run.get("conclusion") == "success" else 0
            else:
                metrics["ci.tests_passing_on_main"] = None

            # Calculate workflow duration metrics
            durations = []
            for run in workflow_runs:
                if run.get("status") != "completed":
                    continue

                created = datetime.fromisoformat(run["created_at"].replace("Z", "+00:00"))
                updated = datetime.fromisoformat(run["updated_at"].replace("Z", "+00:00"))
                duration_minutes = (updated - created).total_seconds() / 60
                durations.append(duration_minutes)

            if durations:
                metrics["ci.avg_workflow_duration_minutes"] = sum(durations) / len(durations)

                # Calculate percentiles
                sorted_durations = sorted(durations)
                metrics["ci.duration_p50_minutes"] = statistics.median(sorted_durations)

                # Use quantiles for p90 and p99
                if len(sorted_durations) >= 10:
                    quantiles = statistics.quantiles(sorted_durations, n=100)
                    metrics["ci.duration_p90_minutes"] = quantiles[89]  # 90th percentile
                    metrics["ci.duration_p99_minutes"] = quantiles[98]  # 99th percentile
                else:
                    # Not enough data for accurate percentiles
                    metrics["ci.duration_p90_minutes"] = None
                    metrics["ci.duration_p99_minutes"] = None
            else:
                metrics["ci.avg_workflow_duration_minutes"] = None
                metrics["ci.duration_p50_minutes"] = None
                metrics["ci.duration_p90_minutes"] = None
                metrics["ci.duration_p99_minutes"] = None

        except Exception as e:
            logger.error(f"Failed to collect CI metrics: {e}", exc_info=True)
            for key in [
                "ci.workflow_runs_7d",
                "ci.failed_workflows_7d",
                "ci.tests_passing_on_main",
                "ci.avg_workflow_duration_minutes",
                "ci.duration_p50_minutes",
                "ci.duration_p90_minutes",
                "ci.duration_p99_minutes",
            ]:
                if "count" in key or "runs" in key:
                    metrics.setdefault(key, 0)
                else:
                    metrics.setdefault(key, None)

        return metrics

    def _collect_developer_metrics(self) -> dict[str, Any]:
        """Collect developer activity metrics."""
        metrics = {}

        try:
            seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)

            # Get commits from last 7 days
            commits = get_commits(
                repo=self.repo,
                since=seven_days_ago.isoformat(),
                Authorization=self._get_auth_header(),
            )

            # Count unique developers
            developers = set()
            for commit in commits:
                author = commit.get("commit", {}).get("author", {}).get("email")
                if author:
                    developers.add(author)

            metrics["developers.active_7d"] = len(developers)

            # Calculate commits per developer
            if developers:
                metrics["commits.per_developer_7d"] = len(commits) / len(developers)
            else:
                metrics["commits.per_developer_7d"] = 0

        except Exception as e:
            logger.error(f"Failed to collect developer metrics: {e}", exc_info=True)
            metrics["developers.active_7d"] = 0
            metrics["commits.per_developer_7d"] = 0

        return metrics
