import logging
import statistics
from base64 import b64encode
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel

from gitta import (
    get_branches,
    get_commit_with_stats,
    get_commits,
    get_pull_requests,
    get_workflow_run_jobs,
    get_workflow_runs,
    list_all_workflow_runs,
)
from metta.common.util.constants import METTA_GITHUB_ORGANIZATION, METTA_GITHUB_REPO
from softmax.aws.secrets_manager import get_secretsmanager_secret
from softmax.dashboard.registry import system_health_metric

logger = logging.getLogger(__name__)


def _get_github_auth_header() -> str:
    """
    Get GitHub Basic auth header value.

    Returns base64-encoded Basic auth credentials for GitHub API.
    We should replace this PAT before January 15, 2026.
    """
    token = get_secretsmanager_secret("github/dashboard-token").strip()
    if not token:
        logger.warning("GitHub dashboard token is empty; API calls may be rate limited.")
        return ""

    # Use HTTP Basic auth (token as username, empty password) to match the CLI usage
    basic_credentials = b64encode(f"{token}:".encode("utf-8")).decode("utf-8")
    return f"Basic {basic_credentials}"


class GitHubJobStatus(BaseModel):
    status: str
    conclusion: str


def _get_job_statuses_by_name(response: dict[str, Any]) -> dict[str, GitHubJobStatus]:
    jobs: list[dict[str, Any]] = (response or {}).get("jobs", []) if response else []
    statuses: dict[str, GitHubJobStatus] = {}
    for job in jobs:
        name = str(job.get("name") or "").strip()
        if not name:
            continue
        statuses[name.lower()] = GitHubJobStatus(
            status=str(job.get("status") or "").lower(),
            conclusion=str(job.get("conclusion") or "").lower(),
        )
    return statuses


def _get_num_commits_with_phrase(phrase: str, lookback_days: int = 7, branch: str = "main") -> int:
    since = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
    repo = f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"

    try:
        commits = get_commits(
            repo=repo,
            branch=branch,
            since=since,
            per_page=100,
            Authorization=_get_github_auth_header(),
        )
    except Exception as e:
        logger.error(f"Failed to get commits: {e}")
        return 0

    count = 0
    for commit in commits:
        message = commit.get("commit", {}).get("message") or ""
        if phrase.lower() in message.lower():
            count += 1

    return count


@system_health_metric(metric_key="commits.hotfix")
def get_num_hotfix_commits() -> int:
    return _get_num_commits_with_phrase("hotfix", lookback_days=7, branch="main")


@system_health_metric(metric_key="commits.reverts")
def get_num_revert_commits() -> int:
    return _get_num_commits_with_phrase("revert", lookback_days=7, branch="main")


def get_latest_workflow_run(branch: str, workflow_filename: str) -> dict[str, Any] | None:
    repo = f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"

    try:
        runs = get_workflow_runs(
            repo=repo,
            workflow_filename=workflow_filename,
            branch=branch,
            status="completed",
            per_page=1,
            Authorization=_get_github_auth_header(),
        )
        return runs[0] if runs else None
    except Exception as e:
        logger.error(f"Failed to get workflow runs: {e}")
        return None


@system_health_metric(metric_key="ci.tests_passing_on_main")
def get_latest_unit_tests_failed() -> int | None:
    run = get_latest_workflow_run(branch="main", workflow_filename="checks.yml")
    if not run:
        return None

    run_id = run.get("id")
    if not run_id:
        logger.error(f"Failed to get run ID: {run}")
        return None

    repo = f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"

    try:
        jobs = get_workflow_run_jobs(
            repo=repo,
            run_id=run_id,
            per_page=100,
            Authorization=_get_github_auth_header(),
        )
    except Exception as e:
        logger.error(f"Failed to get jobs: {e}")
        return None

    job_statuses = _get_job_statuses_by_name({"jobs": jobs})
    unit_tests_all_packages = job_statuses.get("unit tests - all packages")
    tests = job_statuses.get("tests")
    if not unit_tests_all_packages or not tests:
        logger.error(f"No unit tests all packages or tests job statuses found: {job_statuses}")
        return 1

    # Cancelled tests can be identified by "Unit Tests - All Packages" job being cancelled and then "Tests" failing
    canceled = unit_tests_all_packages.status == "cancelled" and tests.conclusion == "failure"
    if canceled:
        return None
    return int(
        all(
            job_status.conclusion in ("success", "skipped")
            for job_status in job_statuses.values()
            if job_status.status == "completed"
        )
    )


# ==================== Pull Request Metrics ====================


@system_health_metric(metric_key="prs.open")
def get_open_prs_count() -> int:
    """Count of currently open pull requests."""
    repo = f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"

    try:
        prs = get_pull_requests(
            repo=repo,
            state="open",
            per_page=100,
            Authorization=_get_github_auth_header(),
        )
        return len(prs)
    except Exception as e:
        logger.error(f"Failed to get open PRs: {e}")
        return 0


@system_health_metric(metric_key="prs.merged_7d")
def get_merged_prs_7d() -> int:
    """Count of pull requests merged in the last 7 days."""
    since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    repo = f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"

    try:
        prs = get_pull_requests(
            repo=repo,
            state="closed",
            since=since,
            per_page=100,
            Authorization=_get_github_auth_header(),
        )
        # Filter for merged PRs (closed but merged)
        merged = [pr for pr in prs if pr.get("merged_at")]
        return len(merged)
    except Exception as e:
        logger.error(f"Failed to get merged PRs: {e}")
        return 0


@system_health_metric(metric_key="prs.closed_without_merge_7d")
def get_closed_without_merge_prs_7d() -> int:
    """Count of pull requests closed without merge in the last 7 days."""
    since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    repo = f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"

    try:
        prs = get_pull_requests(
            repo=repo,
            state="closed",
            since=since,
            per_page=100,
            Authorization=_get_github_auth_header(),
        )
        # Filter for closed but not merged PRs
        closed_no_merge = [pr for pr in prs if not pr.get("merged_at")]
        return len(closed_no_merge)
    except Exception as e:
        logger.error(f"Failed to get closed PRs: {e}")
        return 0


@system_health_metric(metric_key="prs.avg_time_to_merge_hours")
def get_avg_time_to_merge() -> float | None:
    """Average time from PR creation to merge (in hours) for PRs merged in last 7 days."""
    since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    repo = f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"

    try:
        prs = get_pull_requests(
            repo=repo,
            state="closed",
            since=since,
            per_page=100,
            Authorization=_get_github_auth_header(),
        )
        # Filter for merged PRs
        merged_prs = [pr for pr in prs if pr.get("merged_at")]

        if not merged_prs:
            return None

        times_to_merge = []
        for pr in merged_prs:
            created = datetime.fromisoformat(pr["created_at"].replace("Z", "+00:00"))
            merged = datetime.fromisoformat(pr["merged_at"].replace("Z", "+00:00"))
            hours = (merged - created).total_seconds() / 3600
            times_to_merge.append(hours)

        return sum(times_to_merge) / len(times_to_merge)
    except Exception as e:
        logger.error(f"Failed to calculate avg time to merge: {e}")
        return None


@system_health_metric(metric_key="prs.with_review_comments_pct")
def get_prs_with_review_comments_pct() -> float | None:
    """Percentage of PRs (last 7 days) that received review comments."""
    since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    repo = f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"

    try:
        prs = get_pull_requests(
            repo=repo,
            state="closed",
            since=since,
            per_page=100,
            Authorization=_get_github_auth_header(),
        )

        if not prs:
            return None

        # Count PRs with review comments (comments > 0 indicates discussion)
        prs_with_comments = sum(1 for pr in prs if pr.get("comments", 0) > 0)

        return (prs_with_comments / len(prs)) * 100.0
    except Exception as e:
        logger.error(f"Failed to calculate review comments percentage: {e}")
        return None


@system_health_metric(metric_key="prs.avg_comments_per_pr")
def get_avg_comments_per_pr() -> float | None:
    """Average number of comments per PR (last 7 days)."""
    since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    repo = f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"

    try:
        prs = get_pull_requests(
            repo=repo,
            state="closed",
            since=since,
            per_page=100,
            Authorization=_get_github_auth_header(),
        )

        if not prs:
            return None

        total_comments = sum(pr.get("comments", 0) for pr in prs)
        return total_comments / len(prs)
    except Exception as e:
        logger.error(f"Failed to calculate avg comments per PR: {e}")
        return None


@system_health_metric(metric_key="prs.time_to_first_review_hours")
def get_time_to_first_review_hours() -> float | None:
    """Average time from PR creation to first comment (in hours) for PRs in last 7 days."""
    since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    repo = f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"

    try:
        prs = get_pull_requests(
            repo=repo,
            state="closed",
            since=since,
            per_page=100,
            Authorization=_get_github_auth_header(),
        )

        if not prs:
            return None

        times_to_first_review = []
        for pr in prs:
            # Skip PRs with no comments
            if pr.get("comments", 0) == 0:
                continue

            created = datetime.fromisoformat(pr["created_at"].replace("Z", "+00:00"))
            # Note: The PR endpoint doesn't include comment timestamps
            # We'd need to fetch individual PR review comments via another API call
            # For now, we'll use a simplified approach based on updated_at
            # This is an approximation - actual implementation would need PR review timeline API
            updated = datetime.fromisoformat(pr["updated_at"].replace("Z", "+00:00"))

            # Only include if updated != created (indicating some activity)
            if updated > created:
                hours = (updated - created).total_seconds() / 3600
                times_to_first_review.append(hours)

        if not times_to_first_review:
            return None

        return sum(times_to_first_review) / len(times_to_first_review)
    except Exception as e:
        logger.error(f"Failed to calculate time to first review: {e}")
        return None


@system_health_metric(metric_key="prs.stale_count_14d")
def get_stale_prs_count() -> int:
    """Count of open PRs that have been open for more than 14 days."""
    repo = f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"
    stale_threshold = datetime.now(timezone.utc) - timedelta(days=14)

    try:
        prs = get_pull_requests(
            repo=repo,
            state="open",
            per_page=100,
            Authorization=_get_github_auth_header(),
        )

        stale_prs = []
        for pr in prs:
            created = datetime.fromisoformat(pr["created_at"].replace("Z", "+00:00"))
            if created < stale_threshold:
                stale_prs.append(pr)

        return len(stale_prs)
    except Exception as e:
        logger.error(f"Failed to get stale PRs: {e}")
        return 0


@system_health_metric(metric_key="prs.cycle_time_hours")
def get_pr_cycle_time_hours() -> float | None:
    """Average cycle time from PR creation to merge (in hours) for PRs merged in last 7 days.

    Note: This is the same as avg_time_to_merge but with a more standard DORA metrics name.
    """
    since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    repo = f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"

    try:
        prs = get_pull_requests(
            repo=repo,
            state="closed",
            since=since,
            per_page=100,
            Authorization=_get_github_auth_header(),
        )
        # Filter for merged PRs
        merged_prs = [pr for pr in prs if pr.get("merged_at")]

        if not merged_prs:
            return None

        cycle_times = []
        for pr in merged_prs:
            created = datetime.fromisoformat(pr["created_at"].replace("Z", "+00:00"))
            merged = datetime.fromisoformat(pr["merged_at"].replace("Z", "+00:00"))
            hours = (merged - created).total_seconds() / 3600
            cycle_times.append(hours)

        return sum(cycle_times) / len(cycle_times)
    except Exception as e:
        logger.error(f"Failed to calculate PR cycle time: {e}")
        return None


# ==================== Branch Metrics ====================


@system_health_metric(metric_key="branches.active")
def get_active_branches_count() -> int:
    """Count of active branches (excluding main/master)."""
    repo = f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"

    try:
        branches = get_branches(
            repo=repo,
            Authorization=_get_github_auth_header(),
        )
        # Filter out main/master branches
        active = [b for b in branches if b["name"] not in ("main", "master")]
        return len(active)
    except Exception as e:
        logger.error(f"Failed to get active branches: {e}")
        return 0


# ==================== Code Change Metrics ====================


@system_health_metric(metric_key="commits.total_7d")
def get_commits_7d() -> int:
    """Total number of commits in the last 7 days."""
    since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    repo = f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"

    try:
        commits = get_commits(
            repo=repo,
            branch="main",
            since=since,
            per_page=100,
            Authorization=_get_github_auth_header(),
        )
        return len(commits)
    except Exception as e:
        logger.error(f"Failed to get commits: {e}")
        return 0


@system_health_metric(metric_key="code.lines_added_7d")
def get_lines_added_7d() -> int:
    """Total lines of code added in the last 7 days."""
    since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    repo = f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"

    try:
        # First get the list of commits to extract SHAs
        commits = get_commits(
            repo=repo,
            branch="main",
            since=since,
            per_page=100,
            Authorization=_get_github_auth_header(),
        )

        # Fetch each commit individually to get stats
        total_additions = 0
        for commit in commits:
            sha = commit.get("sha")
            if not sha:
                continue

            # Get full commit with stats
            commit_with_stats = get_commit_with_stats(
                repo=repo,
                sha=sha,
                Authorization=_get_github_auth_header(),
            )

            stats = commit_with_stats.get("stats", {})
            total_additions += stats.get("additions", 0)

        return total_additions
    except Exception as e:
        logger.error(f"Failed to get lines added: {e}")
        return 0


@system_health_metric(metric_key="code.lines_deleted_7d")
def get_lines_deleted_7d() -> int:
    """Total lines of code deleted in the last 7 days."""
    since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    repo = f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"

    try:
        # First get the list of commits to extract SHAs
        commits = get_commits(
            repo=repo,
            branch="main",
            since=since,
            per_page=100,
            Authorization=_get_github_auth_header(),
        )

        # Fetch each commit individually to get stats
        total_deletions = 0
        for commit in commits:
            sha = commit.get("sha")
            if not sha:
                continue

            # Get full commit with stats
            commit_with_stats = get_commit_with_stats(
                repo=repo,
                sha=sha,
                Authorization=_get_github_auth_header(),
            )

            stats = commit_with_stats.get("stats", {})
            total_deletions += stats.get("deletions", 0)

        return total_deletions
    except Exception as e:
        logger.error(f"Failed to get lines deleted: {e}")
        return 0


@system_health_metric(metric_key="code.files_changed_7d")
def get_files_changed_7d() -> int:
    """Total number of files changed in the last 7 days."""
    since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    repo = f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"

    try:
        # First get the list of commits to extract SHAs
        commits = get_commits(
            repo=repo,
            branch="main",
            since=since,
            per_page=100,
            Authorization=_get_github_auth_header(),
        )

        # Use a set to track unique files
        files = set()
        for commit in commits:
            sha = commit.get("sha")
            if not sha:
                continue

            # Get full commit with stats and files list
            commit_with_stats = get_commit_with_stats(
                repo=repo,
                sha=sha,
                Authorization=_get_github_auth_header(),
            )

            files_data = commit_with_stats.get("files", [])
            for file_data in files_data:
                filename = file_data.get("filename")
                if filename:
                    files.add(filename)

        return len(files)
    except Exception as e:
        logger.error(f"Failed to get files changed: {e}")
        return 0


# ==================== CI/CD Runtime Metrics ====================


@system_health_metric(metric_key="ci.workflow_runs_7d")
def get_workflow_runs_7d() -> int:
    """Total number of workflow runs in the last 7 days."""
    since_date = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    created_filter = f">={since_date}"
    repo = f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"

    try:
        runs = list_all_workflow_runs(
            repo=repo,
            created=created_filter,
            per_page=100,
            Authorization=_get_github_auth_header(),
        )
        return len(runs)
    except Exception as e:
        logger.error(f"Failed to get workflow runs: {e}")
        return 0


@system_health_metric(metric_key="ci.failed_workflows_7d")
def get_failed_workflows_7d() -> int:
    """Number of failed workflow runs in the last 7 days."""
    since_date = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    created_filter = f">={since_date}"
    repo = f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"

    try:
        runs = list_all_workflow_runs(
            repo=repo,
            status="completed",
            created=created_filter,
            per_page=100,
            Authorization=_get_github_auth_header(),
        )
        failed = [run for run in runs if run.get("conclusion") == "failure"]
        return len(failed)
    except Exception as e:
        logger.error(f"Failed to get failed workflows: {e}")
        return 0


@system_health_metric(metric_key="ci.avg_workflow_duration_minutes")
def get_avg_workflow_duration() -> float | None:
    """Average workflow run duration in minutes for runs in the last 7 days."""
    since_date = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    created_filter = f">={since_date}"
    repo = f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"

    try:
        runs = list_all_workflow_runs(
            repo=repo,
            status="completed",
            created=created_filter,
            per_page=100,
            Authorization=_get_github_auth_header(),
        )

        if not runs:
            return None

        durations = []
        for run in runs:
            created = run.get("created_at")
            updated = run.get("updated_at")
            if created and updated:
                created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                updated_dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                duration_minutes = (updated_dt - created_dt).total_seconds() / 60
                durations.append(duration_minutes)

        if not durations:
            return None

        return sum(durations) / len(durations)
    except Exception as e:
        logger.error(f"Failed to calculate avg workflow duration: {e}")
        return None


@system_health_metric(metric_key="ci.duration_p50_minutes")
def get_ci_duration_p50() -> float | None:
    """Median (50th percentile) workflow run duration in minutes for runs in the last 7 days."""
    since_date = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    created_filter = f">={since_date}"
    repo = f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"

    try:
        runs = list_all_workflow_runs(
            repo=repo,
            status="completed",
            created=created_filter,
            per_page=100,
            Authorization=_get_github_auth_header(),
        )

        if not runs:
            return None

        durations = []
        for run in runs:
            created = run.get("created_at")
            updated = run.get("updated_at")
            if created and updated:
                created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                updated_dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                duration_minutes = (updated_dt - created_dt).total_seconds() / 60
                durations.append(duration_minutes)

        if not durations:
            return None

        return statistics.median(durations)
    except Exception as e:
        logger.error(f"Failed to calculate p50 workflow duration: {e}")
        return None


@system_health_metric(metric_key="ci.duration_p90_minutes")
def get_ci_duration_p90() -> float | None:
    """90th percentile workflow run duration in minutes for runs in the last 7 days."""
    since_date = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    created_filter = f">={since_date}"
    repo = f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"

    try:
        runs = list_all_workflow_runs(
            repo=repo,
            status="completed",
            created=created_filter,
            per_page=100,
            Authorization=_get_github_auth_header(),
        )

        if not runs:
            return None

        durations = []
        for run in runs:
            created = run.get("created_at")
            updated = run.get("updated_at")
            if created and updated:
                created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                updated_dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                duration_minutes = (updated_dt - created_dt).total_seconds() / 60
                durations.append(duration_minutes)

        if not durations:
            return None

        return statistics.quantiles(durations, n=10)[8]  # 9th decile = 90th percentile
    except Exception as e:
        logger.error(f"Failed to calculate p90 workflow duration: {e}")
        return None


@system_health_metric(metric_key="ci.duration_p99_minutes")
def get_ci_duration_p99() -> float | None:
    """99th percentile workflow run duration in minutes for runs in the last 7 days."""
    since_date = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    created_filter = f">={since_date}"
    repo = f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"

    try:
        runs = list_all_workflow_runs(
            repo=repo,
            status="completed",
            created=created_filter,
            per_page=100,
            Authorization=_get_github_auth_header(),
        )

        if not runs:
            return None

        durations = []
        for run in runs:
            created = run.get("created_at")
            updated = run.get("updated_at")
            if created and updated:
                created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                updated_dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                duration_minutes = (updated_dt - created_dt).total_seconds() / 60
                durations.append(duration_minutes)

        if not durations:
            return None

        return statistics.quantiles(durations, n=100)[98]  # 99th percentile
    except Exception as e:
        logger.error(f"Failed to calculate p99 workflow duration: {e}")
        return None


# ==================== Developer Activity Metrics ====================


@system_health_metric(metric_key="developers.active_7d")
def get_active_developers_7d() -> int:
    """Number of unique developers who committed in the last 7 days."""
    since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    repo = f"{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}"

    try:
        commits = get_commits(
            repo=repo,
            branch="main",
            since=since,
            per_page=100,
            Authorization=_get_github_auth_header(),
        )

        # Use set to track unique authors
        authors = set()
        for commit in commits:
            author = commit.get("commit", {}).get("author", {}).get("email")
            if author:
                authors.add(author)

        return len(authors)
    except Exception as e:
        logger.error(f"Failed to get active developers: {e}")
        return 0


@system_health_metric(metric_key="commits.per_developer_7d")
def get_avg_commits_per_developer_7d() -> float | None:
    """Average commits per developer in the last 7 days."""
    total_commits = get_commits_7d()
    active_devs = get_active_developers_7d()

    if active_devs == 0:
        return None

    return total_commits / active_devs
