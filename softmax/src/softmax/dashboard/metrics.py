import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel

from gitta import get_commits, get_workflow_run_jobs, get_workflow_runs
from metta.common.util.constants import METTA_GITHUB_ORGANIZATION, METTA_GITHUB_REPO
from softmax.aws.secrets_manager import get_secretsmanager_secret
from softmax.dashboard.registry import system_health_metric

logger = logging.getLogger(__name__)


def _get_github_token() -> str:
    """Get GitHub token from AWS Secrets Manager."""
    # Auth to avoid rate limiting
    # We should replace this with a PAT before Dec 13 2026
    return get_secretsmanager_secret("github/dashboard-token")


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
            Authorization=f"Basic {_get_github_token()}",
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
            Authorization=f"Basic {_get_github_token()}",
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
            Authorization=f"Basic {_get_github_token()}",
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
