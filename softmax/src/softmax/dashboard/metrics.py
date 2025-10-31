import logging
from base64 import b64encode
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel

from gitta import get_commits, get_workflow_run_jobs, get_workflow_runs
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
        logger.error(f"Failed to get commits: {e}", exc_info=True)
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
        logger.error(f"Failed to get workflow runs: {e}", exc_info=True)
        return None


@system_health_metric(metric_key="ci.tests_passing_on_main")
def get_latest_unit_tests_failed() -> int | None:
    run = get_latest_workflow_run(branch="main", workflow_filename="checks.yml")
    if not run:
        return None

    run_id = run.get("id")
    if not run_id:
        logger.error(f"Failed to get run ID: {run}", exc_info=True)
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
        logger.error(f"Failed to get jobs: {e}", exc_info=True)
        return None

    job_statuses = _get_job_statuses_by_name({"jobs": jobs})
    unit_tests_all_packages = job_statuses.get("unit tests - all packages")
    tests = job_statuses.get("tests")
    if not unit_tests_all_packages or not tests:
        logger.error(f"No unit tests all packages or tests job statuses found: {job_statuses}", exc_info=True)
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
