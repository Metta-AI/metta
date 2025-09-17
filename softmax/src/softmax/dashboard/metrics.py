import logging
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Generator

import httpx
from pydantic import BaseModel

from metta.common.util.constants import METTA_GITHUB_ORGANIZATION, METTA_GITHUB_REPO
from softmax.aws.secrets_manager import get_secretsmanager_secret
from softmax.dashboard.registry import metric_goal

logger = logging.getLogger(__name__)


@contextmanager
def _github_client() -> Generator[httpx.Client, None, None]:
    with httpx.Client(
        base_url=f"https://api.github.com/repos/{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}",
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "softmax-metrics",
            # Auth to avoid rate limiting
            "Authorization": f"Basic {get_secretsmanager_secret('github/dashboard-token')}",
        },
        timeout=30,
    ) as client:
        yield client


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
    per_page = 100
    params = {"sha": branch, "since": since, "per_page": per_page}

    count = 0
    page = 1

    with _github_client() as client:
        while True:
            resp = client.get(
                "/commits",
                params={**params, "page": page},
            )
            if resp.status_code >= 400:
                logger.error(f"Failed to get commits: {resp.status_code} {resp.text}")
                break

            commits: list[dict[str, Any]] = resp.json() or []
            for commit in commits:
                message = commit.get("commit", {}).get("message") or ""
                if phrase.lower() in message.lower():
                    count += 1
            if len(commits) < per_page:
                break
            page += 1

    return count


@metric_goal(
    metric_key="commits.reverts",
    aggregate="sum",
    target=1.0,
    comparison="<",
    window="7d",
    description="Keep the rolling 7-day sum of reverts below one per week.",
)
def get_num_revert_commits(lookback_days: int = 7, branch: str = "main") -> int:
    return _get_num_commits_with_phrase("revert", lookback_days=lookback_days, branch=branch)


def get_latest_workflow_run(branch: str, workflow_filename: str) -> dict[str, Any] | None:
    params = {"branch": branch, "status": "completed", "per_page": 1}
    with _github_client() as client:
        resp = client.get(
            f"/actions/workflows/{workflow_filename}/runs",
            params=params,
        )
        if resp.status_code >= 400:
            logger.error(f"Failed to get workflow runs: {resp.status_code} {resp.text}")
            return None
        runs = (resp.json() or {}).get("workflow_runs", [])
        return runs[0] if runs else None


@metric_goal(
    metric_key="ci.tests_passing_on_main",
    aggregate="min",
    target=1.0,
    comparison=">=",
    window="1h",
    description="Unit-test jobs should be passing on main",
)
def get_latest_unit_tests_failed() -> int:
    run = get_latest_workflow_run(branch="main", workflow_filename="checks.yml")
    if not run:
        return 1

    run_id = run.get("id")
    if not run_id:
        logger.error(f"Failed to get run ID: {run}")
        return 1

    params = {"per_page": 100}
    with _github_client() as client:
        resp = client.get(
            f"/actions/runs/{run_id}/jobs",
            params=params,
        )
        if resp.status_code >= 400:
            logger.error(f"Failed to get jobs: {resp.status_code} {resp.text}")
            return 1

        job_statuses = _get_job_statuses_by_name(resp.json() or {})
        unit_tests_all_packages = job_statuses.get("unit tests - all packages")
        tests = job_statuses.get("tests")
        if not unit_tests_all_packages or not tests:
            logger.error(f"No unit tests all packages or tests job statuses found: {job_statuses}")
            return 1

        # Cancelled tests can be identified by "Unit Tests - All Packages" job being cancelled and then "Tests" failing
        canceled = unit_tests_all_packages.status == "cancelled" and tests.conclusion == "failure"
        if canceled:
            return 1
        return int(
            all(
                job_status.conclusion in ("success", "skipped")
                for job_status in job_statuses.values()
                if job_status.status == "completed"
            )
        )
