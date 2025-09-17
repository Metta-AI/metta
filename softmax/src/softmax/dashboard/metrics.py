from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Generator

import httpx

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
            "Authorization": f"Basic {get_secretsmanager_secret('github/dashboard-token')}",
        },
        timeout=30,
    ) as client:
        yield client


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
                message = (commit.get("commit", {}).get("message") or "").strip().lower()
                if phrase in message:
                    count += 1
            if len(commits) < per_page:
                break
            page += 1

    return count


@metric_goal(
    metric_key="commits.count.revert",
    aggregate="sum",
    target=1.0,
    comparison="<",
    window="7d",
    description="Keep the rolling 7-day sum of reverts below one per week.",
)
def get_num_revert_commits(lookback_days: int = 7, branch: str = "main") -> int:
    return _get_num_commits_with_phrase("revert", lookback_days=lookback_days, branch=branch)


def get_latest_workflow_run(branch: str = "main", workflow_filename: str = "checks.yml") -> dict[str, Any] | None:
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
    metric_key="ci.workflow_failing_on_main",
    aggregate="max",
    target=0.0,
    comparison="<=",
    window="1h",
    description="Latest or max over the last hour should be zero failing workflow runs on main.",
)
def get_latest_workflow_run_failed(branch: str = "main", workflow_filename: str = "checks.yml") -> int:
    run = get_latest_workflow_run(branch=branch, workflow_filename=workflow_filename)
    if not run:
        return 0
    conclusion = (run.get("conclusion") or "").lower()
    return 0 if conclusion == "success" else 1


@metric_goal(
    metric_key="ci.tests_failing_on_main",
    aggregate="max",
    target=0.0,
    comparison="<=",
    window="1h",
    description="No unit-test jobs should fail on main",
)
def get_latest_unit_tests_failed(branch: str = "main", workflow_filename: str = "checks.yml") -> int:
    run = get_latest_workflow_run(branch=branch, workflow_filename=workflow_filename)
    if not run:
        return 0

    run_id = run.get("id") or run.get("database_id") or run.get("run_number")
    if not run_id:
        logger.error(f"Failed to get run ID: {run}")
        return 0

    params = {"per_page": 100}
    with _github_client() as client:
        resp = client.get(
            f"/actions/runs/{run_id}/jobs",
            params=params,
        )
        if resp.status_code >= 400:
            logger.error(f"Failed to get jobs: {resp.status_code} {resp.text}")
            return 0

        jobs = (resp.json() or {}).get("jobs", [])
        for job in jobs:
            name = (job.get("name") or "").lower()
            if "unit tests" in name or "unit-tests" in name or name.startswith("unit"):
                conclusion = (job.get("conclusion") or "").lower()
                if conclusion != "success":
                    return 1

    return 0
