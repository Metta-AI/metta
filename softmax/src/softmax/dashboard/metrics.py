from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Generator

import httpx

from metta.common.util.constants import (
    METTA_GITHUB_ORGANIZATION,
    METTA_GITHUB_REPO,
)
from softmax.dashboard.registry import (
    metric_goal,
)


@contextmanager
def _github_client() -> Generator[httpx.Client, None, None]:
    with httpx.Client(
        base_url=f"https://api.github.com/repos/{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO}",
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "softmax-metrics",
        },
        timeout=30,
    ) as client:
        yield client


def _parse_iso8601(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


@metric_goal(
    metric_key="reverts.count",
    aggregate="sum",
    target=1.0,
    comparison="<",
    window="7d",
    description="Keep the rolling 7-day sum of reverts below one per week.",
)
def get_num_reverts(lookback_days: int = 7, branch: str = "main") -> int:
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
                break

            commits: list[dict[str, Any]] = resp.json() or []
            for commit in commits:
                message = (commit.get("commit", {}).get("message") or "").strip().lower()
                if message.startswith("revert ") or "this reverts commit" in message:
                    count += 1
            if len(commits) < per_page:
                break
            page += 1

    return count


def get_latest_workflow_run(branch: str = "main", workflow_filename: str = "checks.yml") -> dict[str, Any] | None:
    params = {"branch": branch, "status": "completed", "per_page": 1}
    with _github_client() as client:
        resp = client.get(
            f"/actions/workflows/{workflow_filename}/runs",
            params=params,
        )
        if resp.status_code >= 400:
            return None
        runs = (resp.json() or {}).get("workflow_runs", [])
        return runs[0] if runs else None


@metric_goal(
    metric_key="ci.test_run",
    aggregate="p95",
    target=300.0,
    comparison="<=",
    window="7d",
    description="P95 CI runtime across the last 7 days should stay under five minutes.",
)
def get_latest_workflow_run_duration_seconds(
    branch: str = "main", workflow_filename: str = "checks.yml"
) -> float | None:
    run = get_latest_workflow_run(branch=branch, workflow_filename=workflow_filename)
    if not run:
        return None
    started_at = run.get("run_started_at") or run.get("created_at")
    updated_at = run.get("updated_at")
    if not (started_at and updated_at):
        return None
    return (_parse_iso8601(updated_at) - _parse_iso8601(started_at)).total_seconds()


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
        return 0

    params = {"per_page": 100}
    with _github_client() as client:
        resp = client.get(
            f"/actions/runs/{run_id}/jobs",
            params=params,
        )
        if resp.status_code >= 400:
            return 0

        jobs = (resp.json() or {}).get("jobs", [])
        for job in jobs:
            name = (job.get("name") or "").lower()
            if "unit tests" in name or "unit-tests" in name or name.startswith("unit"):
                conclusion = (job.get("conclusion") or "").lower()
                if conclusion != "success":
                    return 1

    return 0


@metric_goal(
    metric_key="pr.assignment_to_merge.p90_seconds",
    aggregate="p90",
    target=172800.0,
    comparison="<=",
    window="30d",
    description="P90 time from assignment to merge should remain under two days on a 30-day window.",
)
def compute_p90_assignment_to_merge_seconds(lookback_days: int = 30, max_prs: int = 50) -> float | None:
    since_date = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date().isoformat()
    search_q = f"repo:{METTA_GITHUB_ORGANIZATION}/{METTA_GITHUB_REPO} is:pr is:merged merged:>={since_date}"

    durations: list[float] = []

    with _github_client() as client:
        page = 1
        per_page = 50
        while len(durations) < max_prs:
            resp = client.get(
                "/search/issues",
                params={
                    "q": search_q,
                    "sort": "updated",
                    "order": "desc",
                    "per_page": per_page,
                    "page": page,
                },
            )
            if resp.status_code >= 400:
                break
            items = (resp.json() or {}).get("items", [])
            if not items:
                break

            for item in items:
                if len(durations) >= max_prs:
                    break
                number = item.get("number")
                if not number:
                    continue

                pr_resp = client.get(f"/pulls/{number}")
                if pr_resp.status_code >= 400:
                    continue
                pr = pr_resp.json()
                author = (pr.get("user") or {}).get("login")
                merged_at = pr.get("merged_at")
                if not (author and merged_at):
                    continue

                merged_dt = _parse_iso8601(merged_at)

                ev_resp = client.get(
                    f"/issues/{number}/events",
                    params={"per_page": 100},
                )
                if ev_resp.status_code >= 400:
                    continue
                events = ev_resp.json() or []

                first_assignment_time: datetime | None = None
                for event in events:
                    etype = event.get("event")
                    created_at = event.get("created_at")
                    if not created_at:
                        continue
                    event_time = _parse_iso8601(created_at)

                    if etype == "assigned":
                        assignee_login = (event.get("assignee") or {}).get("login")
                        if assignee_login and assignee_login != author:
                            if first_assignment_time is None or event_time < first_assignment_time:
                                first_assignment_time = event_time
                    elif etype == "review_requested":
                        reviewer_login = (event.get("requested_reviewer") or {}).get("login")
                        if reviewer_login and reviewer_login != author:
                            if first_assignment_time is None or event_time < first_assignment_time:
                                first_assignment_time = event_time

                if first_assignment_time is None:
                    continue

                if merged_dt > first_assignment_time:
                    durations.append((merged_dt - first_assignment_time).total_seconds())

            if len(items) < per_page:
                break
            page += 1

    if not durations:
        return None

    durations.sort()
    idx = max(0, int(round(0.9 * (len(durations) - 1))))
    return float(durations[idx])
