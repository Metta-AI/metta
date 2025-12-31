#!/usr/bin/env -S uv run
# need this to import and call suppress_noisy_logs first
# ruff: noqa: E402

from metta.common.util.log_config import suppress_noisy_logs

suppress_noisy_logs()

import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from gitta import get_current_commit
from metta.app_backend.clients.stats_client import (
    PROD_STATS_SERVER_URI,
    StatsClient,
)
from metta.app_backend.leaderboard_constants import (
    COGAMES_SUBMITTED_PV_KEY,
    LEADERBOARD_ATTEMPTS_PV_KEY,
    LEADERBOARD_EVAL_CANCELED_VALUE,
    LEADERBOARD_EVAL_DONE_PV_KEY,
    LEADERBOARD_EVAL_DONE_VALUE,
    LEADERBOARD_JOB_ID_PV_KEY,
)
from metta.app_backend.metta_repo import TaskStatus
from metta.app_backend.routes.eval_task_routes import TaskCreateRequest
from metta.common.datadog.tracing import init_tracing, trace
from metta.common.util.fs import get_repo_root

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL_SECONDS = 60.0
DEFAULT_MAX_LEADERBOARD_ATTEMPTS = 5


@dataclass(frozen=True)
class PolicyAttemptInfo:
    policy_version_id: uuid.UUID
    attempts: int


@dataclass(frozen=True)
class PolicyRemoteJobStatus:
    policy_version_id: uuid.UUID
    job_id: int
    status: TaskStatus | None
    attempts: int


class LeaderboardEvalScheduler:
    def __init__(
        self,
        stats_client: StatsClient,
        repo_root: str,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        eval_git_hash: Optional[str] = None,
        max_attempts: int = DEFAULT_MAX_LEADERBOARD_ATTEMPTS,
    ):
        self._stats_client = stats_client
        self._repo_root = repo_root
        self._poll_interval_seconds = poll_interval_seconds
        self._eval_git_hash = eval_git_hash
        self._max_attempts = max_attempts

    def _fetch_unscheduled_policy_versions(self) -> list[PolicyAttemptInfo]:
        """Get submitted policy versions that still need evals or whose prior eval failed."""
        rows = self._stats_client.sql_query(
            query=f"""
SELECT DISTINCT
    pv.id,
    COALESCE(attempt_tag.value::INT, 0) AS attempts
FROM policy_versions pv
JOIN policy_version_tags pvt ON pv.id = pvt.policy_version_id
LEFT JOIN policy_version_tags done_tag
    ON pv.id = done_tag.policy_version_id
    AND done_tag.key = '{LEADERBOARD_EVAL_DONE_PV_KEY}'
LEFT JOIN policy_version_tags attempt_tag
    ON pv.id = attempt_tag.policy_version_id
    AND attempt_tag.key = '{LEADERBOARD_ATTEMPTS_PV_KEY}'
WHERE pvt.key = '{COGAMES_SUBMITTED_PV_KEY}'
AND pvt.value = 'true'
AND (
    done_tag.value IS NULL
    OR done_tag.value NOT IN ('{LEADERBOARD_EVAL_DONE_VALUE}', '{LEADERBOARD_EVAL_CANCELED_VALUE}')
)
AND NOT EXISTS (
    SELECT 1
    FROM policy_version_tags pvt2
    WHERE pvt2.policy_version_id = pv.id
    AND pvt2.key = '{LEADERBOARD_JOB_ID_PV_KEY}'
)"""
        ).rows
        return [
            PolicyAttemptInfo(policy_version_id=row[0], attempts=int(row[1]) if row[1] is not None else 0)
            for row in rows
        ]

    def _mark_policy_version_canceled(self, policy_version_id: uuid.UUID, attempts: int) -> None:
        final_attempts = max(attempts, self._max_attempts)
        logger.info(
            "Marking policy version %s as canceled after %d attempts",
            policy_version_id,
            final_attempts,
        )
        self._stats_client.update_policy_version_tags(
            policy_version_id,
            {
                LEADERBOARD_EVAL_DONE_PV_KEY: LEADERBOARD_EVAL_CANCELED_VALUE,
                LEADERBOARD_ATTEMPTS_PV_KEY: str(final_attempts),
            },
        )

    @trace("eval_scheduler.schedule_eval")
    def _schedule_eval(self, policy_version_id: uuid.UUID, attempts: int) -> Optional[int]:
        if attempts >= self._max_attempts:
            self._mark_policy_version_canceled(policy_version_id, attempts)
            return None

        logger.info("Scheduling eval for policy: %s", policy_version_id)
        command_parts = [
            "uv run tools/run.py recipes.experiment.v0_leaderboard.evaluate",
            f"policy_version_id={str(policy_version_id)}",
        ]
        eval_task = self._stats_client.create_eval_task(
            TaskCreateRequest(
                command=" ".join(command_parts),
                git_hash=self._eval_git_hash,
                attributes={"parallelism": 10},
            )
        )
        logger.info("Successfully scheduled eval for policy: %s: %s", policy_version_id, eval_task.id)
        eval_task_id = eval_task.id
        total_attempts = attempts + 1
        self._stats_client.update_policy_version_tags(
            policy_version_id,
            {
                LEADERBOARD_JOB_ID_PV_KEY: str(eval_task_id),
                LEADERBOARD_ATTEMPTS_PV_KEY: str(total_attempts),
            },
        )
        logger.info(
            "Successfully marked policy version %s as scheduled (attempt %d/%d)",
            policy_version_id,
            total_attempts,
            self._max_attempts,
        )
        return eval_task.id

    def _fetch_scheduled_but_incomplete_jobs(self) -> list[PolicyRemoteJobStatus]:
        rows = self._stats_client.sql_query(
            query=f"""
SELECT pv.id,
    job_tag.value::BIGINT AS job_id,
    task.status,
    COALESCE(attempt_tag.value::INT, 0) AS attempts
FROM policy_versions pv
JOIN policy_version_tags submit_tag
    ON pv.id = submit_tag.policy_version_id
    AND submit_tag.key = '{COGAMES_SUBMITTED_PV_KEY}'
    AND submit_tag.value = 'true'
JOIN policy_version_tags job_tag
    ON pv.id = job_tag.policy_version_id
    AND job_tag.key = '{LEADERBOARD_JOB_ID_PV_KEY}'
LEFT JOIN eval_tasks_view task
    ON task.id = job_tag.value::BIGINT
LEFT JOIN policy_version_tags done_tag
    ON pv.id = done_tag.policy_version_id
    AND done_tag.key = '{LEADERBOARD_EVAL_DONE_PV_KEY}'
LEFT JOIN policy_version_tags attempt_tag
    ON pv.id = attempt_tag.policy_version_id
    AND attempt_tag.key = '{LEADERBOARD_ATTEMPTS_PV_KEY}'
WHERE (
    done_tag.value IS NULL
    OR done_tag.value NOT IN ('{LEADERBOARD_EVAL_DONE_VALUE}', '{LEADERBOARD_EVAL_CANCELED_VALUE}')
)
"""
        ).rows
        remote_jobs: list[PolicyRemoteJobStatus] = []
        for policy_version_id, job_id, status, attempts in rows:
            if job_id is None:
                continue
            remote_jobs.append(
                PolicyRemoteJobStatus(
                    policy_version_id=policy_version_id,
                    job_id=int(job_id),
                    status=status,
                    attempts=int(attempts) if attempts is not None else 0,
                )
            )
        return remote_jobs

    def _mark_successful_remote_jobs(self, remote_jobs: list[PolicyRemoteJobStatus]) -> None:
        completed_jobs = [j for j in remote_jobs if j.status == "done"]
        logger.info("Marking %d successful remote jobs as complete", len(completed_jobs))
        for job in completed_jobs:
            logger.info("Marking policy version %s as leaderboard eval complete", job.policy_version_id)
            self._stats_client.update_policy_version_tags(
                job.policy_version_id, {LEADERBOARD_EVAL_DONE_PV_KEY: LEADERBOARD_EVAL_DONE_VALUE}
            )

    @trace("eval_scheduler.run_cycle")
    def run_cycle(self) -> None:
        remote_jobs = self._fetch_scheduled_but_incomplete_jobs()
        logger.info("Found %d incomplete but scheduled", len(remote_jobs))
        self._mark_successful_remote_jobs(remote_jobs)
        unscheduled = self._fetch_unscheduled_policy_versions()
        logger.info("Found %d unscheduled policy versions", len(unscheduled))
        needs_rescheduling = [j for j in remote_jobs if j.status in ("error", "canceled", "system_error", None)]
        logger.info("Found %d remote jobs that need to be rescheduled", len(needs_rescheduling))

        for job in needs_rescheduling:
            try:
                self._schedule_eval(job.policy_version_id, job.attempts)
            except Exception:
                logger.error("Failed to reschedule eval for policy: %s", job.policy_version_id, exc_info=True)
                continue

        for submission in unscheduled:
            try:
                self._schedule_eval(submission.policy_version_id, submission.attempts)
            except Exception:
                logger.error("Failed to schedule eval for policy: %s", submission.policy_version_id, exc_info=True)
                continue

    def run(self) -> None:
        logger.info("eval scheduler poll interval: %ss", self._poll_interval_seconds)
        while True:
            start_time = time.monotonic()
            try:
                self.run_cycle()
            except Exception as exc:
                logger.error("Error in eval scheduler loop: %s", exc, exc_info=True)
            elapsed = time.monotonic() - start_time
            sleep_time = max(0.0, self._poll_interval_seconds - elapsed)
            time.sleep(sleep_time)


def init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def create_stats_client(backend_url: str, machine_token: Optional[str]) -> StatsClient:
    stats_client = StatsClient(backend_url=backend_url, machine_token=machine_token)
    stats_client._validate_authenticated()
    return stats_client


def main() -> None:
    init_logging()
    init_tracing()

    backend_url = os.environ.get("BACKEND_URL", PROD_STATS_SERVER_URI)
    machine_token = os.environ.get("MACHINE_TOKEN")
    eval_git_hash = os.environ.get("EVAL_GIT_HASH", get_current_commit())
    poll_interval = float(os.environ.get("POLL_INTERVAL", DEFAULT_POLL_INTERVAL_SECONDS))
    repo_root = get_repo_root()

    logger.info("Backend URL: %s", backend_url)
    logger.info("Repo root: %s", repo_root)
    logger.info("Eval git hash: %s", eval_git_hash)

    stats_client = create_stats_client(backend_url, machine_token) if machine_token else StatsClient.create(backend_url)
    try:
        scheduler = LeaderboardEvalScheduler(
            stats_client=stats_client,
            repo_root=str(repo_root),
            poll_interval_seconds=poll_interval,
            eval_git_hash=None if eval_git_hash == "main" else eval_git_hash,
        )
        scheduler.run()
    finally:
        stats_client.close()


if __name__ == "__main__":
    main()
