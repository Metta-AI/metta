#!/usr/bin/env -S uv run
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
from metta.app_backend.metta_repo import TaskStatus
from metta.app_backend.routes.eval_task_routes import TaskCreateRequest
from metta.common.datadog.tracing import init_tracing, trace
from metta.common.util.fs import get_repo_root
from metta.common.util.log_config import init_suppress_warnings
from recipes.experiment.v0_leaderboard_eval import V0_LEADERBOARD_NAME_TAG_KEY

logger = logging.getLogger(__name__)

SUBMITTED_KEY = "cogames-submitted"
REMOTE_JOB_ID_KEY = "v0-leaderboard-eval-remote-job-id"
EVALS_DONE_KEY = "v0-leaderboard-evals-done"
DEFAULT_POLL_INTERVAL_SECONDS = 60.0


@dataclass(frozen=True)
class PolicyRemoteJobStatus:
    policy_version_id: uuid.UUID
    job_id: int
    status: TaskStatus | None


class LeaderboardEvalScheduler:
    def __init__(
        self,
        stats_client: StatsClient,
        repo_root: str,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        eval_git_hash: Optional[str] = None,
    ):
        self._stats_client = stats_client
        self._repo_root = repo_root
        self._poll_interval_seconds = poll_interval_seconds
        self._eval_git_hash = eval_git_hash

    def _get_per_sim_scores(self, policy_version_id: uuid.UUID) -> dict[str, float]:
        rows = self._stats_client.sql_query(
            query=f"""
SELECT
    et1.value,
    AVG(epm.value / ep.num_agents) as avg_reward_per_agent
FROM episode_policies ep
JOIN episodes e ON e.id = ep.episode_id
JOIN episode_policy_metrics epm ON epm.episode_internal_id = e.internal_id
    AND epm.pv_internal_id = (SELECT internal_id FROM policy_versions WHERE id = '{policy_version_id}')
JOIN episode_tags et1 ON et1.episode_id = e.id
WHERE ep.policy_version_id = '{policy_version_id}'
    AND epm.metric_name = 'reward'
    AND et1.key = '{V0_LEADERBOARD_NAME_TAG_KEY}'
GROUP BY et1.value
ORDER BY et1.value
"""
        ).rows
        return {leaderboard_name: score for leaderboard_name, score in rows}

    def _fetch_unscheduled_policy_versions(self) -> list[uuid.UUID]:
        """Get submitted policy versions that still need evals or whose prior eval failed."""
        rows = self._stats_client.sql_query(
            query=f"""
SELECT DISTINCT pv.id
FROM policy_versions pv
JOIN policy_version_tags pvt ON pv.id = pvt.policy_version_id
WHERE pvt.key = '{SUBMITTED_KEY}'
AND pvt.value = 'true'
AND NOT EXISTS (
    SELECT 1
    FROM policy_version_tags pvt2
    WHERE pvt2.policy_version_id = pv.id
    AND pvt2.key = '{REMOTE_JOB_ID_KEY}'
)"""
        ).rows
        unprocessed: list[uuid.UUID] = [row[0] for row in rows]
        return unprocessed

    @trace("eval_scheduler.schedule_eval")
    def _schedule_eval(self, policy_version_id: uuid.UUID) -> int:
        logger.info("Scheduling eval for policy: %s", policy_version_id)
        command_parts = [
            "uv run tools/run.py recipes.experiment.v0_leaderboard_eval.run",
            f"policy_version_id={str(policy_version_id)}",
            f"stats_server_uri={self._stats_client._backend_url}",
        ]
        eval_task = self._stats_client.create_eval_task(
            TaskCreateRequest(
                command=" ".join(command_parts),
                git_hash=self._eval_git_hash,
            )
        )
        logger.info("Successfully scheduled eval for policy: %s: %s", policy_version_id, eval_task.id)
        eval_task_id = eval_task.id
        self._stats_client.update_policy_version_tags(policy_version_id, {REMOTE_JOB_ID_KEY: str(eval_task_id)})
        logger.info("Successfully marked policy version %s as scheduled", policy_version_id)
        return eval_task.id

    def _fetch_scheduled_but_incomplete_jobs(self) -> list[PolicyRemoteJobStatus]:
        rows = self._stats_client.sql_query(
            query=f"""
SELECT pv.id,
    job_tag.value::BIGINT AS job_id,
    task.status
FROM policy_versions pv
JOIN policy_version_tags submit_tag
    ON pv.id = submit_tag.policy_version_id
    AND submit_tag.key = '{SUBMITTED_KEY}'
    AND submit_tag.value = 'true'
JOIN policy_version_tags job_tag
    ON pv.id = job_tag.policy_version_id
    AND job_tag.key = '{REMOTE_JOB_ID_KEY}'
LEFT JOIN eval_tasks_view task
    ON task.id = job_tag.value::BIGINT
LEFT JOIN policy_version_tags done_tag
    ON pv.id = done_tag.policy_version_id
    AND done_tag.key = '{EVALS_DONE_KEY}'
WHERE (done_tag.value IS NULL OR done_tag.value != 'true')
"""
        ).rows
        remote_jobs: list[PolicyRemoteJobStatus] = []
        for policy_version_id, job_id, status in rows:
            if job_id is None:
                continue
            remote_jobs.append(
                PolicyRemoteJobStatus(
                    policy_version_id=policy_version_id,
                    job_id=int(job_id),
                    status=status,
                )
            )
        return remote_jobs

    def _mark_successful_remote_jobs(self, remote_jobs: list[PolicyRemoteJobStatus]) -> None:
        completed_jobs = [j for j in remote_jobs if j.status == "done"]
        logger.info("Marking %d successful remote jobs as complete", len(completed_jobs))
        for job in completed_jobs:
            logger.info("Marking policy version %s as leaderboard eval complete", job.policy_version_id)
            self._stats_client.update_policy_version_tags(job.policy_version_id, {EVALS_DONE_KEY: "true"})

    @trace("eval_scheduler.run_cycle")
    def run_cycle(self) -> None:
        remote_jobs = self._fetch_scheduled_but_incomplete_jobs()
        logger.info("Found %d incomplete but scheduled", len(remote_jobs))
        self._mark_successful_remote_jobs(remote_jobs)
        unscheduled = self._fetch_unscheduled_policy_versions()
        logger.info("Found %d unscheduled policy versions", len(unscheduled))
        needs_rescheduling = [j for j in remote_jobs if j.status in ("error", "canceled", "system_error", None)]
        logger.info("Found %d remote jobs that need to be rescheduled", len(needs_rescheduling))

        for policy_version_id in [r.policy_version_id for r in needs_rescheduling] + unscheduled:
            try:
                self._schedule_eval(policy_version_id)
            except Exception:
                logger.error("Failed to schedule eval for policy: %s", policy_version_id, exc_info=True)
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
    logging.getLogger("httpx").setLevel(logging.WARNING)


def create_stats_client(backend_url: str, machine_token: Optional[str]) -> StatsClient:
    stats_client = StatsClient(backend_url=backend_url, machine_token=machine_token)
    stats_client._validate_authenticated()
    return stats_client


def main() -> None:
    init_logging()
    init_suppress_warnings()
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
            eval_git_hash=eval_git_hash,
        )
        scheduler.run()
    finally:
        stats_client.close()


if __name__ == "__main__":
    main()
