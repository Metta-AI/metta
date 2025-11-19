#!/usr/bin/env -S uv run
import logging
import os
import sys
import time
import uuid
from typing import Optional

from metta.app_backend.clients.stats_client import (
    PROD_STATS_SERVER_URI,
    StatsClient,
)
from metta.app_backend.routes.eval_task_routes import TaskCreateRequest
from metta.common.datadog.tracing import init_tracing, trace
from metta.common.util.fs import get_repo_root
from metta.common.util.log_config import init_suppress_warnings

logger = logging.getLogger(__name__)

SUBMITTED_KEY = "cogames-submitted"
DEFAULT_POLL_INTERVAL_SECONDS = 60.0


class LeaderboardMatchScheduler:
    def __init__(
        self,
        stats_client: StatsClient,
        repo_root: str,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    ):
        self._stats_client = stats_client
        self._repo_root = repo_root
        self._poll_interval_seconds = poll_interval_seconds

    @trace("match_scheduler.fetch_unprocessed_policies")
    def _fetch_unprocessed_policies(self) -> list[uuid.UUID]:
        """Get all the policy versions that are submitted through cogames but not scheduled for a match"""
        # TODO: implement, use the stats_client.sql_query endpoint to get policy_version ids
        return []

    @trace("match_scheduler.schedule_match")
    def _schedule_match(self, policy_version_id: uuid.UUID) -> int:
        logger.info("Scheduling match for policy: %s", policy_version_id)
        # TODO: request a remote eval for the right thing, and
        eval_task = self._stats_client.create_eval_task(
            TaskCreateRequest(
                command="recipes.experiment.multi_policy_eval.run",
            )
        )
        logger.info("Successfully scheduled match for policy: %s: %s", policy_version_id, eval_task.id)
        eval_task_id = eval_task.id
        self._stats_client.update_policy_version_tags(
            policy_version_id, {"v0-leaderboard-match-remote-job-id": str(eval_task_id)}
        )
        logger.info("Successfully marked policy version %s as scheduled", policy_version_id)
        return eval_task.id

    @trace("match_scheduler.run_cycle")
    def run_cycle(self) -> None:
        for policy_version_id in self._fetch_unprocessed_policies():
            try:
                self._schedule_match(policy_version_id)
            except Exception:
                logger.error("Failed to schedule match for policy: %s", policy_version_id, exc_info=True)
                continue

    def run(self) -> None:
        logger.info("Match scheduler poll interval: %ss", self._poll_interval_seconds)
        while True:
            start_time = time.monotonic()
            try:
                self.run_cycle()
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Error in match scheduler loop: %s", exc, exc_info=True)
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
    poll_interval = float(os.environ.get("POLL_INTERVAL", DEFAULT_POLL_INTERVAL_SECONDS))
    repo_root = get_repo_root()

    logger.info("Backend URL: %s", backend_url)
    logger.info("Repo root: %s", repo_root)

    stats_client = create_stats_client(backend_url, machine_token)
    try:
        scheduler = LeaderboardMatchScheduler(
            stats_client=stats_client,
            repo_root=str(repo_root),
            poll_interval_seconds=poll_interval,
        )
        scheduler.run()
    finally:
        stats_client.close()


if __name__ == "__main__":
    main()
