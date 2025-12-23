#!/usr/bin/env -S uv run
# ruff: noqa: E402

from metta.common.util.log_config import suppress_noisy_logs

suppress_noisy_logs()

import logging
import os
import sys
import time

from metta.app_backend.clients.stats_client import StatsClient
from metta.app_backend.tournament.runner import TournamentRunner
from metta.common.util.constants import PROD_STATS_SERVER_URI

logger = logging.getLogger(__name__)

DEFAULT_POLL_INTERVAL_SECONDS = 60.0


class TournamentScheduler:
    def __init__(
        self,
        runner: TournamentRunner,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    ):
        self._runner = runner
        self._poll_interval_seconds = poll_interval_seconds

    def run_cycle(self) -> None:
        logger.info("Running tournament scheduler cycle")
        result = self._runner.run_all_referees()
        logger.info(
            "Cycle complete: %d pools processed, %d matches created",
            result.pools_processed,
            result.total_matches_created,
        )
        if result.errors:
            for error in result.errors:
                logger.error("Error: %s", error)

    def run(self) -> None:
        logger.info("Tournament scheduler poll interval: %ss", self._poll_interval_seconds)
        while True:
            start_time = time.monotonic()
            try:
                self.run_cycle()
            except Exception as exc:
                logger.error("Error in tournament scheduler loop: %s", exc, exc_info=True)
            elapsed = time.monotonic() - start_time
            sleep_time = max(0.0, self._poll_interval_seconds - elapsed)
            time.sleep(sleep_time)


def init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def create_stats_client(backend_url: str, machine_token: str | None) -> StatsClient:
    stats_client = StatsClient(backend_url=backend_url, machine_token=machine_token)
    stats_client._validate_authenticated()
    return stats_client


def main() -> None:
    init_logging()

    backend_url = os.environ.get("BACKEND_URL", PROD_STATS_SERVER_URI)
    machine_token = os.environ.get("MACHINE_TOKEN")
    poll_interval = float(os.environ.get("POLL_INTERVAL", DEFAULT_POLL_INTERVAL_SECONDS))

    logger.info("Backend URL: %s", backend_url)

    stats_client = create_stats_client(backend_url, machine_token) if machine_token else StatsClient.create(backend_url)
    runner = TournamentRunner(stats_client)

    try:
        scheduler = TournamentScheduler(
            runner=runner,
            poll_interval_seconds=poll_interval,
        )
        scheduler.run()
    finally:
        stats_client.close()


if __name__ == "__main__":
    main()
