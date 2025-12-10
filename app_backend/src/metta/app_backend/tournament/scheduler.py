#!/usr/bin/env -S uv run
# ruff: noqa: E402

from metta.common.util.log_config import suppress_noisy_logs

suppress_noisy_logs()

import asyncio
import logging
import os
import sys
import time

from metta.app_backend.config import settings
from metta.app_backend.metta_repo import MettaRepo
from metta.app_backend.tournament.runner import TournamentRunner

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

    async def run_cycle(self) -> None:
        logger.info("Running tournament scheduler cycle")
        result = await self._runner.run_all_referees()
        logger.info(
            "Cycle complete: %d pools processed, %d matches created",
            result.pools_processed,
            result.total_matches_created,
        )
        if result.errors:
            for error in result.errors:
                logger.error("Error: %s", error)

    async def run(self) -> None:
        logger.info("Tournament scheduler poll interval: %ss", self._poll_interval_seconds)
        while True:
            start_time = time.monotonic()
            try:
                await self.run_cycle()
            except Exception as exc:
                logger.error("Error in tournament scheduler loop: %s", exc, exc_info=True)
            elapsed = time.monotonic() - start_time
            sleep_time = max(0.0, self._poll_interval_seconds - elapsed)
            await asyncio.sleep(sleep_time)


def init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


async def async_main() -> None:
    db_uri = os.environ.get("DATABASE_URL", settings.DATABASE_URL)
    poll_interval = float(os.environ.get("POLL_INTERVAL", DEFAULT_POLL_INTERVAL_SECONDS))

    logger.info("Database URL: %s", db_uri[:20] + "..." if len(db_uri) > 20 else db_uri)

    repo = MettaRepo(db_uri)
    runner = TournamentRunner(repo)

    try:
        scheduler = TournamentScheduler(
            runner=runner,
            poll_interval_seconds=poll_interval,
        )
        await scheduler.run()
    finally:
        await repo.close()


def main() -> None:
    init_logging()
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
