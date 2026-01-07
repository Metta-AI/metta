import asyncio

from metta.app_backend.health_server import start_health_server
from metta.app_backend.tournament.registry import SEASONS
from metta.common.otel.tracing import init_tracing
from metta.common.util.log_config import init_logging, suppress_noisy_logs


def run_commissioner():
    start_health_server(port=8081)

    async def run_all() -> None:
        commissioners = [cls() for cls in SEASONS.values()]
        await asyncio.gather(*[c.run() for c in commissioners])

    asyncio.run(run_all())


if __name__ == "__main__":
    init_logging()
    suppress_noisy_logs()
    init_tracing(service_name="tournament-commissioner")
    run_commissioner()
