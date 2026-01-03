import asyncio
import os

import typer

app = typer.Typer(help="Tournament commissioner CLI")

LOCAL_DB_URI = "postgres://postgres:password@127.0.0.1:5432/metta"


@app.command()
def run(local: bool = typer.Option(False, "--local", "-l", help="Use local dev database")) -> None:
    """Run all commissioners concurrently."""
    if local:
        os.environ["STATS_DB_URI"] = LOCAL_DB_URI

    from metta.app_backend.tournament.registry import SEASONS
    from metta.common.util.log_config import init_logging, suppress_noisy_logs

    init_logging()
    suppress_noisy_logs()

    async def run_all() -> None:
        commissioners = [cls() for cls in SEASONS.values()]
        await asyncio.gather(*[c.run() for c in commissioners])

    asyncio.run(run_all())


if __name__ == "__main__":
    app()
