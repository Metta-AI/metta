#!/usr/bin/env -S uv run
# need this to import and call suppress_noisy_logs first
# ruff: noqa: E402

from metta.common.util.log_config import suppress_noisy_logs

suppress_noisy_logs()

import asyncio
import logging
import sys

import fastapi
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from pydantic.main import BaseModel

from metta.app_backend.auth import user_from_header_or_token
from metta.app_backend.metta_repo import MettaRepo
from metta.app_backend.routes import (
    eval_task_routes,
    job_routes,
    leaderboard_routes,
    sql_routes,
    stats_routes,
    sweep_routes,
)


class WhoAmIResponse(BaseModel):
    user_email: str


_logging_configured = False


class NoWhoAmIFilter(logging.Filter):
    """Filter out /whoami requests from uvicorn access logs."""

    def filter(self, record):
        # Filter out /whoami requests from uvicorn access logs
        if hasattr(record, "getMessage"):
            message = record.getMessage()
            return not ("/whoami" in message and "GET" in message)
        return True


def setup_logging():
    """Configure logging for the application, including scorecard performance logging."""
    global _logging_configured

    if _logging_configured:
        return

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Configure scorecard performance logger specifically
    scorecard_logger = logging.getLogger("dashboard_performance")
    scorecard_logger.setLevel(logging.INFO)

    # Configure database query performance logger
    db_logger = logging.getLogger("db_performance")
    db_logger.setLevel(logging.INFO)

    # Configure route performance logger
    route_logger = logging.getLogger("route_performance")
    route_logger.setLevel(logging.INFO)

    # Configure scorecard logger
    scorecard_routes_logger = logging.getLogger("policy_scorecard_routes")
    scorecard_routes_logger.setLevel(logging.INFO)

    # Configure metta repo logger
    metta_repo_logger = logging.getLogger("metta_repo")
    metta_repo_logger.setLevel(logging.INFO)

    # Configure psycopg pool logger
    psycopg_pool_logger = logging.getLogger("psycopg.pool")
    psycopg_pool_logger.setLevel(logging.WARNING)

    # Ensure the loggers don't duplicate messages from root logger
    scorecard_logger.propagate = True
    db_logger.propagate = True
    route_logger.propagate = True
    scorecard_routes_logger.propagate = True

    # Filter out /whoami requests from uvicorn access logs
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.addFilter(NoWhoAmIFilter())

    _logging_configured = True
    print(
        "Logging configured - performance logging enabled (routes, db queries, scorecards), /whoami requests filtered"
    )


def create_app(stats_repo: MettaRepo) -> fastapi.FastAPI:
    """Create a FastAPI app with the given StatsRepo instance."""
    # Ensure logging is configured
    setup_logging()

    app = fastapi.FastAPI()

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://localhost:3000",
            "https://observatory.softmax-research.net",
        ],  # Frontend URLs
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Create routers with the provided StatsRepo
    eval_task_router = eval_task_routes.create_eval_task_router(stats_repo)
    sql_router = sql_routes.create_sql_router(stats_repo)
    stats_router = stats_routes.create_stats_router(stats_repo)
    sweep_router = sweep_routes.create_sweep_router(stats_repo)
    leaderboard_router = leaderboard_routes.create_leaderboard_router(stats_repo)
    jobs_router = job_routes.create_job_router()

    app.include_router(eval_task_router)
    app.include_router(sql_router)
    app.include_router(stats_router)
    app.include_router(sweep_router)
    app.include_router(leaderboard_router)
    app.include_router(jobs_router)

    @app.get("/whoami")
    async def whoami(request: fastapi.Request) -> WhoAmIResponse:
        user_id = await user_from_header_or_token(request)
        return WhoAmIResponse(user_email=user_id or "unknown")

    return app


if __name__ == "__main__":
    from metta.app_backend.config import settings

    stats_repo = MettaRepo(settings.STATS_DB_URI)
    app = create_app(stats_repo)

    # Start the updater in an async context
    async def main():
        # Run uvicorn in a way that doesn't block
        config = uvicorn.Config(app, host=settings.HOST, port=settings.PORT)
        server = uvicorn.Server(config)
        await server.serve()

    asyncio.run(main())
