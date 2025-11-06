#!/usr/bin/env -S uv run

import asyncio
import logging
import sys

import fastapi
import fastapi.middleware.cors
import pydantic.main
import uvicorn

import metta.app_backend.auth
import metta.app_backend.leaderboard_updater
import metta.app_backend.metta_repo
import metta.app_backend.routes


class WhoAmIResponse(pydantic.main.BaseModel):
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

    logging.getLogger("httpx").setLevel(logging.WARNING)

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

    # Configure leaderboard updater logger
    leaderboard_updater_logger = logging.getLogger("leaderboard_updater")
    leaderboard_updater_logger.setLevel(logging.INFO)

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


def create_app(stats_repo: metta.app_backend.metta_repo.MettaRepo) -> fastapi.FastAPI:
    """Create a FastAPI app with the given StatsRepo instance."""
    # Ensure logging is configured
    setup_logging()

    app = fastapi.FastAPI()

    # Add CORS middleware
    app.add_middleware(
        fastapi.middleware.cors.CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Frontend URLs
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Create routers with the provided StatsRepo
    cogames_router = metta.app_backend.routes.cogames_routes.create_cogames_router(stats_repo)
    dashboard_router = metta.app_backend.routes.dashboard_routes.create_dashboard_router(stats_repo)
    eval_task_router = metta.app_backend.routes.eval_task_routes.create_eval_task_router(stats_repo)
    leaderboard_router = metta.app_backend.routes.leaderboard_routes.create_leaderboard_router(stats_repo)
    sql_router = metta.app_backend.routes.sql_routes.create_sql_router(stats_repo)
    stats_router = metta.app_backend.routes.stats_routes.create_stats_router(stats_repo)
    token_router = metta.app_backend.routes.token_routes.create_token_router(stats_repo)
    policy_scorecard_router = metta.app_backend.routes.scorecard_routes.create_policy_scorecard_router(stats_repo)
    score_router = metta.app_backend.routes.score_routes.create_score_router(stats_repo)
    sweep_router = metta.app_backend.routes.sweep_routes.create_sweep_router(stats_repo)
    entity_router = metta.app_backend.routes.entity_routes.create_entity_router(stats_repo)

    app.include_router(cogames_router)
    app.include_router(dashboard_router)
    app.include_router(eval_task_router)
    app.include_router(leaderboard_router)
    app.include_router(sql_router)
    app.include_router(stats_router)
    app.include_router(token_router)
    app.include_router(policy_scorecard_router, prefix="/scorecard")
    app.include_router(score_router)
    # TODO: remove this once we're confident we've migrated all clients to use the /scorecard prefix
    app.include_router(policy_scorecard_router, prefix="/heatmap")
    app.include_router(sweep_router)
    app.include_router(entity_router)

    @app.get("/whoami")
    async def whoami(request: fastapi.Request) -> WhoAmIResponse:
        user_id = await metta.app_backend.auth.user_from_header_or_token(request, stats_repo)
        return WhoAmIResponse(user_email=user_id or "unknown")

    return app


if __name__ == "__main__":
    import metta.app_backend.config

    stats_repo = metta.app_backend.metta_repo.MettaRepo(metta.app_backend.config.stats_db_uri)
    app = create_app(stats_repo)
    leaderboard_updater = metta.app_backend.leaderboard_updater.LeaderboardUpdater(stats_repo)

    # Start the updater in an async context
    async def main():
        if metta.app_backend.config.run_leaderboard_updater:
            await leaderboard_updater.start()
        # Run uvicorn in a way that doesn't block
        config = uvicorn.Config(app, host=metta.app_backend.config.host, port=metta.app_backend.config.port)
        server = uvicorn.Server(config)
        await server.serve()

    asyncio.run(main())
