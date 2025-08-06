#!/usr/bin/env -S uv run

import logging
import sys

import fastapi
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from pydantic.main import BaseModel

from metta.app_backend.auth import user_from_header_or_token
from metta.app_backend.metta_repo import MettaRepo
from metta.app_backend.routes import (
    dashboard_routes,
    entity_routes,
    eval_task_routes,
    scorecard_routes,
    sql_routes,
    stats_routes,
    sweep_routes,
    token_routes,
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
        allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Frontend URLs
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Create routers with the provided StatsRepo
    dashboard_router = dashboard_routes.create_dashboard_router(stats_repo)
    eval_task_router = eval_task_routes.create_eval_task_router(stats_repo)
    sql_router = sql_routes.create_sql_router(stats_repo)
    stats_router = stats_routes.create_stats_router(stats_repo)
    token_router = token_routes.create_token_router(stats_repo)
    policy_scorecard_router = scorecard_routes.create_policy_scorecard_router(stats_repo)
    sweep_router = sweep_routes.create_sweep_router(stats_repo)
    entity_router = entity_routes.create_entity_router(stats_repo)

    app.include_router(dashboard_router)
    app.include_router(eval_task_router)
    app.include_router(sql_router)
    app.include_router(stats_router)
    app.include_router(token_router)
    app.include_router(policy_scorecard_router, prefix="/scorecard")
    # TODO: remove this once we're confident we've migrated all clients to use the /scorecard prefix
    app.include_router(policy_scorecard_router, prefix="/heatmap")
    app.include_router(sweep_router)
    app.include_router(entity_router)

    @app.get("/whoami")
    async def whoami(request: fastapi.Request) -> WhoAmIResponse:
        user_id = await user_from_header_or_token(request, stats_repo)
        return WhoAmIResponse(user_email=user_id or "unknown")

    return app


if __name__ == "__main__":
    from metta.app_backend.config import host, port, stats_db_uri

    # Setup logging first
    # setup_logging()

    stats_repo = MettaRepo(stats_db_uri)
    app = create_app(stats_repo)

    uvicorn.run(app, host=host, port=port)
