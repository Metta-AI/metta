#!/usr/bin/env -S uv run

import fastapi
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from metta.app.metta_repo import MettaRepo
from metta.app.routes import dashboard_routes, stats_routes


def create_app(stats_repo: MettaRepo) -> fastapi.FastAPI:
    """Create a FastAPI app with the given StatsRepo instance."""
    app = fastapi.FastAPI()

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Frontend URLs
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # TODO: add auth middleware

    # Create routers with the provided StatsRepo
    dashboard_router = dashboard_routes.create_dashboard_router(stats_repo)
    stats_router = stats_routes.create_stats_router(stats_repo)

    app.include_router(dashboard_router)
    app.include_router(stats_router)

    return app


if __name__ == "__main__":
    from metta.app import config

    stats_repo = MettaRepo(config.stats_db_uri)
    app = create_app(stats_repo)

    uvicorn.run(app, host=config.host, port=config.port)
