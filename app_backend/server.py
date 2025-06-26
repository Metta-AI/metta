#!/usr/bin/env -S uv run

import fastapi
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from app_backend.auth import user_from_header_or_token
from app_backend.metta_repo import MettaRepo
from app_backend.routes import dashboard_routes, stats_routes, token_routes


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

    # Create routers with the provided StatsRepo
    dashboard_router = dashboard_routes.create_dashboard_router(stats_repo)
    stats_router = stats_routes.create_stats_router(stats_repo)
    token_router = token_routes.create_token_router(stats_repo)

    app.include_router(dashboard_router)
    app.include_router(stats_router)
    app.include_router(token_router)

    @app.get("/whoami")
    def whoami(request: fastapi.Request):  # type: ignore
        user_id = user_from_header_or_token(request, stats_repo)
        return {"user_email": user_id or "unknown"}

    return app


if __name__ == "__main__":
    from app_backend.config import host, port, stats_db_uri

    stats_repo = MettaRepo(stats_db_uri)
    app = create_app(stats_repo)

    uvicorn.run(app, host=host, port=port)
