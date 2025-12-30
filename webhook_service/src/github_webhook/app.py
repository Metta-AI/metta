"""Standalone FastAPI app for GitHub webhook service."""

import logging
import sys

import fastapi
from fastapi.middleware.cors import CORSMiddleware

from github_webhook.routes import create_github_webhook_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def create_app() -> fastapi.FastAPI:
    """Create FastAPI app for GitHub webhook service."""
    app = fastapi.FastAPI(title="GitHub Webhook Service", version="1.0.0")

    # Add CORS middleware (allow all for webhook)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add webhook router
    webhook_router = create_github_webhook_router()
    app.include_router(webhook_router)

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "ok"}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


