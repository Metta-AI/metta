import logging
import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing_extensions import TypedDict

from metta.common.util.log_config import init_logging
from softmax.maptools.gridworks.routes.configs import make_configs_router
from softmax.maptools.gridworks.routes.stored_maps import make_stored_maps_router

logger = logging.getLogger(__name__)


def make_app():
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class RepoRootResult(TypedDict):
        repo_root: str

    @app.get("/repo-root")
    async def route_repo_root() -> RepoRootResult:
        return {
            "repo_root": os.getcwd(),
        }

    app.include_router(make_stored_maps_router())
    app.include_router(make_configs_router())

    return app


def main() -> None:
    init_logging()
    uvicorn.run(
        "softmax.maptools.gridworks.server:make_app",
        port=8001,
        factory=True,
        reload=True,
    )


if __name__ == "__main__":
    main()
