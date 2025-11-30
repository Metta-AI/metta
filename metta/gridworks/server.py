import logging
import os

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing_extensions import TypedDict

from metta.common.util.log_config import init_logging
from metta.gridworks.routes.cogames import make_cogames_routes
from metta.gridworks.routes.configs import make_configs_router
from metta.gridworks.routes.schemas import make_schemas_router
from mettagrid.mapgen.utils.ascii_grid import default_char_to_name

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

    # Add this exception handler to catch all unhandled exceptions
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled exception in {request.method} {request.url.path}")
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc)},
            headers={
                "Access-Control-Allow-Origin": "http://localhost:3000",
                "Access-Control-Allow-Credentials": "true",
            },
        )

    class RepoRootResult(TypedDict):
        repo_root: str

    @app.get("/repo-root")
    async def route_repo_root() -> RepoRootResult:
        return {
            "repo_root": os.getcwd(),
        }

    @app.get("/mettagrid-encoding")
    async def route_mettagrid_encoding() -> dict[str, str]:
        return default_char_to_name()

    app.include_router(make_configs_router())
    app.include_router(make_schemas_router())
    app.include_router(make_cogames_routes())

    return app


def main() -> None:
    init_logging()
    uvicorn.run(
        "metta.gridworks.server:make_app",
        port=8001,
        factory=True,
        reload=True,
    )


if __name__ == "__main__":
    main()
