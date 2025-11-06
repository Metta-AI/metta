import logging
import os

import fastapi
import fastapi.middleware.cors
import typing_extensions
import uvicorn

import metta.common.util.log_config
import metta.gridworks.routes.configs
import metta.gridworks.routes.schemas

logger = logging.getLogger(__name__)


def make_app():
    app = fastapi.FastAPI()

    app.add_middleware(
        fastapi.middleware.cors.CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class RepoRootResult(typing_extensions.TypedDict):
        repo_root: str

    @app.get("/repo-root")
    async def route_repo_root() -> RepoRootResult:
        return {
            "repo_root": os.getcwd(),
        }

    app.include_router(metta.gridworks.routes.configs.make_configs_router())
    app.include_router(metta.gridworks.routes.schemas.make_schemas_router())

    return app


def main() -> None:
    metta.common.util.log_config.init_logging()
    uvicorn.run(
        "metta.gridworks.server:make_app",
        port=8001,
        factory=True,
        reload=True,
    )


if __name__ == "__main__":
    main()
