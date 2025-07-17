import json
import logging
import os
from urllib.parse import unquote

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing_extensions import TypedDict

from metta.common.util.mettagrid_cfgs import (
    CfgKind,
    MettagridCfgFile,
    MettagridCfgFileMetadata,
)
from metta.common.util.resolvers import register_resolvers
from metta.map.utils.storable_map import StorableMap, StorableMapDict
from metta.map.utils.storable_map_index import StorableMapIndex
from metta.mettagrid.util.file import read
from metta.util.init.logging import init_logging

logger = logging.getLogger("metta.gridworks.server")


class ErrorResult(TypedDict):
    error: str


def make_app():
    init_logging()
    register_resolvers()
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class StoredMapsDirsResult(TypedDict):
        dirs: list[str]

    @app.get("/stored-maps/dirs")
    async def route_stored_maps_dirs() -> StoredMapsDirsResult:
        return {
            "dirs": [
                # TODO - list all dirs in s3://softmax-public/maps/
                "s3://softmax-public/maps/test-collection",
            ]
        }

    class StoredMapsFindMapsResult(TypedDict):
        maps: list[str]

    @app.get("/stored-maps/find-maps")
    async def route_stored_maps_find_maps(dir: str, filter: str) -> StoredMapsFindMapsResult:
        print(filter)
        filter_items: list[tuple[str, str]] = []
        for item in filter.split(","):
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"Invalid filter item: {item}")
            key, value = item.split("=")
            filter_items.append((key, unquote(value)))

        index = StorableMapIndex.load(dir)
        map_files = index.find_maps(filter_items)

        return {
            "maps": map_files,
        }

    @app.get("/stored-maps/get-map")
    async def route_stored_maps_get_map(url: str) -> StorableMapDict:
        return StorableMap.from_uri(url).to_dict()

    class StoredMapsIndexDirResult(TypedDict):
        success: bool

    @app.post("/stored-maps/index-dir")
    async def route_stored_maps_index_dir(dir: str) -> StoredMapsIndexDirResult:
        StorableMapIndex.create(dir)
        return {"success": True}

    @app.get("/stored-maps/get-index")
    async def route_stored_maps_get_index(dir: str) -> dict:
        return json.loads(read(f"{dir}/index.json").decode("utf-8"))

    @app.get("/mettagrid-cfgs")
    async def route_mettagrid_cfgs() -> dict[CfgKind, list[dict]]:
        metadata_by_kind = MettagridCfgFileMetadata.get_all()

        result: dict[CfgKind, list[dict]] = {
            kind: [e.to_dict() for e in cfgs] for kind, cfgs in metadata_by_kind.items()
        }
        return result

    @app.get("/mettagrid-cfgs/get")
    async def route_mettagrid_cfgs_get(path: str) -> MettagridCfgFile.AsDict | ErrorResult:
        try:
            cfg = MettagridCfgFile.from_path(path)
            return cfg.to_dict()
        except Exception as e:
            logger.error(f"Error getting mettagrid cfg for {path}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/mettagrid-cfgs/get-map")
    async def route_mettagrid_cfgs_get_map(path: str) -> StorableMapDict | ErrorResult:
        cfg = MettagridCfgFile.from_path(path)

        try:
            map_cfg = cfg.get_map_cfg()
            storable_map = StorableMap.from_cfg(map_cfg)
            return storable_map.to_dict()
        except Exception as e:
            logger.error(f"Error getting map for {path}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e)) from e

    class RepoRootResult(TypedDict):
        repo_root: str

    @app.get("/repo-root")
    async def route_repo_root() -> RepoRootResult:
        return {
            "repo_root": os.getcwd(),
        }

    return app


def main() -> None:
    uvicorn.run(
        "metta.gridworks.server:make_app",
        port=8001,
        factory=True,
        reload=True,
    )


if __name__ == "__main__":
    main()
