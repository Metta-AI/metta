import json

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import mettagrid.util.file
from metta.map.utils.s3utils import list_objects
from metta.map.utils.storable_map import StorableMap, index_storable_maps, map_builder_cfg_to_storable_map
from metta.util.mettagrid_cfgs import (
    MettagridCfgFile,
    MettagridCfgFileMetadata,
)
from metta.util.resolvers import register_resolvers


def make_app():
    register_resolvers()
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/stored-maps/dirs")
    async def route_stored_maps_dirs():
        return {
            "dirs": [
                # TODO - list all dirs in s3://softmax-public/maps/
                "s3://softmax-public/maps/test-collection",
            ]
        }

    @app.get("/stored-maps/find-maps")
    async def route_stored_maps_find_maps(dir: str, filter: str):
        filter_items: list[tuple[str, str]] = []
        for item in filter.split(","):
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"Invalid filter item: {item}")
            key, value = item.split("=")
            filter_items.append((key, value))

        map_files: list[str]

        if filter_items:
            map_index = mettagrid.util.file.read(f"{dir}/index.json")
            map_index = json.loads(map_index.decode("utf-8"))

            map_files_set: set[str] | None = None
            for key, value in filter_items:
                filter_files = map_index[key][value]
                map_files_set = map_files_set.intersection(set(filter_files)) if map_files_set else set(filter_files)

            map_files = list(map_files_set) if map_files_set else []
        else:
            map_files = [f for f in list_objects(dir) if f.endswith(".yaml")]

        return {
            "maps": map_files,
        }

    @app.get("/stored-maps/get-map")
    async def route_stored_maps_get_map(url: str):
        return StorableMap.from_uri(url).to_dict()

    @app.post("/stored-maps/index-dir")
    async def route_stored_maps_index_dir(dir: str):
        return index_storable_maps(dir)

    @app.get("/stored-maps/get-index")
    async def route_stored_maps_get_index(dir: str):
        return json.loads(mettagrid.util.file.read(f"{dir}/index.json").decode("utf-8"))

    @app.get("/mettagrid-cfgs")
    async def route_mettagrid_cfgs():
        metadata_by_kind = MettagridCfgFileMetadata.get_all()
        result = {kind: [e.to_dict() for e in cfgs] for kind, cfgs in metadata_by_kind.items()}
        return result

    @app.get("/mettagrid-cfgs/get")
    async def route_mettagrid_cfgs_get(path: str):
        cfg = MettagridCfgFile.from_path(path)
        return cfg.to_dict()

    @app.get("/mettagrid-cfgs/get-map")
    async def route_mettagrid_cfgs_get_map(path: str):
        cfg = MettagridCfgFile.from_path(path)

        try:
            map_cfg = cfg.get_map_cfg()
            storable_map = map_builder_cfg_to_storable_map(map_cfg)
        except Exception as e:
            return {
                "error": str(e),
            }

        return storable_map.to_dict()

    return app


def main():
    app = make_app()
    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()
