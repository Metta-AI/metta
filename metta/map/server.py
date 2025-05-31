import json
import os

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from omegaconf import DictConfig, OmegaConf
from pufferlib.utils import unroll_nested_dict

import mettagrid.util.file
from metta.map.utils.s3utils import get_s3_client, list_objects, parse_s3_uri
from metta.map.utils.storable_map import StorableMap


def omegaconf_to_key_values(cfg: DictConfig, prefix: str) -> dict[str, str]:
    key_to_value: dict[str, str] = {}
    for k, v in unroll_nested_dict(OmegaConf.to_container(cfg, resolve=False)):
        key_to_value[f"{prefix}.{str(k).replace('/', '.')}"] = str(v)
    return key_to_value


class StorableMapIndex:
    index: dict[str, dict[str, list[str]]] = {}

    def process(self, map: StorableMap, uri: str):
        key_to_value = omegaconf_to_key_values(map.config, "config")
        key_to_value.update({f"metadata.{k}": v for k, v in map.metadata.items()})
        for key, value in key_to_value.items():
            if key not in self.index:
                self.index[key] = {}
            if value not in self.index[key]:
                self.index[key][value] = []
            self.index[key][value].append(uri)

    def index_dir(self, dir: str):
        filenames = list_objects(dir)
        for filename in filenames:
            map = StorableMap.from_uri(filename)
            self.process(map, filename)


def index_storable_maps(dir: str):
    indexer = StorableMapIndex()
    indexer.index_dir(dir)
    index_uri = f"{dir}/index.json"
    mettagrid.util.file.write_data(index_uri, json.dumps(indexer.index), content_type="text/plain")


def make_app():
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/stored-maps/dirs")
    async def get_stored_maps_dirs():
        return {
            "dirs": [
                # TODO
                "s3://softmax-public/maps/test-collection",
            ]
        }

    @app.get("/stored-maps/find-maps")
    async def find_stored_maps(dir: str, filter: str):
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
    async def get_stored_map(url: str):
        bucket, key = parse_s3_uri(url)
        obj = get_s3_client().get_object(Bucket=bucket, Key=key)
        return {
            "content": obj["Body"].read().decode("utf-8"),
        }

    @app.post("/stored-maps/index-dir")
    async def index_dir(dir: str):
        return index_storable_maps(dir)

    @app.get("/stored-maps/get-index")
    async def get_index(dir: str):
        return json.loads(mettagrid.util.file.read(f"{dir}/index.json").decode("utf-8"))

    @app.get("/envs")
    async def list_envs():
        envs = []
        for root, _, files in os.walk("configs/env/mettagrid"):
            for f in files:
                if f.endswith((".yaml", ".yml")):
                    rel_path = os.path.relpath(os.path.join(root, f), "configs/env/mettagrid")
                    envs.append(rel_path)
        return {"envs": envs}

    @app.get("/envs/get")
    async def get_env(name: str):
        # Try with .yaml extension first, then .yml
        for ext in [".yaml", ".yml"]:
            env_path = os.path.join("configs/env/mettagrid", f"{name}{ext}")
            if os.path.isfile(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    content = f.read()
                return {"content": content}

        raise HTTPException(status_code=404, detail=f"Environment '{name}' not found")

    return app


def main():
    app = make_app()
    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()
