import json
import os
from dataclasses import dataclass
from typing import Literal

import hydra
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from omegaconf import DictConfig, OmegaConf
from pufferlib.utils import unroll_nested_dict

import mettagrid.util.file
from metta.map.utils.s3utils import get_s3_client, list_objects, parse_s3_uri
from metta.map.utils.storable_map import StorableMap, map_builder_cfg_to_storable_map
from metta.util.config import config_from_path
from metta.util.resolvers import register_resolvers


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


@dataclass
class MettagridCfgData:
    path: str
    cfg: DictConfig
    kind: Literal["env", "curriculum", "map", "unknown"]

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "kind": self.kind,
            "cfg": OmegaConf.to_container(self.cfg, resolve=False),
        }


mettagrid_cfg_root = "env/mettagrid"


def get_mettagrid_cfg_data(path: str) -> MettagridCfgData:
    with hydra.initialize(config_path="../../configs", version_base=None):
        cfg = config_from_path(mettagrid_cfg_root + "/" + path)
        if not isinstance(cfg, DictConfig):
            raise ValueError(f"Invalid config type: {type(cfg)}")

    cfg_kind = "unknown"
    if cfg.get("game") and cfg.game.get("map_builder"):
        cfg_kind = "env"
    elif "curriculum" in cfg.get("_target_", ""):
        cfg_kind = "curriculum"  # TODO - check if this is correct
    elif "metta.map" in cfg.get("_target_", "") or "mettagrid.room" in cfg.get("_target_", ""):
        cfg_kind = "map"

    return MettagridCfgData(path=path, cfg=cfg, kind=cfg_kind)


def get_mettagrid_cfgs():
    cfgs: list[MettagridCfgData] = []

    for root, _, files in os.walk("configs/" + mettagrid_cfg_root):
        for f in files:
            if not f.endswith(".yaml"):
                continue
            path = os.path.relpath(os.path.join(root, f), "configs/" + mettagrid_cfg_root)
            cfgs.append(get_mettagrid_cfg_data(path))

    return cfgs


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
                # TODO
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
        bucket, key = parse_s3_uri(url)
        obj = get_s3_client().get_object(Bucket=bucket, Key=key)
        return {
            "content": obj["Body"].read().decode("utf-8"),
        }

    @app.post("/stored-maps/index-dir")
    async def route_stored_maps_index_dir(dir: str):
        return index_storable_maps(dir)

    @app.get("/stored-maps/get-index")
    async def route_stored_maps_get_index(dir: str):
        return json.loads(mettagrid.util.file.read(f"{dir}/index.json").decode("utf-8"))

    @app.get("/mettagrid-cfgs")
    async def route_mettagrid_cfgs():
        cfgs = get_mettagrid_cfgs()
        # group by kind
        cfgs_by_kind: dict[Literal["env", "curriculum", "map", "unknown"], list[MettagridCfgData]] = {}
        for cfg in cfgs:
            if cfg.kind not in cfgs_by_kind:
                cfgs_by_kind[cfg.kind] = []
            cfgs_by_kind[cfg.kind].append(cfg)

        result = {kind: [e.to_dict() for e in cfgs] for kind, cfgs in cfgs_by_kind.items()}
        print(result)
        return result

    @app.get("/mettagrid-cfgs/get")
    async def route_mettagrid_cfgs_get(path: str):
        cfg = get_mettagrid_cfg_data(path)
        return cfg.to_dict()

    @app.get("/mettagrid-cfgs/get-map")
    async def route_mettagrid_cfgs_get_map(path: str):
        cfg = get_mettagrid_cfg_data(path)
        if cfg.kind == "map":
            storable_map = map_builder_cfg_to_storable_map(cfg.cfg)
        elif cfg.kind == "env":
            storable_map = map_builder_cfg_to_storable_map(cfg.cfg.game.map_builder)
        else:
            raise ValueError(f"Config {path} is not a map or env")

        return {
            "content": str(storable_map),
        }

    return app


def main():
    app = make_app()
    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()
