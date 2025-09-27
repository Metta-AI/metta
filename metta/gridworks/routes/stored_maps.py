import json
import logging
from urllib.parse import unquote

from fastapi import APIRouter
from typing_extensions import TypedDict

from softmax.lib.utils import file as file_utils
from mettagrid.mapgen.utils.storable_map import StorableMap, StorableMapDict
from mettagrid.mapgen.utils.storable_map_index import StorableMapIndex

logger = logging.getLogger(__name__)


def make_stored_maps_router() -> APIRouter:
    router = APIRouter(prefix="/stored-maps", tags=["stored-maps"])

    class StoredMapsDirsResult(TypedDict):
        dirs: list[str]

    @router.get("/dirs")
    def get_dirs() -> StoredMapsDirsResult:
        return {
            "dirs": [
                # TODO - list all dirs in s3://softmax-public/maps/
                "s3://softmax-public/maps/test-collection",
            ]
        }

    class StoredMapsFindMapsResult(TypedDict):
        maps: list[str]

    @router.get("/find-maps")
    async def get_find_maps(dir: str, filter: str) -> StoredMapsFindMapsResult:
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

    @router.get("/get-map")
    async def get_get_map(url: str) -> StorableMapDict:
        return StorableMap.from_uri(url).to_dict()

    class StoredMapsIndexDirResult(TypedDict):
        success: bool

    @router.post("/index-dir")
    async def get_index_dir(dir: str) -> StoredMapsIndexDirResult:
        StorableMapIndex.create(dir)
        return {"success": True}

    @router.get("/get-index")
    async def get_get_index(dir: str) -> dict:
        return json.loads(file_utils.read(f"{dir}/index.json").decode("utf-8"))

    return router
