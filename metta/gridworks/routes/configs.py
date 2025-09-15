import logging

from fastapi import APIRouter, HTTPException

from metta.gridworks.common import ErrorResult
from metta.gridworks.configs import ConfigMaker, ConfigMakerKind, ConfigMakerRegistry
from metta.mettagrid.mapgen.utils.storable_map import StorableMap, StorableMapDict
from metta.mettagrid.mettagrid_config import MettaGridConfig

logger = logging.getLogger(__name__)


def make_configs_router() -> APIRouter:
    router = APIRouter(prefix="/configs", tags=["configs"])

    @router.get("/")
    async def get_configs() -> dict[ConfigMakerKind, list[dict]]:
        registry = ConfigMakerRegistry()

        result: dict[ConfigMakerKind, list[dict]] = {
            kind: [e.to_dict() for e in cfgs] for kind, cfgs in registry.grouped_by_kind().items()
        }
        return result

    @router.get("/get")
    async def get_config(path: str) -> dict | ErrorResult:
        try:
            cfg = ConfigMaker.from_path(path)
            return {
                "metadata": cfg.to_dict(),
                "config": cfg._maker(),
            }
        except Exception as e:
            logger.error(f"Error getting mettagrid cfg for {path}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.get("/get-map")
    async def get_map(path: str) -> StorableMapDict | ErrorResult:
        cfg = ConfigMaker.from_path(path)

        if cfg.kind() != "MettaGrid":
            raise HTTPException(status_code=400, detail=f"Config {path} is not a MettaGrid")

        mettagrid_config = cfg._maker()
        print(mettagrid_config)
        if not isinstance(mettagrid_config, MettaGridConfig):
            raise HTTPException(status_code=400, detail=f"Config {path} is not a MettaGrid")

        try:
            storable_map = StorableMap.from_cfg(mettagrid_config.game.map_builder)
            return storable_map.to_dict()
        except Exception as e:
            logger.error(f"Error getting map for {path}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e)) from e

    return router
