import logging

from fastapi import APIRouter, HTTPException

from metta.gridworks.common import ErrorResult, dump_config_with_implicit_info
from metta.gridworks.configs.registry import ConfigMakerKind, ConfigMakerRegistry
from mettagrid.mapgen.utils.storable_map import StorableMap, StorableMapDict
from mettagrid.mettagrid_config import MettaGridConfig

logger = logging.getLogger(__name__)


def make_configs_router() -> APIRouter:
    router = APIRouter(prefix="/configs", tags=["configs"])

    registry = ConfigMakerRegistry()

    @router.get("/")
    async def get_configs() -> dict[ConfigMakerKind, list[dict]]:
        result: dict[ConfigMakerKind, list[dict]] = {
            kind: [e.to_dict() for e in cfgs] for kind, cfgs in registry.grouped_by_kind().items()
        }
        return result

    @router.get("/get")
    async def get_config(path: str) -> dict | ErrorResult:
        try:
            cfg = registry.get_by_path(path)
            if cfg is None:
                raise HTTPException(status_code=404, detail=f"Config {path} not found")
            return {
                "maker": cfg.to_dict(),
                "config": dump_config_with_implicit_info(cfg.maker()),
            }
        except Exception as e:
            logger.error(f"Error getting mettagrid cfg for {path}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e)) from e

    @router.get("/get-map")
    async def get_map(path: str) -> StorableMapDict | ErrorResult:
        cfg = registry.get_by_path(path)
        if cfg is None:
            raise HTTPException(status_code=404, detail=f"Config {path} not found")

        if cfg.return_type != "MettaGridConfig":
            raise HTTPException(status_code=400, detail=f"Config {path} is not a MettaGrid")

        mettagrid_config = cfg.maker()
        if not isinstance(mettagrid_config, MettaGridConfig):
            raise HTTPException(status_code=400, detail=f"Config {path} is not a MettaGrid")

        try:
            storable_map = StorableMap.from_cfg(mettagrid_config.game.map_builder)
            return storable_map.to_dict()
        except Exception as e:
            logger.error(f"Error getting map for {path}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e)) from e

    return router
