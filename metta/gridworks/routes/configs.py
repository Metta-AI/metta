import logging

from fastapi import APIRouter, HTTPException

from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.gridworks.common import ErrorResult, dump_config_with_implicit_info
from metta.gridworks.configs.registry import ConfigMaker, ConfigMakerKind, ConfigMakerRegistry
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvalTool
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.train import TrainTool
from mettagrid.base_config import Config
from mettagrid.config import MettaGridConfig
from mettagrid.map_builder.map_builder import AnyMapBuilderConfig
from mettagrid.mapgen.utils.storable_map import StorableMap, StorableMapDict

logger = logging.getLogger(__name__)


def make_configs_router() -> APIRouter:
    router = APIRouter(prefix="/configs", tags=["configs"])

    registry = ConfigMakerRegistry()

    def get_config_maker_or_404(path: str) -> ConfigMaker:
        cfg = registry.get_by_path(path)
        if cfg is None:
            raise HTTPException(status_code=404, detail=f"Config {path} not found")
        return cfg

    def config_to_map_builder(cfg: Config | list[Config]) -> AnyMapBuilderConfig:
        if isinstance(cfg, MettaGridConfig):
            return cfg.game.map_builder
        if isinstance(cfg, SimulationConfig):
            return cfg.env.game.map_builder
        if isinstance(cfg, PlayTool) or isinstance(cfg, ReplayTool):
            return cfg.sim.env.game.map_builder
        if isinstance(cfg, CurriculumConfig):
            return cfg.make().get_task().get_env_cfg().game.map_builder
        if isinstance(cfg, TrainTool):
            return config_to_map_builder(cfg.training_env.curriculum)

        raise HTTPException(
            status_code=400, detail=f"Config of type {type(cfg)} can't be converted to a MapBuilderConfig"
        )

    def config_to_map_builder_by_name(cfg: Config | list[Config], name: str) -> AnyMapBuilderConfig:
        if isinstance(cfg, EvalTool):
            return config_to_map_builder_by_name(list(cfg.simulations), name)

        if isinstance(cfg, ReplayTool) or isinstance(cfg, PlayTool):
            return config_to_map_builder_by_name(cfg.sim, name)

        if not isinstance(cfg, list):
            raise HTTPException(status_code=400, detail="Config must be a list")

        for c in cfg:
            if not isinstance(c, SimulationConfig):
                raise HTTPException(status_code=400, detail="Config must be a list[SimulationConfig]")
            if c.name == name:
                return config_to_map_builder(c)

        raise HTTPException(status_code=404, detail=f"Config of type {type(cfg)} doesn't have map for name {name}")

    @router.get("/")
    async def get_configs() -> dict[ConfigMakerKind, list[dict]]:
        result: dict[ConfigMakerKind, list[dict]] = {
            kind: [e.to_dict() for e in cfgs] for kind, cfgs in registry.grouped_by_kind().items()
        }
        return result

    @router.get("/get")
    async def get_config(path: str) -> dict | ErrorResult:
        cfg = get_config_maker_or_404(path)
        return {
            "maker": cfg.to_dict(),
            "config": dump_config_with_implicit_info(cfg.maker()),
        }

    @router.get("/get-map")
    async def get_map(path: str, name: str | None = None) -> StorableMapDict | ErrorResult:
        cfg = get_config_maker_or_404(path)

        if name:
            map_builder_config = config_to_map_builder_by_name(cfg.maker(), name)
        else:
            map_builder_config = config_to_map_builder(cfg.maker())

        storable_map = StorableMap.from_cfg(map_builder_config)
        return storable_map.to_dict()

    return router
