import logging

import fastapi

import metta.cogworks.curriculum.curriculum
import metta.common.util.fs
import metta.gridworks.common
import metta.gridworks.configs.registry
import metta.sim.simulation_config
import metta.tools.eval
import metta.tools.play
import metta.tools.replay
import metta.tools.train
import mettagrid.base_config
import mettagrid.config
import mettagrid.map_builder.map_builder
import mettagrid.mapgen.utils.storable_map

logger = logging.getLogger(__name__)


def make_configs_router() -> fastapi.APIRouter:
    router = fastapi.APIRouter(prefix="/configs", tags=["configs"])

    repo_root = metta.common.util.fs.get_repo_root()
    registry = metta.gridworks.configs.registry.ConfigMakerRegistry(
        root_dirs=[
            repo_root / "experiments",
            repo_root / "packages/cogames/src/cogames",
        ]
    )

    def get_config_maker_or_404(path: str) -> metta.gridworks.configs.registry.ConfigMaker:
        cfg = registry.get_by_path(path)
        if cfg is None:
            raise fastapi.HTTPException(status_code=404, detail=f"Config {path} not found")
        return cfg

    def config_to_map_builder(
        cfg: mettagrid.base_config.Config | list[mettagrid.base_config.Config],
    ) -> mettagrid.map_builder.map_builder.MapBuilderConfig:
        if isinstance(cfg, mettagrid.config.MettaGridConfig):
            return cfg.game.map_builder
        if isinstance(cfg, metta.sim.simulation_config.SimulationConfig):
            return cfg.env.game.map_builder
        if isinstance(cfg, metta.tools.play.PlayTool) or isinstance(cfg, metta.tools.replay.ReplayTool):
            return cfg.sim.env.game.map_builder
        if isinstance(cfg, metta.cogworks.curriculum.curriculum.CurriculumConfig):
            return cfg.make().get_task().get_env_cfg().game.map_builder
        if isinstance(cfg, metta.tools.train.TrainTool):
            return config_to_map_builder(cfg.training_env.curriculum)

        raise fastapi.HTTPException(
            status_code=400, detail=f"Config of type {type(cfg)} can't be converted to a MapBuilderConfig"
        )

    def config_to_map_builder_by_name(
        cfg: mettagrid.base_config.Config
        | list[mettagrid.base_config.Config]
        | dict[str, mettagrid.base_config.Config],
        name: str,
    ) -> mettagrid.map_builder.map_builder.MapBuilderConfig:
        if isinstance(cfg, metta.tools.eval.EvaluateTool):
            return config_to_map_builder_by_name(list(cfg.simulations), name)

        if isinstance(cfg, metta.tools.replay.ReplayTool) or isinstance(cfg, metta.tools.play.PlayTool):
            return config_to_map_builder_by_name(cfg.sim, name)

        if isinstance(cfg, list):
            for c in cfg:
                if not isinstance(c, metta.sim.simulation_config.SimulationConfig):
                    raise fastapi.HTTPException(status_code=400, detail="Config must be a list[SimulationConfig]")
                if c.name == name:
                    return config_to_map_builder(c)
        elif isinstance(cfg, dict) and name in cfg:
            return config_to_map_builder(cfg[name])
        else:
            raise fastapi.HTTPException(status_code=400, detail="Config must be a list")

        raise fastapi.HTTPException(
            status_code=404, detail=f"Config of type {type(cfg)} doesn't have map for name {name}"
        )

    @router.get("/")
    async def get_configs() -> dict[metta.gridworks.configs.registry.ConfigMakerKind, list[dict]]:
        result: dict[metta.gridworks.configs.registry.ConfigMakerKind, list[dict]] = {
            kind: [e.to_dict() for e in cfgs] for kind, cfgs in registry.grouped_by_kind().items()
        }
        return result

    @router.get("/get")
    async def get_config(path: str) -> dict | metta.gridworks.common.ErrorResult:
        cfg = get_config_maker_or_404(path)
        return {
            "maker": cfg.to_dict(),
            "config": metta.gridworks.common.dump_config_with_implicit_info(cfg.maker()),
        }

    @router.get("/get-map")
    async def get_map(
        path: str, name: str | None = None
    ) -> mettagrid.mapgen.utils.storable_map.StorableMapDict | metta.gridworks.common.ErrorResult:
        cfg = get_config_maker_or_404(path)

        if name:
            map_builder_config = config_to_map_builder_by_name(cfg.maker(), name)
        else:
            map_builder_config = config_to_map_builder(cfg.maker())

        storable_map = mettagrid.mapgen.utils.storable_map.StorableMap.from_cfg(map_builder_config)
        return storable_map.to_dict()

    return router
