from fastapi import APIRouter, HTTPException

from cogames.cli.mission import parse_variants
from cogames.cogs_vs_clips.mission import Mission, MissionVariant
from cogames.cogs_vs_clips.missions import MISSIONS
from cogames.cogs_vs_clips.variants import VARIANTS
from metta.gridworks.common import ConfigWithExtraInfo, extend_config
from mettagrid.mapgen.utils.storable_map import StorableMap, StorableMapDict


def _get_mission(site_name: str, mission_name: str, variants: str = "") -> Mission:
    mission = next(
        (mission for mission in MISSIONS if mission.site.name == site_name and mission.name == mission_name), None
    )
    if mission is None:
        raise HTTPException(status_code=404, detail=f"Mission {site_name}.{mission_name} not found")

    variant_names = [v for v in variants.split(",") if v]
    try:
        parsed_variants = parse_variants(variant_names)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    mission = mission.with_variants(parsed_variants)

    return mission


def make_cogames_routes() -> APIRouter:
    router = APIRouter(prefix="/cogames")

    @router.get("/missions")
    def get_missions() -> list[ConfigWithExtraInfo]:
        return [extend_config(mission) for mission in MISSIONS]

    @router.get("/missions/{site_name}.{mission_name}")
    def get_mission(site_name: str, mission_name: str, variants: str = "") -> ConfigWithExtraInfo:
        mission = _get_mission(site_name, mission_name, variants)
        return extend_config(mission)

    @router.get("/missions/{site_name}.{mission_name}/map")
    def get_mission_map(site_name: str, mission_name: str, variants: str = "") -> StorableMapDict:
        mission = _get_mission(site_name, mission_name, variants)
        env = mission.make_env()
        storable_map = StorableMap.from_cfg(env.game.map_builder)
        return storable_map.to_dict()

    @router.get("/missions/{site_name}.{mission_name}/env")
    def get_mission_env(site_name: str, mission_name: str, variants: str = "") -> ConfigWithExtraInfo:
        mission = _get_mission(site_name, mission_name, variants)
        env = mission.make_env()
        return extend_config(env)

    @router.get("/variants")
    def get_variants() -> list[MissionVariant]:
        return VARIANTS

    return router
