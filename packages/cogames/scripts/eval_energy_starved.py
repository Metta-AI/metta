from cogames.cogs_vs_clips.eval_missions import EnergyStarved, MACHINA_EVAL
from mettagrid.config.mettagrid_config import MettaGridConfig


def get_config() -> MettaGridConfig:
    mission = EnergyStarved()
    # Use the eval site's map builder; mission.instantiate will enforce its own map
    inst = mission.instantiate(MACHINA_EVAL.map_builder, num_cogs=1)
    return inst.make_env()
