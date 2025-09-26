from cogames.cogs_vs_clips.scenarios import machina_sanctum
from mettagrid.config.mettagrid_config import MettaGridConfig


def make_mettagrid() -> MettaGridConfig:
    """MettaGridConfig

    Gridworks Config Maker: returns the Sanctum-in-the-Quadrants map with central base.
    """
    return machina_sanctum(num_cogs=4)
