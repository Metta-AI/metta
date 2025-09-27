from cogames.cogs_vs_clips.scenarios import machina_symmetry_sanctum
from mettagrid.config.mettagrid_config import MettaGridConfig


def make_mettagrid() -> MettaGridConfig:
    """MettaGridConfig

    Gridworks Config Maker: sanctum with enforced horizontal+vertical symmetry.
    """
    return machina_symmetry_sanctum(num_cogs=4)
