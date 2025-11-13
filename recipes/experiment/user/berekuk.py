import random

from mettagrid import MettaGridConfig
from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.scene import GridTransform
from mettagrid.mapgen.scenes.inline_ascii import InlineAscii


def mapgen_transform_demo() -> MettaGridConfig:
    mg_config = MettaGridConfig()
    mg_config.game.num_agents = 0
    mg_config.game.map_builder = MapGen.Config(
        num_agents=mg_config.game.num_agents,
        border_width=1,
        instance=InlineAscii.Config(
            data="""
.....
.#...
.###.
.....
            """,
            transform=random.choice(list(GridTransform)),
        ),
    )
    return mg_config
